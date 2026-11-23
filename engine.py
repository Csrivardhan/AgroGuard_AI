from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agronomy_kb import get_regional_intelligence

ART = ROOT / "artifacts"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@lru_cache(maxsize=1)
def load_artifacts():
    required = [
        ART / "calibrated_crop_model.joblib",
        ART / "arena_models.joblib",
        ART / "anomaly_detector.joblib",
        ART / "metadata.json",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact {p}. Run `python src/train.py` first.")

    champion = joblib.load(required[0])
    arena = joblib.load(required[1])
    detector = joblib.load(required[2])
    meta = json.loads(required[3].read_text(encoding="utf-8"))

    for cal_est in getattr(champion, "calibrated_classifiers_", []):
        est = getattr(cal_est, "estimator", None)
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
    for m in arena.values():
        for cal_est in getattr(m, "calibrated_classifiers_", []):
            est = getattr(cal_est, "estimator", None)
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1

    return champion, arena, detector, meta


def validate_inputs(values: Dict[str, float]) -> List[str]:
    errors = []
    for f in FEATURES:
        if f not in values or values[f] is None:
            errors.append(f"Missing required feature: {f}")
    if errors:
        return errors

    for f in ["N", "P", "K", "rainfall"]:
        if values[f] < 0:
            errors.append(f"{f} cannot be negative")
    if not 0 <= values["humidity"] <= 100:
        errors.append("Humidity must be 0–100%")
    if not 0 <= values["ph"] <= 14:
        errors.append("pH must be 0–14")
    if not -50 <= values["temperature"] <= 60:
        errors.append("Temperature outside broad sanity range (-50°C to 60°C)")
    return errors


def compute_decision_autopsy(
    champion,
    values: Dict[str, float],
    meta: Dict[str, Any],
    top_crop: str,
    runner_up_crop: str,
    base_probs: np.ndarray,
) -> Dict[str, Any]:
    classes = list(champion.classes_)
    top_idx = classes.index(top_crop)
    runner_idx = classes.index(runner_up_crop)
    base_top_p = float(base_probs[top_idx])

    medians = meta["training_median"]
    profiles = meta["crop_profiles"][top_crop]

    base_arr = np.array([[values[f] for f in FEATURES]], dtype=float)
    batch_arr = np.repeat(base_arr, len(FEATURES), axis=0)
    for idx, f in enumerate(FEATURES):
        batch_arr[idx, idx] = float(medians[f])

    batch_df = pd.DataFrame(batch_arr, columns=FEATURES)
    batch_probs = champion.predict_proba(batch_df)[:, top_idx]

    contributions = []
    for idx, f in enumerate(FEATURES):
        new_p = float(batch_probs[idx])
        delta_prob = round((base_top_p - new_p) * 100, 2)

        prof = profiles[f]
        in_iqr = prof["q25"] <= values[f] <= prof["q75"]
        z_dev = round((values[f] - prof["median"]) / (prof["std"] if prof["std"] > 0 else 1.0), 2)

        contributions.append({
            "feature": f,
            "input_value": values[f],
            "crop_ideal_iqr": [prof["q25"], prof["q75"]],
            "within_ideal_iqr": bool(in_iqr),
            "probability_impact_percent": delta_prob,
            "direction": "SUPPORTS" if delta_prob > 0.5 else ("OPPOSES" if delta_prob < -0.5 else "NEUTRAL"),
            "z_deviation_from_crop_median": z_dev,
        })

    contributions.sort(key=lambda x: abs(x["probability_impact_percent"]), reverse=True)
    margin_percent = round((float(base_probs[top_idx]) - float(base_probs[runner_idx])) * 100, 2)
    return {
        "target_crop": top_crop,
        "runner_up_crop": runner_up_crop,
        "confidence_margin_over_runner_up_percent": margin_percent,
        "primary_drivers": [c["feature"] for c in contributions[:3] if c["probability_impact_percent"] > 0],
        "feature_attributions": contributions,
    }


def compute_model_arena(
    arena_models: Dict[str, Any],
    row: pd.DataFrame,
    meta: Dict[str, Any],
    champion_crop: str,
) -> Dict[str, Any]:
    live_votes = []
    vote_counts: Dict[str, int] = {}

    for model_name, est in arena_models.items():
        probs = est.predict_proba(row)[0]
        idx = int(np.argmax(probs))
        pred_crop = str(est.classes_[idx])
        conf = round(float(probs[idx] * 100), 2)
        vote_counts[pred_crop] = vote_counts.get(pred_crop, 0) + 1
        live_votes.append({
            "model": model_name,
            "predicted_crop": pred_crop,
            "calibrated_confidence_percent": conf,
            "agrees_with_champion": pred_crop == champion_crop,
        })

    total_models = len(live_votes)
    agreeing = sum(1 for v in live_votes if v["agrees_with_champion"])
    consensus_ratio = round(agreeing / total_models, 2)

    return {
        "consensus_score": f"{agreeing}/{total_models}",
        "consensus_ratio": consensus_ratio,
        "disagreement_detected": agreeing < total_models,
        "vote_distribution": vote_counts,
        "per_model_predictions": live_votes,
        "benchmark_leaderboard": meta.get("arena_leaderboard", []),
    }


def compute_compatibility_frontier(
    values: Dict[str, float],
    meta: Dict[str, Any],
    candidate_crops: List[str],
) -> Dict[str, Any]:
    profiles = meta["crop_profiles"]
    frontier = []

    for crop in candidate_crops:
        if crop not in profiles:
            continue
        prof = profiles[crop]
        feature_scores = []
        limiting_factors = []

        for f in FEATURES:
            val = values[f]
            q25, q75 = prof[f]["q25"], prof[f]["q75"]
            fmin, fmax = prof[f]["min"], prof[f]["max"]
            iqr = max(q75 - q25, 1e-3)

            if q25 <= val <= q75:
                fit = 100.0
                status = "OPTIMAL"
            elif fmin <= val <= fmax:
                dist = (q25 - val) if val < q25 else (val - q75)
                fit = max(50.0, round(100.0 - (dist / iqr) * 20.0, 1))
                status = "ACCEPTABLE"
            else:
                dist = (fmin - val) if val < fmin else (val - fmax)
                fit = max(0.0, round(50.0 - (dist / iqr) * 25.0, 1))
                status = "OUT_OF_RANGE"
                limiting_factors.append(f"{f} ({val} vs ideal {q25}–{q75})")

            feature_scores.append({
                "feature": f,
                "user_value": val,
                "ideal_range_iqr": [q25, q75],
                "observed_range": [fmin, fmax],
                "fit_score": fit,
                "status": status,
            })

        overall_suitability = round(float(np.mean([fs["fit_score"] for fs in feature_scores])), 1)
        frontier.append({
            "crop": crop,
            "agronomic_suitability_score": overall_suitability,
            "limiting_factors": limiting_factors,
            "feature_breakdown": feature_scores,
        })

    frontier.sort(key=lambda x: x["agronomic_suitability_score"], reverse=True)
    return {
        "evaluated_candidates": frontier,
        "best_agronomic_fit": frontier[0]["crop"] if frontier else None,
    }


def compute_soil_prescription(
    values: Dict[str, float],
    meta: Dict[str, Any],
    top_crop: str,
) -> Dict[str, Any]:
    prof = meta["crop_profiles"][top_crop]
    ha_to_acre = 1.0 / 2.471

    n_gap_acre = round(max(0.0, prof["N"]["median"] - values["N"]) * ha_to_acre, 1)
    p_gap_acre = round(max(0.0, prof["P"]["median"] - values["P"]) * ha_to_acre, 1)
    k_gap_acre = round(max(0.0, prof["K"]["median"] - values["K"]) * ha_to_acre, 1)

    dap_kg = round(p_gap_acre / 0.46, 1) if p_gap_acre > 0 else 0.0
    n_from_dap = dap_kg * 0.18
    remaining_n = max(0.0, n_gap_acre - n_from_dap)
    urea_kg = round(remaining_n / 0.46, 1) if remaining_n > 0 else 0.0
    mop_kg = round(k_gap_acre / 0.60, 1) if k_gap_acre > 0 else 0.0

    urea_bags = round(urea_kg / 45.0, 2)
    dap_bags = round(dap_kg / 50.0, 2)
    mop_bags = round(mop_kg / 50.0, 2)
    est_cost_inr = int(round(urea_bags * 266.5 + dap_bags * 1350.0 + mop_bags * 1650.0, 0))

    ph_val = values["ph"]
    ph_med = prof["ph"]["median"]
    if ph_val < prof["ph"]["q25"]:
        ph_action = f"Acidic soil (pH {ph_val} vs ideal {ph_med}): Apply Agricultural Lime (~200 kg/acre) before sowing."
    elif ph_val > prof["ph"]["q75"]:
        ph_action = f"Alkaline soil (pH {ph_val} vs ideal {ph_med}): Apply Gypsum (~250 kg/acre) + compost."
    else:
        ph_action = f"Soil pH ({ph_val}) is within ideal range for {top_crop}."

    return {
        "target_crop": top_crop,
        "nutrient_deficit_kg_per_acre": {"N": n_gap_acre, "P": p_gap_acre, "K": k_gap_acre},
        "fertilizer_prescription_per_acre": {
            "urea_46N_kg": urea_kg,
            "urea_bags_45kg": urea_bags,
            "dap_18N_46P_kg": dap_kg,
            "dap_bags_50kg": dap_bags,
            "mop_60K_kg": mop_kg,
            "mop_bags_50kg": mop_bags,
        },
        "estimated_input_cost_inr_per_acre": est_cost_inr,
        "soil_ph_amendment": ph_action,
        "voice_summaries": {
            "en": (
                f"Recommended crop is {top_crop}. Per acre estimate: {urea_bags} bags Urea, "
                f"{dap_bags} bags DAP, and {mop_bags} bags MOP, costing around {est_cost_inr} rupees."
            ),
            "hi": (
                f"अनुशंसित फसल {top_crop} है। प्रति एकड़ {urea_bags} बोरी यूरिया, "
                f"{dap_bags} बोरी डीएपी और {mop_bags} बोरी पोटाश का अनुमान है, लागत लगभग {est_cost_inr} रुपये है।"
            ),
            "te": (
                f"సిఫార్సు చేయబడిన పంట {top_crop}. ఎకరాకు సుమారు {urea_bags} బస్తాల యూరియా, "
                f"{dap_bags} బస్తాల డీఏపీ మరియు {mop_bags} బస్తాల పొటాష్ అవసరం, అంచనా ఖర్చు {est_cost_inr} రూపాయలు."
            ),
        },
    }


def build_decision_passport(
    values: Dict[str, float],
    meta: Dict[str, Any],
    trust_block: Dict[str, Any],
    autopsy_block: Dict[str, Any],
    arena_block: Dict[str, Any],
) -> Dict[str, Any]:
    issued_at = datetime.now(timezone.utc).isoformat()
    passport_id = f"AGP-{uuid.uuid4().hex[:10].upper()}"
    payload_str = json.dumps({"inputs": values, "trust": trust_block, "model": meta["model_version"]}, sort_keys=True)
    digest = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:16]

    return {
        "passport_id": passport_id,
        "issued_at_utc": issued_at,
        "integrity_sha256_prefix": digest,
        "model_version": meta["model_version"],
        "champion_algorithm": meta["selected_model"],
        "recorded_inputs": values,
        "verdict": {
            "decision_status": trust_block["decision_status"],
            "recommended_crop": trust_block["top_prediction"],
            "calibrated_confidence_percent": trust_block["top_prediction_probability_percent"],
            "conformal_set": trust_block["conformal_prediction_set"],
            "anomaly_status": trust_block["anomaly_status"],
            "model_consensus": arena_block["consensus_score"],
            "key_supporting_features": autopsy_block["primary_drivers"],
        },
        "governance_limitations": meta.get("limitations", []),
    }


def run_agroguard_pipeline(
    values: Dict[str, float],
    season: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    errors = validate_inputs(values)
    if errors:
        return {"status": "INVALID_INPUT", "errors": errors}

    champion, arena_models, detector, meta = load_artifacts()
    clean_vals = {f: float(values[f]) for f in FEATURES}
    row = pd.DataFrame([[clean_vals[f] for f in FEATURES]], columns=FEATURES)

    probs = champion.predict_proba(row)[0]
    classes = champion.classes_
    order = np.argsort(probs)[::-1]

    top_three = [
        {"crop": str(classes[i]), "calibrated_probability_percent": round(float(probs[i] * 100), 2)}
        for i in order[:3]
    ]
    top_crop = top_three[0]["crop"]
    runner_up = top_three[1]["crop"]

    qhat = float(meta["conformal_lac_qhat"])
    pred_set = [str(classes[i]) for i, p in enumerate(probs) if (1.0 - p) <= qhat]

    range_warnings = []
    for f in FEATURES:
        if clean_vals[f] < meta["training_min"][f] or clean_vals[f] > meta["training_max"][f]:
            range_warnings.append(f"{f} outside observed training range")

    flag = int(detector.predict(row)[0])
    anomaly_score = float(detector.decision_function(row)[0])
    review_required = bool(range_warnings) or flag == -1 or len(pred_set) != 1

    if len(pred_set) == 1:
        conformal_status = "SINGLETON_CERTAIN"
    elif len(pred_set) > 1:
        conformal_status = "AMBIGUOUS_MULTI_CROP_SET"
    else:
        conformal_status = "EMPTY_HIGH_UNCERTAINTY"

    trust_engine = {
        "decision_status": "REVIEW_REQUIRED" if review_required else "MODEL_OUTPUT_AVAILABLE",
        "top_prediction": top_crop,
        "top_prediction_probability_percent": top_three[0]["calibrated_probability_percent"],
        "top_three": top_three,
        "conformal_prediction_set": pred_set,
        "conformal_set_status": conformal_status,
        "conformal_nominal_coverage": 1.0 - float(meta["conformal_alpha"]),
        "range_warnings": range_warnings,
        "anomaly_status": "POTENTIAL_ANOMALY" if flag == -1 else "NOT_FLAGGED",
        "anomaly_decision_score": round(anomaly_score, 6),
    }

    autopsy = compute_decision_autopsy(champion, clean_vals, meta, top_crop, runner_up, probs)
    arena = compute_model_arena(arena_models, row, meta, top_crop)

    candidate_crops = list(dict.fromkeys([c["crop"] for c in top_three] + list(arena["vote_distribution"].keys())))
    frontier = compute_compatibility_frontier(clean_vals, meta, candidate_crops)
    regional_intel = get_regional_intelligence(top_crop, season_hint=season, region_hint=region)
    farmer_plan = compute_soil_prescription(clean_vals, meta, top_crop)
    passport = build_decision_passport(clean_vals, meta, trust_engine, autopsy, arena)

    return {
        "status": "SUCCESS",
        "trust_engine": trust_engine,
        "decision_autopsy": autopsy,
        "model_challenge_arena": arena,
        "crop_compatibility_frontier": frontier,
        "regional_agricultural_intelligence": regional_intel,
        "farmer_action_plan": farmer_plan,
        "decision_passport": passport,
        "important_note": "Benchmark ML output only; validate local season, soil, water and crop requirements with agronomic expertise.",
    }


if __name__ == "__main__":
    sample = run_agroguard_pipeline(
        {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9},
        season="Kharif",
        region="Telangana",
    )
    print("Top Crop:", sample["trust_engine"]["top_prediction"])
    print("Farmer Plan:", sample["farmer_action_plan"]["fertilizer_prescription_per_acre"])