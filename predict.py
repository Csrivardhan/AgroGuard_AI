from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import FEATURES, run_agroguard_pipeline


def main():
    print("AGROGUARD AI v2.0 CLI Prediction & Audit Test")
    vals = {}
    for f in FEATURES:
        while True:
            try:
                vals[f] = float(input(f"Enter {f}: "))
                break
            except ValueError:
                print("Enter a valid number.")

    season = input("Enter season (Kharif/Rabi/Zaid or press Enter to skip): ").strip() or None
    region = input("Enter Indian state/region (or press Enter to skip): ").strip() or None

    result = run_agroguard_pipeline(vals, season=season, region=region)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts"
FEATURES=["N","P","K","temperature","humidity","ph","rainfall"]

def load():
    paths=[ART/"calibrated_crop_model.joblib",ART/"anomaly_detector.joblib",ART/"metadata.json"]
    for p in paths:
        if not p.exists(): raise FileNotFoundError(f"{p} missing. Run python src/train.py first.")
    return joblib.load(paths[0]),joblib.load(paths[1]),json.loads(paths[2].read_text(encoding="utf-8"))

def predict(values):
    model,detector,meta=load()
    errors=[]
    for f in ["N","P","K","rainfall"]:
        if values[f]<0: errors.append(f"{f} cannot be negative")
    if not 0<=values["humidity"]<=100: errors.append("Humidity must be 0–100%")
    if not 0<=values["ph"]<=14: errors.append("pH must be 0–14")
    if not -50<=values["temperature"]<=60: errors.append("Temperature outside broad sanity range")
    if errors: return {"status":"INVALID_INPUT","errors":errors}

    row=pd.DataFrame([[values[f] for f in FEATURES]],columns=FEATURES)
    probs=model.predict_proba(row)[0]; classes=model.classes_; order=np.argsort(probs)[::-1]
    top=[{"crop":str(classes[i]),"calibrated_probability_percent":round(float(probs[i]*100),2)} for i in order[:3]]
    q=float(meta["conformal_lac_qhat"])
    pred_set=[str(classes[i]) for i,p in enumerate(probs) if 1-p<=q]
    warnings=[]
    for f in FEATURES:
        if values[f]<meta["training_min"][f] or values[f]>meta["training_max"][f]:
            warnings.append(f"{f} outside observed training range")
    flag=int(detector.predict(row)[0]); score=float(detector.decision_function(row)[0])
    review=bool(warnings) or flag==-1 or len(pred_set)!=1
    return {
        "decision_status":"REVIEW_REQUIRED" if review else "MODEL_OUTPUT_AVAILABLE",
        "top_prediction":top[0]["crop"],
        "top_prediction_probability_percent":top[0]["calibrated_probability_percent"],
        "top_three":top,
        "conformal_prediction_set":pred_set,
        "conformal_nominal_coverage":1-float(meta["conformal_alpha"]),
        "range_warnings":warnings,
        "anomaly_status":"POTENTIAL_ANOMALY" if flag==-1 else "NOT_FLAGGED",
        "anomaly_decision_score":round(score,6),
        "important_note":"Benchmark ML output only; validate local season, soil, water and crop requirements with agronomic expertise."
    }

def main():
    print("AGROGUARD AI prediction test")
    vals={}
    for f in FEATURES:
        while True:
            try: vals[f]=float(input(f"Enter {f}: ")); break
            except ValueError: print("Enter a number.")
    print(json.dumps(predict(vals),indent=2))

if __name__=="__main__": main()
