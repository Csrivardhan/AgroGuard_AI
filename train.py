from pathlib import Path
import json, warnings
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA, ART, REP = ROOT / "data" / "Crop_recommendation.csv", ROOT / "artifacts", ROOT / "reports"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEED, ALPHA = 42, 0.10
MODEL_VERSION = "AGROGUARD-ML-v2.0"


def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Place Kaggle CSV at {DATA}")
    df = pd.read_csv(DATA)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = [c.lower() for c in FEATURES] + ["label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}; found {list(df.columns)}")
    df = df[required].copy()
    df.rename(columns={c.lower(): c for c in FEATURES}, inplace=True)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["label"].astype("string").str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES + ["label"], inplace=True)
    df = df[df["label"] != ""].drop_duplicates().reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("At least two crop classes are required.")
    print("Clean rows:", len(df), "Classes:", df["label"].nunique())
    return df


def build_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
        "SVM": Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=SEED))]),
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=4000, random_state=SEED))]),
    }


def ece_score(y, probs, classes, bins=10):
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx[v] for v in y])
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == yi).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for b in range(bins):
        mask = (conf >= edges[b]) & ((conf <= edges[b + 1]) if b == bins - 1 else (conf < edges[b + 1]))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def lac_threshold(y_cal, p_cal, classes, alpha=ALPHA):
    idx = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([idx[v] for v in y_cal])
    scores = 1.0 - p_cal[np.arange(len(y_cal)), true_idx]
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return 1.0 if k > len(scores) else float(np.sort(scores)[k - 1])


def compute_crop_profiles(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Computes strictly train-set feature quantiles per crop for Compatibility Frontier."""
    train_df = X_train.copy()
    train_df["label"] = y_train.values
    profiles = {}
    for crop, group in train_df.groupby("label"):
        profiles[str(crop)] = {}
        for f in FEATURES:
            s = group[f]
            profiles[str(crop)][f] = {
                "min": round(float(s.min()), 3),
                "q25": round(float(s.quantile(0.25)), 3),
                "median": round(float(s.median()), 3),
                "q75": round(float(s.quantile(0.75)), 3),
                "max": round(float(s.max()), 3),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std(ddof=0)) or 1e-6, 3),
            }
    return profiles


def main():
    ART.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)
    df = load_data()
    X = df[FEATURES].astype(float)
    y = df["label"].astype(str)

    # Leak-free 60% train, 20% conformal calibration, 20% untouched test
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.40, random_state=SEED, stratify=y)
    X_cal, X_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    print("Split sizes:", len(X_train), len(X_cal), len(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = build_models()
    rows = []
    for name, est in models.items():
        print("Cross-validating", name)
        s = cross_validate(
            est, X_train, y_train, cv=cv, n_jobs=-1,
            scoring={"accuracy": "accuracy", "precision": "precision_macro", "recall": "recall_macro", "f1": "f1_macro"}
        )
        rows.append({
            "model": name,
            "cv_accuracy": float(s["test_accuracy"].mean()),
            "cv_macro_precision": float(s["test_precision"].mean()),
            "cv_macro_recall": float(s["test_recall"].mean()),
            "cv_macro_f1": float(s["test_f1"].mean()),
            "cv_macro_f1_std": float(s["test_f1"].std()),
        })

    # Hyperparameter tuning across tree and kernel/distance families
    search_spaces = {
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 10, 15, 20, 30],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 3],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
            {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 10, 15, 20, 30],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 3],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "SVM": (
            Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=SEED))]),
            {"model__C": [0.5, 1.0, 5.0, 10.0, 25.0], "model__gamma": ["scale", "auto", 0.05, 0.1]},
        ),
    }

    tuned = {}
    for name, (est, space) in search_spaces.items():
        search = RandomizedSearchCV(
            est, space, n_iter=15, scoring="f1_macro", cv=cv, random_state=SEED, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        tuned[name] = (search.best_estimator_, float(search.best_score_))
        print(name, "best CV Macro F1:", round(search.best_score_, 4))

    candidates = {n: clone(m) for n, m in models.items()}
    scores = {r["model"]: float(r["cv_macro_f1"]) for r in rows}
    for n, (m, s) in tuned.items():
        candidates["Tuned " + n] = m
        scores["Tuned " + n] = s

    winner = max(scores, key=scores.get)
    print("Selected champion model:", winner)

    # Fit & calibrate ALL 6 baseline models for the Model Challenge Arena
    arena_models = {}
    arena_leaderboard = []
    for name, est in models.items():
        cal_est = CalibratedClassifierCV(estimator=clone(est), method="sigmoid", cv=cv, n_jobs=-1)
        cal_est.fit(X_train, y_train)
        arena_models[name] = cal_est

        p_t = cal_est.predict_proba(X_test)
        pr_t = cal_est.classes_[p_t.argmax(axis=1)]
        cv_row = next(r for r in rows if r["model"] == name)
        arena_leaderboard.append({
            "model": name,
            "cv_macro_f1": round(cv_row["cv_macro_f1"], 4),
            "cv_macro_f1_std": round(cv_row["cv_macro_f1_std"], 4),
            "test_accuracy": round(float(accuracy_score(y_test, pr_t)), 4),
            "test_macro_precision": round(float(precision_score(y_test, pr_t, average="macro", zero_division=0)), 4),
            "test_macro_recall": round(float(recall_score(y_test, pr_t, average="macro", zero_division=0)), 4),
            "test_macro_f1": round(float(f1_score(y_test, pr_t, average="macro", zero_division=0)), 4),
            "test_log_loss": round(float(log_loss(y_test, p_t, labels=cal_est.classes_)), 4),
            "test_ece": round(ece_score(y_test.to_numpy(), p_t, cal_est.classes_), 4),
        })

    comparison = pd.DataFrame(arena_leaderboard).sort_values("test_macro_f1", ascending=False)
    comparison.to_csv(REP / "model_comparison.csv", index=False)

    # Calibrate Champion Model + LAC Conformal Prediction
    calibrated = CalibratedClassifierCV(estimator=candidates[winner], method="sigmoid", cv=cv, n_jobs=-1)
    calibrated.fit(X_train, y_train)
    classes = calibrated.classes_
    p_cal = calibrated.predict_proba(X_cal)
    qhat = lac_threshold(y_cal.to_numpy(), p_cal, classes)

    p_test = calibrated.predict_proba(X_test)
    pred = classes[p_test.argmax(axis=1)]
    sets = [[str(classes[i]) for i, p in enumerate(row) if 1 - p <= qhat] for row in p_test]
    coverage = np.mean([str(actual) in s for actual, s in zip(y_test.to_numpy(), sets)])

    metrics = {
        "model_version": MODEL_VERSION,
        "selected_model": winner,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "test_log_loss": float(log_loss(y_test, p_test, labels=classes)),
        "test_top_label_ece": ece_score(y_test.to_numpy(), p_test, classes),
        "conformal_nominal_coverage": 1 - ALPHA,
        "test_empirical_conformal_coverage": float(coverage),
        "test_mean_prediction_set_size": float(np.mean([len(s) for s in sets])),
        "train_rows": len(X_train),
        "conformal_calibration_rows": len(X_cal),
        "test_rows": len(X_test),
        "classes": len(classes),
    }
    print("FINAL TEST METRICS\n", json.dumps(metrics, indent=2))

    # Save reports & plots
    pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).transpose().to_csv(
        REP / "classification_report.csv"
    )
    cm = confusion_matrix(y_test, pred, labels=classes)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Final Test Confusion Matrix")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(REP / "confusion_matrix.png", dpi=180)
    plt.close()

    conf = p_test.max(axis=1)
    correct = (pred == y_test.to_numpy()).astype(int)
    edges = np.linspace(0, 1, 11)
    bx, by = [], []
    for b in range(10):
        mask = (conf >= edges[b]) & ((conf <= edges[b + 1]) if b == 9 else (conf < edges[b + 1]))
        if mask.any():
            bx.append(conf[mask].mean())
            by.append(correct[mask].mean())
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", label="Ideal")
    plt.plot(bx, by, "o-", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REP / "reliability_diagram.png")
    plt.close()

    anomaly = Pipeline([("scale", StandardScaler()), ("iforest", IsolationForest(n_estimators=300, contamination="auto", random_state=SEED))])
    anomaly.fit(X_train)
    anomaly_rate = float(np.mean(anomaly.predict(X_test) == -1))

    imp = permutation_importance(calibrated, X_test, y_test, n_repeats=10, random_state=SEED, scoring="f1_macro", n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(REP / "permutation_importance.csv", index=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=imp_df, x="importance_mean", y="feature")
    plt.xlabel("Decrease in Macro F1 after permutation")
    plt.title("Permutation Importance")
    plt.tight_layout()
    plt.savefig(REP / "permutation_importance.png")
    plt.close()

    stability = []
    sample = X_test.head(min(100, len(X_test))).copy()
    original = calibrated.predict(sample)
    for f in FEATURES:
        iqr = float(X_train[f].quantile(0.75) - X_train[f].quantile(0.25))
        if iqr == 0:
            continue
        for delta in [-0.05, -0.01, 0.01, 0.05]:
            changed = sample.copy()
            changed[f] += delta * iqr
            stability.append({
                "feature": f,
                "delta_iqr": delta,
                "prediction_consistency": float(np.mean(calibrated.predict(changed) == original)),
            })
    pd.DataFrame(stability).to_csv(REP / "stability_report.csv", index=False)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": winner,
        "features": FEATURES,
        "classes": [str(c) for c in classes],
        "training_min": {k: float(v) for k, v in X_train.min().items()},
        "training_median": {k: float(v) for k, v in X_train.median().items()},
        "training_max": {k: float(v) for k, v in X_train.max().items()},
        "crop_profiles": compute_crop_profiles(X_train, y_train),
        "arena_leaderboard": arena_leaderboard,
        "conformal_alpha": ALPHA,
        "conformal_lac_qhat": float(qhat),
        "random_state": SEED,
        "limitations": [
            "Training ranges are not agronomic limits.",
            "Isolation Forest flags are not proof of sensor error.",
            "Conformal coverage relies on exchangeability.",
            "Benchmark performance does not establish regional farm suitability.",
        ],
    }

    joblib.dump(calibrated, ART / "calibrated_crop_model.joblib")
    joblib.dump(arena_models, ART / "arena_models.joblib")
    joblib.dump(anomaly, ART / "anomaly_detector.joblib")
    (ART / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REP / "evaluation_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Upgraded training complete. Artifacts:", ART, "Anomaly test flag rate:", round(anomaly_rate, 4))


if __name__ == "__main__":
    main()