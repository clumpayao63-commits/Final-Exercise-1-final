"""
Stand-in for your MIDTERM notebook code.
REPLACE the data loading / feature list / model with the exact ones from your
midterm .ipynb, then re-run:  python train_model.py
It saves the fitted preprocessing+model pipeline once (never retrain in the app).
"""
import json, datetime as dt
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "sla_breached"
CATEGORICAL = ["request_category", "priority", "channel", "department"]
NUMERIC = ["submitted_hour", "submitted_dayofweek", "staff_open_tickets",
           "attachments_count", "sla_hours_allowed", "is_resubmission"]
FEATURES = CATEGORICAL + NUMERIC
# LEAKAGE (excluded): resolution_hours, closed_at, first_response_hours, final_status

def make_synthetic(n=2000, seed=SEED):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "request_id": [f"REQ-{i:05d}" for i in range(1, n + 1)],
        "request_category": r.choice(["Enrollment", "Scholarship", "Records", "Clearance", "Shifting", "ID Replacement"], n),
        "priority": r.choice(["Low", "Medium", "High"], n, p=[.4, .4, .2]),
        "channel": r.choice(["Portal", "Email", "Walk-in"], n, p=[.5, .3, .2]),
        "department": r.choice(["Registrar", "Finance", "Student Affairs", "Admissions"], n),
        "submitted_hour": r.integers(7, 20, n),
        "submitted_dayofweek": r.integers(0, 5, n),
        "staff_open_tickets": r.integers(2, 40, n),
        "attachments_count": r.integers(0, 6, n),
        "sla_hours_allowed": r.choice([24, 48, 72, 120], n),
        "is_resubmission": r.choice([0, 1], n, p=[.85, .15]),
    })
    z = (-2.2 + 0.07 * df.staff_open_tickets - 0.012 * df.sla_hours_allowed
         + 0.5 * df.is_resubmission + 0.15 * df.attachments_count
         + 0.6 * (df.priority == "High") + 0.5 * (df.request_category.isin(["Scholarship", "Records"]))
         + 0.4 * (df.submitted_dayofweek == 4) + r.normal(0, .6, n))
    df[TARGET] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df

if __name__ == "__main__":
    df = make_synthetic()
    df.to_csv("data/sample_service_requests.csv", index=False)
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC)])
    pipe = Pipeline([("prep", pre),
                     ("model", RandomForestClassifier(n_estimators=200, max_depth=8,
                                                      class_weight="balanced", random_state=SEED))])
    pipe.fit(Xtr, ytr)
    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
    meta = {
        "model_version": "v1.0.0", "model_type": "RandomForestClassifier",
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "target": TARGET, "features": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC,
        "prediction_checkpoint": "At request submission/assignment (before any handling outcome is known)",
        "excluded_leakage": ["resolution_hours", "closed_at", "first_response_hours", "final_status"],
        "metrics": {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
                    "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                    "roc_auc": roc_auc_score(yte, proba),
                    "confusion_matrix": confusion_matrix(yte, pred).tolist()},
        "test_size": len(yte),
    }
    joblib.dump(pipe, "models/trained_model.joblib")
    json.dump(meta, open("models/model_metadata.json", "w"), indent=2)
    print(json.dumps(meta["metrics"], indent=2))
