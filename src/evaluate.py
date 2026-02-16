import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "tourism.csv")

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_pipeline.joblib")
TARGET = "ProdTaken"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].str.lower().isin(["nan", "none", "null", ""]), col] = np.nan
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = clean_df(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete.")
    print({k: metrics[k] for k in ["roc_auc", "f1"]})


if __name__ == "__main__":
    main()

