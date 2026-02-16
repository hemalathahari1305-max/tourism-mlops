import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "tourism.csv")

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "ProdTaken"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    # Drop ID col if exists
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    # Strip whitespace in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].str.lower().isin(["nan", "none", "null", ""]), col] = np.nan
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = clean_df(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 4],
        "model__min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Save best model
    model_path = os.path.join(ARTIFACTS_DIR, "best_pipeline.joblib")
    joblib.dump(best_model, model_path)

    # Save params + best CV score
    out = {
        "best_params": grid.best_params_,
        "best_cv_roc_auc": float(grid.best_score_),
        "model_artifact": "artifacts/best_pipeline.joblib",
    }
    with open(os.path.join(REPORTS_DIR, "best_params.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Save feature list for deployment UI
    features = [c for c in X.columns]
    with open(os.path.join(REPORTS_DIR, "features.json"), "w") as f:
        json.dump(features, f, indent=2)

    print("Training complete.")
    print(out)


if __name__ == "__main__":
    main()

