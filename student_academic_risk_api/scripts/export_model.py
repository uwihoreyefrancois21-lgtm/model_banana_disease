"""
One-time artifact export aligned with student_academic_risk_kfold_models.ipynb (sections 4–7).

Uses only the standard library + numpy + scikit-learn (no pandas).

    python student_academic_risk_api/scripts/export_model.py --data dataset/student_academic_risk_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_ROOT))

from schema import CATEGORICAL_FEATURES, FEATURE_FIELDS, NUMERIC_FEATURES  # noqa: E402

RANDOM_STATE = 42
TARGET = "Academic_Risk"
DROP_COLUMNS = {"Student_ID", TARGET}
BEST_MODEL_NAME = "Logistic Regression"


def _parse_value(field: str, raw: str):
    if field in NUMERIC_FEATURES:
        if raw is None or str(raw).strip() == "":
            return np.nan
        return float(raw)
    if raw is None or str(raw).strip() == "":
        return np.nan
    return str(raw).strip()


def load_dataset(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            if not record.get(TARGET, "").strip():
                continue
            feature_row = {
                key: _parse_value(key, record.get(key, ""))
                for key in FEATURE_FIELDS
            }
            label = str(record[TARGET]).strip()
            rows.append((tuple(feature_row[k] for k in FEATURE_FIELDS), label, feature_row))

    unique = {}
    for key, label, feature_row in rows:
        unique[key] = (label, feature_row)
    cleaned = list(unique.values())
    if not cleaned:
        raise ValueError(f"No usable rows found in {csv_path}")

    y = np.array([item[0] for item in cleaned], dtype=object)
    feature_dicts = [item[1] for item in cleaned]
    x = np.array(
        [[row[key] for key in FEATURE_FIELDS] for row in feature_dicts],
        dtype=object,
    )
    return x, y


def make_preprocessor():
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_cols = [FEATURE_FIELDS.index(name) for name in NUMERIC_FEATURES]
    categorical_cols = [FEATURE_FIELDS.index(name) for name in CATEGORICAL_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ]
    )


def build_pipeline():
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", model),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=API_ROOT / "models" / "academic_risk_pipeline.joblib",
    )
    args = parser.parse_args()

    x, y = load_dataset(args.data)
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)

    metadata = {
        "model_name": BEST_MODEL_NAME,
        "selection_metric": "mean 5-fold macro F1 (from notebook cross-validation)",
        "random_state": RANDOM_STATE,
        "train_test_split": {"test_size": 0.2, "stratify": True},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "preprocessing": "median/mode imputation, StandardScaler, OneHotEncoder(handle_unknown='ignore')",
        "target_classes": sorted(set(y.tolist())),
    }
    metadata_path = args.output.parent / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved pipeline: {args.output}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
