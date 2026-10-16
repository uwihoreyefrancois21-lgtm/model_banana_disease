"""
Export student academic risk model for deployment.

Aligns with student_academic_risk_kfold_models.ipynb:
  - same cleaning, features, preprocessing, train/test split (80/20 stratified)
  - notebook best classifier: Logistic Regression (CV macro F1)
  - deployable neural net saved as models/student_academicrisk.keras

Usage (repo root):
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_ROOT))

from schema import CATEGORICAL_FEATURES, FEATURE_FIELDS, NUMERIC_FEATURES  # noqa: E402

RANDOM_STATE = 42
TARGET = "Academic_Risk"
NOTEBOOK_BEST_MODEL = "Logistic Regression"
KERAS_FILENAME = "student_academicrisk.keras"
PREPROCESSOR_FILENAME = "student_academicrisk_preprocessor.joblib"
LABELS_FILENAME = "student_academicrisk_labels.json"


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
            feature_row = {key: _parse_value(key, record.get(key, "")) for key in FEATURE_FIELDS}
            label = str(record[TARGET]).strip()
            rows.append((tuple(feature_row[k] for k in FEATURE_FIELDS), label, feature_row))

    unique = {}
    for key, label, feature_row in rows:
        unique[key] = (label, feature_row)
    cleaned = list(unique.values())
    if not cleaned:
        raise ValueError(f"No usable rows found in {csv_path}")

    y = np.array([item[0] for item in cleaned], dtype=object)
    x = np.array(
        [[feature_row[key] for key in FEATURE_FIELDS] for _, feature_row in cleaned],
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


def evaluate_sklearn_baseline(x_train, x_test, y_train, y_test):
    """Notebook section 7 — Logistic Regression on full pipeline."""
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[("preprocess", make_preprocessor()), ("model", lr)])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    return {
        "model_name": NOTEBOOK_BEST_MODEL,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def train_keras_classifier(x_train, x_test, y_train, y_test, preprocessor):
    from tensorflow import keras

    x_train_p = preprocessor.transform(x_train).astype(np.float32)
    x_test_p = preprocessor.transform(x_test).astype(np.float32)

    label_encoder = LabelEncoder()
    y_train_i = label_encoder.fit_transform(y_train)
    y_test_i = label_encoder.transform(y_test)
    n_classes = len(label_encoder.classes_)

    # Softmax linear layer = Logistic Regression (notebook best model), saved as .keras
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x_train_p.shape[1],)),
            keras.layers.Dense(n_classes, activation="softmax", name="logistic_softmax"),
        ],
        name="student_academic_risk_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_counts = np.bincount(y_train_i)
    total = class_counts.sum()
    class_weight = {i: total / (n_classes * class_counts[i]) for i in range(n_classes)}

    model.fit(
        x_train_p,
        y_train_i,
        validation_split=0.15,
        epochs=200,
        batch_size=128,
        verbose=0,
        class_weight=class_weight,
    )

    proba = model.predict(x_test_p, verbose=0)
    y_pred_i = np.argmax(proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_i)

    return model, label_encoder, {
        "model_name": "Keras Logistic Regression (deployed .keras)",
        "deploy_file": KERAS_FILENAME,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=API_ROOT / "models")
    args = parser.parse_args()

    x, y = load_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("=== Notebook baseline (Logistic Regression, held-out test) ===")
    sklearn_metrics = evaluate_sklearn_baseline(x_train, x_test, y_train, y_test)
    print(f"  Accuracy:  {_format_pct(sklearn_metrics['accuracy'])}")
    print(f"  F1 macro:  {_format_pct(sklearn_metrics['f1_macro'])}")
    print(sklearn_metrics["classification_report"])

    preprocessor = make_preprocessor()
    preprocessor.fit(x_train)

    print("\n=== Deployed Keras model (student_academicrisk.keras) ===")
    keras_model, label_encoder, keras_metrics = train_keras_classifier(
        x_train, x_test, y_train, y_test, preprocessor
    )
    print(f"  Accuracy:  {_format_pct(keras_metrics['accuracy'])}")
    print(f"  F1 macro:  {_format_pct(keras_metrics['f1_macro'])}")
    print(keras_metrics["classification_report"])

    args.models_dir.mkdir(parents=True, exist_ok=True)
    keras_path = args.models_dir / KERAS_FILENAME
    preprocessor_path = args.models_dir / PREPROCESSOR_FILENAME
    labels_path = args.models_dir / LABELS_FILENAME
    metadata_path = args.models_dir / "model_metadata.json"

    keras_model.save(keras_path)
    joblib.dump(preprocessor, preprocessor_path)
    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(list(label_encoder.classes_), handle, indent=2)

    metadata = {
        "notebook_reference": "student_academic_risk_kfold_models.ipynb",
        "notebook_best_model": NOTEBOOK_BEST_MODEL,
        "deploy_model_file": KERAS_FILENAME,
        "preprocessor_file": PREPROCESSOR_FILENAME,
        "random_state": RANDOM_STATE,
        "train_test_split": {"test_size": 0.2, "stratify": True},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "preprocessing": "median/mode imputation, StandardScaler, OneHotEncoder(handle_unknown='ignore')",
        "target_classes": sorted(set(y.tolist())),
        "notebook_baseline_test_metrics": {
            "accuracy": sklearn_metrics["accuracy"],
            "accuracy_display": _format_pct(sklearn_metrics["accuracy"]),
            "precision_macro": sklearn_metrics["precision_macro"],
            "recall_macro": sklearn_metrics["recall_macro"],
            "f1_macro": sklearn_metrics["f1_macro"],
            "f1_macro_display": _format_pct(sklearn_metrics["f1_macro"]),
        },
        "deployed_keras_test_metrics": {
            "model_name": keras_metrics["model_name"],
            "accuracy": keras_metrics["accuracy"],
            "accuracy_display": _format_pct(keras_metrics["accuracy"]),
            "precision_macro": keras_metrics["precision_macro"],
            "recall_macro": keras_metrics["recall_macro"],
            "f1_macro": keras_metrics["f1_macro"],
            "f1_macro_display": _format_pct(keras_metrics["f1_macro"]),
        },
        "model_performance_summary": (
            f"Held-out test accuracy {_format_pct(keras_metrics['accuracy'])} "
            f"(F1 macro {_format_pct(keras_metrics['f1_macro'])})"
        ),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nSaved: {keras_path}")
    print(f"Saved: {preprocessor_path}")
    print(f"Saved: {labels_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
