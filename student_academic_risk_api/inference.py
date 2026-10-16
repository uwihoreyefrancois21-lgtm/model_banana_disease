import json

import joblib
import numpy as np
from tensorflow import keras

from . import config
from .schema import ALLOWED_CATEGORICAL, CATEGORICAL_FEATURES, FEATURE_FIELDS, NUMERIC_FEATURES

_model = None
_preprocessor = None
_class_labels = None
_metadata = config.load_metadata()


def _load_class_labels():
    global _class_labels
    if _class_labels is None:
        if config.LABELS_PATH.is_file():
            with open(config.LABELS_PATH, "r", encoding="utf-8") as handle:
                _class_labels = json.load(handle)
        else:
            _class_labels = _metadata.get("target_classes", ["At_Risk", "Not_At_Risk"])
    return _class_labels


def get_model_bundle():
    global _model, _preprocessor
    if _model is None:
        if not config.MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found: {config.MODEL_PATH}. "
                "Run: python student_academic_risk_api/scripts/export_model.py --data dataset/student_academic_risk_dataset.csv"
            )
        if not config.PREPROCESSOR_PATH.is_file():
            raise FileNotFoundError(f"Preprocessor not found: {config.PREPROCESSOR_PATH}")
        _model = keras.models.load_model(config.MODEL_PATH)
        _preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    return _model, _preprocessor


def get_pipeline():
    """Health-check compatibility alias."""
    get_model_bundle()
    return _model


def validate_payload(data):
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object"

    unknown = [key for key in data if key not in FEATURE_FIELDS]
    if unknown:
        return None, f"Unknown field(s): {', '.join(sorted(unknown))}"

    missing = [key for key in FEATURE_FIELDS if key not in data]
    if missing:
        return None, f"Missing required field(s): {', '.join(sorted(missing))}"

    cleaned = {}
    for key in NUMERIC_FEATURES:
        value = data[key]
        if value is None or (isinstance(value, str) and value.strip() == ""):
            cleaned[key] = np.nan
            continue
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            return None, f"Field '{key}' must be numeric or null"

    for key in CATEGORICAL_FEATURES:
        value = data[key]
        if value is None or (isinstance(value, str) and value.strip() == ""):
            cleaned[key] = np.nan
            continue
        if not isinstance(value, str):
            return None, f"Field '{key}' must be a string category or null"
        if value not in ALLOWED_CATEGORICAL[key]:
            allowed = ", ".join(ALLOWED_CATEGORICAL[key])
            return None, f"Invalid value for '{key}': {value}. Allowed: {allowed}"
        cleaned[key] = value

    return cleaned, None


def predict_from_features(data):
    cleaned, error = validate_payload(data)
    if error:
        return None, error, 400

    row = [cleaned[key] for key in FEATURE_FIELDS]
    features = np.array([row], dtype=object)
    model, preprocessor = get_model_bundle()
    x = preprocessor.transform(features)
    proba = model.predict(x, verbose=0)[0]
    classes = _load_class_labels()
    probabilities = {str(classes[i]): float(proba[i]) for i in range(len(classes))}
    predicted_idx = int(np.argmax(proba))
    predicted_class = str(classes[predicted_idx])

    perf = _metadata.get("deployed_keras_test_metrics", {})
    return {
        "predicted_class": predicted_class,
        "prediction": predicted_class,
        "probabilities": probabilities,
        "model_file": config.MODEL_PATH.name,
        "model_name": perf.get("model_name", "Keras MLP (deployed)"),
        "model_accuracy_display": perf.get("accuracy_display"),
        "model_f1_macro_display": perf.get("f1_macro_display"),
    }, None, 200
