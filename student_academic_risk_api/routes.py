from flask import Blueprint, jsonify, request

from . import config
from .inference import get_pipeline, predict_from_features
from .schema import FEATURE_FIELDS, TARGET_CLASSES


def create_blueprint(url_prefix: str = "") -> Blueprint:
    bp = Blueprint("student_risk", __name__, url_prefix=url_prefix)

    def api_metadata():
        metadata = config.load_metadata()
        model_ready = config.MODEL_PATH.is_file() and config.PREPROCESSOR_PATH.is_file()
        prefix = url_prefix or ""
        keras_metrics = metadata.get("deployed_keras_test_metrics", {})
        baseline = metadata.get("notebook_baseline_test_metrics", {})
        return {
            "service": "student-academic-risk-api",
            "status": "ok" if model_ready else "model_missing",
            "model_file": config.MODEL_PATH.name,
            "model_ready": model_ready,
            "notebook_reference": metadata.get("notebook_reference", "student_academic_risk_kfold_models.ipynb"),
            "notebook_best_model": metadata.get("notebook_best_model", "Logistic Regression"),
            "deployed_model": keras_metrics.get("model_name", "Keras MLP (deployed)"),
            "model_performance": {
                "deployed_keras_test": keras_metrics,
                "notebook_logistic_regression_test": baseline,
                "summary": metadata.get("model_performance_summary"),
            },
            "target_classes": TARGET_CLASSES,
            "feature_fields": FEATURE_FIELDS,
            "preprocessing": metadata.get(
                "preprocessing",
                "median/mode imputation, one-hot encoding, standardization (notebook pipeline)",
            ),
            "base_path": prefix or "/",
            "endpoints": {
                "health": f"GET {prefix}/health",
                "predict": f"POST {prefix}/predict (JSON body with feature fields)",
            },
        }

    @bp.route("/health", methods=["GET"])
    def health():
        artifacts = {
            "keras_model": config.MODEL_PATH.name,
            "keras_model_present": config.MODEL_PATH.is_file(),
            "preprocessor_present": config.PREPROCESSOR_PATH.is_file(),
            "labels_present": config.LABELS_PATH.is_file(),
        }
        if not all(
            [
                artifacts["keras_model_present"],
                artifacts["preprocessor_present"],
            ]
        ):
            return jsonify(
                {
                    "status": "degraded",
                    "reason": "Deploy student_academicrisk.keras and student_academicrisk_preprocessor.joblib",
                    "artifacts": artifacts,
                }
            ), 503
        try:
            get_pipeline()
        except Exception as exc:
            return jsonify({"status": "degraded", "reason": str(exc), "artifacts": artifacts}), 503
        return jsonify({"status": "ok", "inference_engine": "tensorflow.keras", "artifacts": artifacts})

    @bp.route("/", methods=["GET"])
    @bp.route("/api", methods=["GET"])
    def api_info():
        return jsonify(api_metadata())

    @bp.route("/predict", methods=["POST"])
    def predict():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        body = request.get_json(silent=True)
        result, error, status = predict_from_features(body)
        if error:
            return jsonify({"error": error}), status
        return jsonify(result)

    return bp
