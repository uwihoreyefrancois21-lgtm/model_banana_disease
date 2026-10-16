"""
Single Render deployment — one repo, one PORT, two APIs:

  /banana/predict   → banana_leaf_mobilenetv2.keras
  /student/predict  → student_academicrisk.keras

Local:  python app.py
Render: gunicorn -c gunicorn.conf.py app:app
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify
from flask_cors import CORS

from banana_leaf_disease_api.routes import create_blueprint as create_banana_blueprint
from student_academic_risk_api.routes import create_blueprint as create_student_blueprint

BANANA_PREFIX = os.environ.get("BANANA_URL_PREFIX", "/banana")
STUDENT_PREFIX = os.environ.get("STUDENT_URL_PREFIX", "/student")

# "*" (default) allows any browser origin. Set CORS_ORIGINS to a comma separated
# list to restrict it, e.g. CORS_ORIGINS="https://my-app.vercel.app,http://localhost:5173"
_raw_origins = os.environ.get("CORS_ORIGINS", "*").strip()
ALLOWED_ORIGINS = (
    "*"
    if _raw_origins in ("", "*")
    else [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]
)

app = Flask("ml_combined_api")

# Applies to every route, including the OPTIONS preflight browser clients send
# before a JSON POST.
CORS(
    app,
    origins=ALLOWED_ORIGINS,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=86400,
)

app.register_blueprint(create_banana_blueprint(BANANA_PREFIX))
app.register_blueprint(create_student_blueprint(STUDENT_PREFIX))


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "ok",
            "message": "One service, one port — consume each model on its own path",
            "banana_leaf_disease": {
                "model": "banana_leaf_mobilenetv2.keras",
                "base": BANANA_PREFIX,
                "health": f"{BANANA_PREFIX}/health",
                "predict": f"{BANANA_PREFIX}/predict",
            },
            "student_academic_risk": {
                "model": "student_academicrisk.keras",
                "base": STUDENT_PREFIX,
                "health": f"{STUDENT_PREFIX}/health",
                "predict": f"{STUDENT_PREFIX}/predict",
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"Combined ML API (port {port})")
    print(f"  Banana:  http://127.0.0.1:{port}{BANANA_PREFIX}/predict")
    print(f"  Student: http://127.0.0.1:{port}{STUDENT_PREFIX}/predict")
    app.run(host="0.0.0.0", port=port, debug=False)
