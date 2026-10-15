"""
One public URL / one PORT for hosting.

Start (production):
  gunicorn -c gunicorn.conf.py host_app:app

Paths:
  GET  /
  GET  /banana/health   POST /banana/predict
  GET  /student/health  POST /student/predict
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from banana_leaf_disease_api.routes import create_blueprint as create_banana_blueprint
from student_academic_risk_api.routes import create_blueprint as create_student_blueprint

BANANA_PREFIX = os.environ.get("BANANA_URL_PREFIX", "/banana")
STUDENT_PREFIX = os.environ.get("STUDENT_URL_PREFIX", "/student")

app = Flask(__name__)
CORS(app)

app.register_blueprint(create_banana_blueprint(BANANA_PREFIX))
app.register_blueprint(create_student_blueprint(STUDENT_PREFIX))


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "ok",
            "message": "Combined ML API (single host URL)",
            "banana_leaf_disease": {
                "base": BANANA_PREFIX,
                "health": f"{BANANA_PREFIX}/health",
                "predict": f"{BANANA_PREFIX}/predict",
            },
            "student_academic_risk": {
                "base": STUDENT_PREFIX,
                "health": f"{STUDENT_PREFIX}/health",
                "predict": f"{STUDENT_PREFIX}/predict",
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
