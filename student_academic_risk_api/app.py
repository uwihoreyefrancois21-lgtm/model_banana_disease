import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask
from flask_cors import CORS

from student_academic_risk_api import config
from student_academic_risk_api.routes import create_blueprint

app = Flask(__name__)
CORS(app)
app.register_blueprint(create_blueprint())

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=False)
