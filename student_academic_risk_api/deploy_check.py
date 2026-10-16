"""Verify deploy artifacts before hosting. Run from repo root:

    python student_academic_risk_api/deploy_check.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student_academic_risk_api import config
from student_academic_risk_api.inference import get_model_bundle, predict_from_features

REQUIRED = [
    config.MODEL_PATH,
    config.PREPROCESSOR_PATH,
    config.LABELS_PATH,
    config.METADATA_PATH,
]


def main():
    missing = [str(p) for p in REQUIRED if not p.is_file()]
    if missing:
        print("MISSING deploy files:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

    print(f"Keras model: {config.MODEL_PATH}")
    get_model_bundle()
    sample = {
        "Age": 23,
        "Gender": "Male",
        "Program": "IT",
        "Study_Hours_Per_Week": 4.8,
        "Attendance_Percent": 80.5,
        "Previous_Average_Score": 70.7,
        "Assignment_Score": 36.0,
        "Internet_Access": "Yes",
        "Part_Time_Work": "No",
        "Financial_Stress": "Low",
        "Commute_Minutes": 49.2,
        "Missed_Deadlines": 3,
    }
    result, error, status = predict_from_features(sample)
    if error:
        print(f"Predict failed ({status}): {error}")
        sys.exit(1)
    print(f"Predict OK: {result['predicted_class']} (engine: student_academicrisk.keras)")
    sys.exit(0)


if __name__ == "__main__":
    main()
