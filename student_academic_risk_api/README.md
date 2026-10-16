# Student Academic Risk API

Part of the **single root repository** — use root `pip install -r requirements.txt` (no pandas). Flask REST API aligned with `student_academic_risk_kfold_models.ipynb`. Deployed weights: **`student_academicrisk.keras`**.

## Model artifact (required before deploy)

Generate once (notebook preprocessing + Logistic Regression equivalent saved as `.keras`):

```bash
pip install -r requirements.txt
python scripts/export_model.py --data path\to\student_academic_risk_dataset.csv
```

This writes:

- `models/student_academicrisk.keras` (deploy this file)
- `models/student_academicrisk_preprocessor.joblib`
- `models/student_academicrisk_labels.json`
- `models/model_metadata.json` (includes test accuracy, e.g. **68.25%**)

The CSV is **not** needed at runtime — only for this one-time export.

## Run (when you are ready to test)

```bash
set PORT=5001
python app.py
```

Production example:

```bash
gunicorn --bind 0.0.0.0:5001 app:app
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STUDENT_RISK_MODEL_PATH` | `models/student_academicrisk.keras` | Deployed Keras classifier |
| `STUDENT_RISK_PREPROCESSOR_PATH` | `models/student_academicrisk_preprocessor.joblib` | Feature preprocessor (required with `.keras`) |
| `STUDENT_RISK_LABELS_PATH` | `models/student_academicrisk_labels.json` | Class label order |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `5001` | Port |

## Deploy bundle (commit or upload these)

1. `models/student_academicrisk.keras`
2. `models/student_academicrisk_preprocessor.joblib`
3. `models/student_academicrisk_labels.json`
4. `models/model_metadata.json`

The API **only** loads `student_academicrisk.keras` for predictions (not legacy `.joblib` pipelines).

See `API_DOCUMENTATION.md` for JSON schema and frontend examples.
