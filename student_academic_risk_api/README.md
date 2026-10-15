# Student Academic Risk API

Part of the **single root repository** — use root `pip install -r requirements.txt` (no pandas). Flask REST API for the **Logistic Regression** pipeline from `student_academic_risk_kfold_models.ipynb`.

## Model artifact (required before deploy)

The notebook fits the final pipeline but does not commit a `.joblib` file. Generate it once (same logic as notebook section 7 — no cross-validation rerun):

```bash
pip install -r requirements.txt
python scripts/export_model.py --data path\to\student_academic_risk_dataset.csv
```

This writes:

- `models/academic_risk_pipeline.joblib`
- updates `models/model_metadata.json` feature lists

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
| `STUDENT_RISK_MODEL_PATH` | `models/academic_risk_pipeline.joblib` | Fitted sklearn pipeline |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `5001` | Port |

See `API_DOCUMENTATION.md` for JSON schema and frontend examples.
