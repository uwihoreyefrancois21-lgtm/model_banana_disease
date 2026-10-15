# Banana Leaf Disease API

Part of the **single root repository** (`requirements.txt` at repo root). Standalone Flask REST API for the MobileNetV2 model trained in `banana_disease.ipynb`.

## Contents (hosting bundle)

- `app.py` — HTTP routes (no bundled frontend)
- `inference.py` — 224×224 RGB preprocessing, MobileNetV2 `preprocess_input`, optional TTA, leaf sanity checks
- `models/banana_leaf_mobilenetv2.keras` — trained weights
- `models/class_names.json` — `["cordana", "healthy", "pestalotiopsis", "sigatoka"]`

## Local setup (after clone)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run (when you are ready to test)

```bash
set PORT=5000
python app.py
```

Production example:

```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BANANA_MODEL_PATH` | `models/banana_leaf_mobilenetv2.keras` | Model file |
| `BANANA_CLASS_NAMES_PATH` | `models/class_names.json` | Class label order |
| `BANANA_USE_TTA` | `true` | Average predictions over 4 flip/rotate views |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `5000` | Port |

See `API_DOCUMENTATION.md` for endpoints and frontend integration.
