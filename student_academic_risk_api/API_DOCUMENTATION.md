# Student Academic Risk API — Documentation

Base URL examples: `http://localhost:5001` (local) or `https://your-student-risk-api.example.com` (hosted).

Predictor: **Logistic Regression** pipeline (preprocess + model) from `student_academic_risk_kfold_models.ipynb`, selected by highest mean 5-fold macro F1 (~0.659).

Target labels: `At_Risk`, `Not_At_Risk`.

CORS is enabled for browser clients.

---

## Endpoints

### `GET /health`

**200** when `models/academic_risk_pipeline.joblib` exists and loads.

**503** if artifact missing:

```json
{ "status": "degraded", "reason": "model artifact missing" }
```

---

### `GET /` and `GET /api`

**200** (may include `"status": "model_missing"` until export is run)

```json
{
  "service": "student-academic-risk-api",
  "status": "ok",
  "model_file": "academic_risk_pipeline.joblib",
  "model_ready": true,
  "best_model": "Logistic Regression",
  "target_classes": ["At_Risk", "Not_At_Risk"],
  "feature_fields": [
    "Age", "Gender", "Program", "Study_Hours_Per_Week",
    "Attendance_Percent", "Previous_Average_Score", "Assignment_Score",
    "Internet_Access", "Part_Time_Work", "Financial_Stress",
    "Commute_Minutes", "Missed_Deadlines"
  ],
  "preprocessing": "median/mode imputation, one-hot encoding, standardization (notebook pipeline)",
  "endpoints": {
    "health": "GET /health",
    "predict": "POST /predict (JSON body with feature fields)"
  }
}
```

---

### `POST /predict`

**Content-Type:** `application/json`

All fields below are **required** in the request body. Use `null` for missing numeric/categorical values — the pipeline imputes them (median / mode), consistent with the notebook.

| Field | Type | Allowed / notes |
|-------|------|-----------------|
| `Age` | number | Student age |
| `Gender` | string | `Female`, `Male` |
| `Program` | string | `Business`, `Education`, `Engineering`, `Hospitality`, `IT` |
| `Study_Hours_Per_Week` | number or null | Weekly study hours |
| `Attendance_Percent` | number | 0–100 |
| `Previous_Average_Score` | number or null | Prior average score |
| `Assignment_Score` | number | Current assignment score |
| `Internet_Access` | string | `Yes`, `No` |
| `Part_Time_Work` | string | `Yes`, `No` |
| `Financial_Stress` | string | `Low`, `Medium`, `High` |
| `Commute_Minutes` | number | Commute time |
| `Missed_Deadlines` | number | Count of missed deadlines |

`Student_ID` is intentionally excluded (identifier only in training data).

**Example request**

```json
{
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
  "Missed_Deadlines": 3
}
```

**Success 200**

```json
{
  "predicted_class": "At_Risk",
  "prediction": "At_Risk",
  "probabilities": {
    "At_Risk": 0.71,
    "Not_At_Risk": 0.29
  },
  "model_name": "Logistic Regression"
}
```

**Errors**

| Status | Example | Cause |
|--------|---------|--------|
| 400 | `{ "error": "Content-Type must be application/json" }` | Wrong content type |
| 400 | `{ "error": "Missing required field(s): Age, Gender" }` | Incomplete body |
| 400 | `{ "error": "Invalid value for 'Program': Law. Allowed: ..." }` | Bad category |
| 400 | `{ "error": "Field 'Age' must be numeric or null" }` | Type error |
| 500 | `{ "error": "..." }` | Missing model file or load failure |

---

## Create the model file (one time)

From the project root, with the training CSV available locally (not shipped in this repo):

```bash
python scripts/export_model.py --data path/to/student_academic_risk_dataset.csv
```

Commit or upload `models/academic_risk_pipeline.joblib` to your host (or build it in CI from a private data source).

---

## Frontend integration

**JavaScript (fetch)**

```javascript
const API_BASE = "https://your-student-risk-api.example.com";

async function predictRisk(student) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(student),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);

  return {
    label: data.predicted_class,
    probs: data.probabilities,
  };
}

// Example payload keys must match API (no Student_ID)
const payload = {
  Age: 26,
  Gender: "Female",
  Program: "Engineering",
  Study_Hours_Per_Week: 11.4,
  Attendance_Percent: 80.2,
  Previous_Average_Score: 81.8,
  Assignment_Score: 66.3,
  Internet_Access: "Yes",
  Part_Time_Work: "Yes",
  Financial_Stress: "Low",
  Commute_Minutes: 41.7,
  Missed_Deadlines: 1,
};
```

**cURL**

```bash
curl -X POST "https://your-student-risk-api.example.com/predict" \
  -H "Content-Type: application/json" \
  -d "{\"Age\":23,\"Gender\":\"Male\",\"Program\":\"IT\",\"Study_Hours_Per_Week\":4.8,\"Attendance_Percent\":80.5,\"Previous_Average_Score\":70.7,\"Assignment_Score\":36.0,\"Internet_Access\":\"Yes\",\"Part_Time_Work\":\"No\",\"Financial_Stress\":\"Low\",\"Commute_Minutes\":49.2,\"Missed_Deadlines\":3}"
```

---

## Deployment checklist

1. Push the **whole repository** (both API folders); deploy this service from `student_academic_risk_api/`.
2. Run `export_model.py` once and include `academic_risk_pipeline.joblib` on the server.
3. `pip install -r requirements.txt`
4. `gunicorn app:app` with `PORT=5001` (or platform default).
5. Wire your dashboard to `POST /predict` with JSON.

**Note:** Predictions use a synthetic teaching dataset; treat outputs as demo/educational, not real institutional decisions.
