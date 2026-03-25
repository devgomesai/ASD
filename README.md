# ASD Prenatal Risk Prediction 

## Project Structure

```
.
├── train_and_save.py     # Train + save best model
├── main.py               # FastAPI server
├── requirements.txt      # Python dependencies
└── model/
    ├── xgb_asd_model.joblib   # Saved XGBoost model
    └── metadata.json          # Metrics, threshold, feature list
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and save the model

```bash
python train_and_save.py
```

### 3. Start the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open Swagger UI

Visit → [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints


| Method | Path             | Description               |
| ------ | ---------------- | ------------------------- |
| GET    | `/`              | Root ping                 |
| GET    | `/health`        | Model health + metrics    |
| GET    | `/model/info`    | Full metadata             |
| POST   | `/predict`       | Single prediction         |
| POST   | `/predict/batch` | Batch predictions (≤1000) |


---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "advanced_maternal": 1,
    "advanced_paternal": 0,
    "gdm": 1,
    "infection": 0,
    "preterm": 1,
    "low_bw": 0,
    "family_history": 0
  }'
```

### Example Response

```json
{
  "asd_risk_probability": 0.087412,
  "asd_predicted": false,
  "risk_level": "MODERATE",
  "threshold_used": 0.5
}
```

---

## Input Features


| Field               | Type | Description                   |
| ------------------- | ---- | ----------------------------- |
| `advanced_maternal` | 0/1  | Maternal age > 35             |
| `advanced_paternal` | 0/1  | Paternal age > 40             |
| `gdm`               | 0/1  | Gestational diabetes mellitus |
| `infection`         | 0/1  | Prenatal infection            |
| `preterm`           | 0/1  | Preterm birth                 |
| `low_bw`            | 0/1  | Low birth weight              |
| `family_history`    | 0/1  | Family history of ASD         |


---

## Risk Categories


| Level    | Probability Range |
| -------- | ----------------- |
| LOW      | < 5%              |
| MODERATE | 5% – 15%          |
| HIGH     | > 15%             |


---

## Model Info

- **Algorithm**: XGBoost Classifier
- **ROC AUC**: ~0.64
- **Training samples**: 80,000 (synthetic, epidemiologically calibrated)
- **Threshold**: Optimised via F1-score maximisation

