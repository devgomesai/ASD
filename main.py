"""
main.py  –  FastAPI serving endpoint for ASD Prenatal Risk Prediction
Run with:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Load model + metadata at startup
# ------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "xgb_asd_model.joblib"
META_PATH  = MODEL_DIR / "metadata.json"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        "Run `python train_and_save.py` first."
    )

model    = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())

FEATURES   = metadata["feature_columns"]
THRESHOLD  = metadata["best_threshold"]

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------
app = FastAPI(
    title="ASD Prenatal Risk Prediction API",
    description=(
        "Predicts autism spectrum disorder (ASD) risk from prenatal "
        "risk factors using an XGBoost classifier."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------
class PredictionRequest(BaseModel):
    advanced_maternal: int = Field(
        ..., ge=0, le=1,
        description="Maternal age > 35 (1=Yes, 0=No)"
    )
    advanced_paternal: int = Field(
        ..., ge=0, le=1,
        description="Paternal age > 40 (1=Yes, 0=No)"
    )
    gdm: int = Field(
        ..., ge=0, le=1,
        description="Gestational diabetes mellitus (1=Yes, 0=No)"
    )
    infection: int = Field(
        ..., ge=0, le=1,
        description="Prenatal infection (1=Yes, 0=No)"
    )
    preterm: int = Field(
        ..., ge=0, le=1,
        description="Preterm birth (1=Yes, 0=No)"
    )
    low_bw: int = Field(
        ..., ge=0, le=1,
        description="Low birth weight (1=Yes, 0=No)"
    )
    family_history: int = Field(
        ..., ge=0, le=1,
        description="Family history of ASD (1=Yes, 0=No)"
    )

    class Config:
        schema_extra = {
            "example": {
                "advanced_maternal": 1,
                "advanced_paternal": 0,
                "gdm": 1,
                "infection": 0,
                "preterm": 1,
                "low_bw": 0,
                "family_history": 0
            }
        }


class PredictionResponse(BaseModel):
    asd_risk_probability: float = Field(
        ..., description="Predicted probability of ASD risk (0–1)"
    )
    asd_predicted: bool = Field(
        ..., description="Binary prediction at optimal threshold"
    )
    risk_level: str = Field(
        ..., description="LOW / MODERATE / HIGH risk category"
    )
    threshold_used: float = Field(
        ..., description="Decision threshold applied"
    )


class BatchPredictionRequest(BaseModel):
    records: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_type: str
    features: List[str]
    metrics: dict
    threshold: float


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def _classify_risk(prob: float) -> str:
    percent = prob * 100

    if percent <= 40:
        return "LOW"
    if percent <= 65:
        return "MODERATE"
    return "HIGH"


def _predict_one(req: PredictionRequest) -> PredictionResponse:
    row = np.array([[getattr(req, f) for f in FEATURES]])
    prob = float(model.predict_proba(row)[0, 1])
    return PredictionResponse(
        asd_risk_probability=round(prob, 6),
        asd_predicted=prob >= THRESHOLD,
        risk_level=_classify_risk(prob),
        threshold_used=THRESHOLD,
    )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    return {"message": "ASD Risk Prediction API is running. Visit /docs for the Swagger UI."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Returns model status and performance metrics."""
    return HealthResponse(
        status="ok",
        model_type=metadata["model_type"],
        features=FEATURES,
        metrics=metadata["metrics"],
        threshold=THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PredictionRequest):
    """
    Predict ASD risk for a single patient record.

    Returns the predicted probability, binary label, and risk category.
    """
    try:
        return _predict_one(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(req: BatchPredictionRequest):
    """
    Predict ASD risk for multiple records in one call.
    Maximum 1000 records per request.
    """
    if len(req.records) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Batch size must not exceed 1000 records."
        )
    try:
        preds = [_predict_one(r) for r in req.records]
        return BatchPredictionResponse(predictions=preds, count=len(preds))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns full model metadata including hyperparameters and training metrics."""
    return metadata