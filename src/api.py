
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_NAME = "Semiconductor_Quality_Model"
MODEL_VERSION = "1"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
EXPECTED_FEATURES = 590

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Semiconductor Quality Prediction API",
    description="Single and batch semiconductor quality prediction",
    version="1.1"
)

model = None


class PredictionRequest(BaseModel):
    features: list[float | None] = Field(
        ...,
        description="590 sensor features; missing values can be null"
    )


class BatchPredictionRequest(BaseModel):
    features: list[list[float | None]] = Field(
        ...,
        min_length=1
    )


@app.on_event("startup")
def load_model():
    global model

    try:
        print("Loading registered MLflow model...")
        model = mlflow.sklearn.load_model(MODEL_URI)
        print(f"Model loaded: {MODEL_NAME}, version {MODEL_VERSION}")

    except Exception as error:
        model = None
        print(f"Model loading failed: {error}")


def run_predictions(samples):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="MLflow model is not loaded."
        )

    if not samples:
        raise HTTPException(
            status_code=400,
            detail="At least one sample is required."
        )

    invalid_rows = [
        index + 1
        for index, row in enumerate(samples)
        if len(row) != EXPECTED_FEATURES
    ]

    if invalid_rows:
        raise HTTPException(
            status_code=400,
            detail=f"Each sample must contain exactly {EXPECTED_FEATURES} features. Invalid rows: {invalid_rows[:20]}"
        )

    try:
        clean_samples = [
            [
                np.nan if value is None else float(value)
                for value in row
            ]
            for row in samples
        ]

        input_data = pd.DataFrame(
            clean_samples,
            columns=range(EXPECTED_FEATURES)
        )

        # Reject infinity, but preserve NaN for the model's imputer.
        if np.isinf(input_data.to_numpy()).any():
            raise ValueError("Sensor values must not contain infinity.")

        predictions = model.predict(input_data).astype(int).tolist()
        probabilities = model.predict_proba(input_data)

        if hasattr(model, "classes_"):
            classes = list(model.classes_)
        else:
            classes = list(model[-1].classes_)

        if 1 not in classes:
            raise ValueError("Model does not contain the Fail class (1).")

        fail_index = classes.index(1)

        fail_probabilities = (
            probabilities[:, fail_index].astype(float).tolist()
        )

        qualities = [
            "Fail" if prediction == 1 else "Pass"
            for prediction in predictions
        ]

        return {
            "predictions": predictions,
            "qualities": qualities,
            "fail_probabilities": fail_probabilities,
            "total_samples": len(predictions),
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


@app.get("/")
def home():
    return {
        "message": "Semiconductor Quality Prediction API",
        "status": "running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
        "input_features": EXPECTED_FEATURES,
        "supports_batch_prediction": True
    }


@app.post("/predict")
def predict_quality(request: PredictionRequest):
    result = run_predictions([request.features])

    return {
        "prediction": result["predictions"][0],
        "quality": result["qualities"][0],
        "fail_probability": result["fail_probabilities"][0],
        "model_name": result["model_name"],
        "model_version": result["model_version"]
    }


@app.post("/predict-batch")
def predict_quality_batch(request: BatchPredictionRequest):
    return run_predictions(request.features)

