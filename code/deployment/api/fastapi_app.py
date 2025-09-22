from typing import List, Optional
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


FEATURE_ORDER = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "C",
    "Q",
    "S",
    "female",
    "male",
]


class PassengerFeatures(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    C: int
    Q: int
    S: int
    female: int
    male: int


class PredictRequest(BaseModel):
    instances: List[PassengerFeatures]


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[float]]


def load_artifacts(model_path: str):
    artifacts = joblib.load(model_path)
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    return scaler, model


MODEL_PATH = os.getenv("MODEL_PATH", "/models/trained_model")
scaler, model = load_artifacts(MODEL_PATH)


app = FastAPI(title="Titanic Survival Predictor API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    data = [row.model_dump() for row in req.instances]
    df = pd.DataFrame(data)
    df = df[FEATURE_ORDER]

    X = scaler.transform(df)
    probabilities = model.predict_proba(X)[:, 1].tolist()
    predictions = model.predict(X).astype(int).tolist()
    return PredictResponse(predictions=predictions, probabilities=probabilities)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


