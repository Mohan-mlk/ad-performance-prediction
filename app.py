"""HTTP API for Iris model predictions."""

import pickle
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).with_name("model.pkl")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
app = FastAPI(title="Iris classifier", version="1.0.0")


class IrisFeatures(BaseModel):
    features: Annotated[list[float], Field(min_length=4, max_length=4)]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@app.get("/health")
def health():
    return {"status": "ready" if MODEL_PATH.exists() else "model_missing"}


@app.post("/predict")
def predict(request: IrisFeatures):
    try:
        model = load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    prediction = int(model.predict([request.features])[0])
    return {"class_id": prediction, "class_name": CLASS_NAMES[prediction]}
