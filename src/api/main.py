from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Literal

import logging
import uuid

from fastapi import FastAPI, Request, HTTPException

import mlflow
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from src.api.storage import build_engine, insert_prediction
from src.features.columns import ALL_FEATURES
from src.inference.pipeline import predict_scores
from dotenv import load_dotenv


load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_name = os.environ.get("MODEL_NAME", "churn-classifier")
        alias = os.environ.get("MODEL_ALIAS", "champion")
        uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

        mlflow.set_tracking_uri(uri)
        model_uri = f"models:/{model_name}@{alias}"
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        model = mlflow.sklearn.load_model(local_path)
        model_settings = mlflow.MlflowClient().get_model_version_by_alias(model_name, alias)

        engine = build_engine()

        app.state.model = model
        app.state.version = model_settings.version
        app.state.threshold = float(model_settings.tags["threshold"])
        app.state.ready = True
        app.state.engine = engine
    except Exception:
        logger.exception("Failed to load model during startup")
        app.state.ready = False

    yield




class PredictRequest(BaseModel):
    customer_id: str

    SeniorCitizen: Literal[0, 1]
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(ge=0)
    gender: Literal["Female", "Male"]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    
    


    

    @model_validator(mode="after")
    def total_charges_null_only_for_new_customers(self):
        if self.TotalCharges is None and self.tenure != 0:
            raise ValueError("TotalCharges can only be null when tenure is 0")
        return self

class PredictResponse(BaseModel):
    request_id: str
    probability: float
    prediction: str
    threshold: float
    model_version: int
    customer_id: str




app = FastAPI(lifespan=lifespan)
app.state.ready = False 




@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok", "version": app.state.version, "threshold": app.state.threshold}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="not ready")
    X_df = pd.DataFrame([payload.model_dump()])[ALL_FEATURES]

    
    y_score = predict_scores(app.state.model, X_df)[0]

    prediction = "churn" if y_score >= request.app.state.threshold else "no_churn"
    request_id = str(uuid.uuid4())
    response = PredictResponse(
        request_id = request_id,
        probability = y_score,
        prediction = prediction,
        threshold = request.app.state.threshold,
        model_version = request.app.state.version,
        customer_id = payload.customer_id 
    )
    time = datetime.now(timezone.utc)

    try:
        insert_prediction(request.app.state.engine, payload, response, time)
    except Exception:
        logger.exception("Failed to insert prediction in db")
        raise HTTPException(status_code=500, detail="Failed to insert prediction")

    return response