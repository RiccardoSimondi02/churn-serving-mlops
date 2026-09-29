import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from src.api.storage import build_engine, insert_prediction, insert_request_log
from src.features.columns import ALL_FEATURES
from src.inference.model_loader import return_model_info
from src.inference.pipeline import predict_scores

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Buckets built around the measured distribution (p50 49 ms at c=1, 150 ms at c=8,
# max 223 ms): dense between 30 and 250 ms, sparse outside.
LATENCY_BUCKETS = (
    0.01, 0.02,
    0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.12, 0.14, 0.16, 0.18, 0.20, 0.225, 0.25,
    0.5, 1.0, 2.5,
)

REQUESTS = Counter(
    "http_requests_total", "HTTP requests", ["method", "path", "status_code"],
)
LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["path"],
    buckets=LATENCY_BUCKETS,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and open the DB engine once at startup.

    On success the app is marked ready; on any failure the exception is logged
    and `ready` stays False so /readyz reports 503.
    """
    try:
        model, version, threshold = return_model_info()
        engine = build_engine()

        app.state.model = model
        app.state.version = version
        app.state.threshold = float(threshold)
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
    event_time: AwareDatetime | None = None

    
    


    

    @model_validator(mode="after")
    def total_charges_null_only_for_new_customers(self):
        if self.TotalCharges is None and self.tenure != 0:
            raise ValueError("TotalCharges can only be null when tenure is 0")
        return self

    @model_validator(mode="after")
    def event_time_not_future(self):
        if self.event_time is not None and self.event_time > datetime.now(UTC) + timedelta(minutes=10):
            raise ValueError("event_time cannot be in the future")
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

@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    """Assign/propagate a request id, time the request, and record telemetry.

    Updates the Prometheus counters/histogram, emits a structured log line, and
    writes a row to request_log (off the event loop). A failed request_log write
    is logged but never blocks the response.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()
    request.state.arrived_at = start
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    path = request.url.path
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(path).observe(duration_ms / 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(json.dumps({
        "request_id": request_id,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2),
        "model_version": getattr(request.state, "model_version", None),
        "outcome": getattr(request.state, "outcome", None),
    }))
    try:
        await asyncio.to_thread(
            insert_request_log,
            request.app.state.engine,
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            getattr(request.state, "routing_ms", None),
            getattr(request.state, "predict_ms", None),
            getattr(request.state, "insert_ms", None),
        )
    except Exception:
        logger.exception("Failed to write request_log")
        
    return response

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
    """Score one customer and persist the prediction.

    Returns 503 until the model is loaded. Defaults event_time to now when the
    caller omits it, scores the payload, applies the champion threshold to turn
    the probability into a churn/no_churn label, and persists the row; a failed
    insert surfaces as a 500. Stage timings are stashed on request.state for the
    middleware to log.
    """
    routing_ms = (time.perf_counter() - request.state.arrived_at) * 1000
    request.state.routing_ms = routing_ms

    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="not ready")

    now = datetime.now(UTC)
    if payload.event_time is None:
        payload.event_time = now

    X_df = pd.DataFrame([payload.model_dump()])[ALL_FEATURES]

    t0 = time.perf_counter()
    y_score = predict_scores(app.state.model, X_df)[0]
    predict_ms = (time.perf_counter() - t0) * 1000
    request.state.predict_ms = predict_ms

    prediction = "churn" if y_score >= request.app.state.threshold else "no_churn"

    request.state.model_version = request.app.state.version
    request.state.outcome = prediction

    response = PredictResponse(
        request_id = request.state.request_id,
        probability = y_score,
        prediction = prediction,
        threshold = request.app.state.threshold,
        model_version = request.app.state.version,
        customer_id = payload.customer_id
    )

    try:
        t1 = time.perf_counter()
        insert_prediction(request.app.state.engine, payload, response, now)
        insert_ms = (time.perf_counter() - t1) * 1000
        
    except Exception as e:
        logger.exception("Failed to insert prediction in db")
        raise HTTPException(status_code=500, detail="Failed to insert prediction") from e
    finally: 
        request.state.insert_ms = insert_ms

    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)