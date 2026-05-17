"""
ML Sentinel — FastAPI Inference Application
Production ML inference API with observability, drift detection, and auto-retraining.
"""

import uuid
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from config import get_settings
from models.predictor import predictor
from models.schema import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
    DriftStatusResponse, DriftReportResponse,
    RetrainingStatusResponse,
)
from monitoring.logger import prediction_logger
from monitoring.metrics import (
    record_prediction, update_model_info,
    API_UPTIME, DB_CONNECTED, MLFLOW_CONNECTED,
    MODEL_PREDICTIONS_SERVED,
)
from drift.detector import drift_detector
from drift.scheduler import start_scheduler, stop_scheduler
from retraining.pipeline import retraining_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

START_TIME = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    global START_TIME
    START_TIME = time.time()
    logger.info("=" * 60)
    logger.info("ML SENTINEL — Starting up")
    logger.info("=" * 60)

    # Connect to database
    prediction_logger.connect()
    DB_CONNECTED.set(1 if prediction_logger.is_connected else 0)

    # Load model from MLflow
    model_loaded = predictor.load_model()
    if model_loaded:
        update_model_info(predictor.model_name, predictor.model_version)
        MLFLOW_CONNECTED.set(1)
        logger.info(f"Model loaded: {predictor.model_name} v{predictor.model_version}")
    else:
        MLFLOW_CONNECTED.set(0)
        logger.warning("Model not loaded — API will return errors until model is available")

    # Load reference data for drift detection
    drift_detector.load_reference_data()

    # Start background scheduler
    start_scheduler()

    logger.info("ML Sentinel ready to serve predictions")
    yield

    # Shutdown
    logger.info("ML Sentinel shutting down...")
    stop_scheduler()


# Create FastAPI app
app = FastAPI(
    title="ML Sentinel",
    description=(
        "ML Observability & Auto-Retraining Platform. "
        "Serves fraud detection predictions with automated drift detection, "
        "model versioning, and retraining pipelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus auto-instrumentation
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")


# ──────────────────────────────────────────────
# Prediction Endpoints
# ──────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a fraud prediction for a single transaction."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    features = request.model_dump()

    prediction, probability, latency_ms = predictor.predict(features)

    # Record metrics
    record_prediction(
        predicted_class=prediction,
        probability=probability,
        latency_seconds=latency_ms / 1000,
        amount=features.get("Amount", 0),
        model_version=predictor.model_version,
    )
    MODEL_PREDICTIONS_SERVED.set(predictor.prediction_count)

    # Log to DB in background
    background_tasks.add_task(
        prediction_logger.log_prediction,
        request_id=request_id,
        model_version=predictor.model_version,
        model_name=predictor.model_name,
        features=features,
        prediction=prediction,
        probability=probability,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        request_id=request_id,
        prediction=prediction,
        fraud_probability=probability,
        is_fraud=prediction == 1,
        model_version=predictor.model_version,
        model_name=predictor.model_name,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Make fraud predictions for a batch of transactions."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    total_latency = 0

    for txn in request.transactions:
        request_id = str(uuid.uuid4())
        features = txn.model_dump()
        pred, prob, lat = predictor.predict(features)
        total_latency += lat

        record_prediction(pred, prob, lat / 1000, features.get("Amount", 0), predictor.model_version)

        background_tasks.add_task(
            prediction_logger.log_prediction,
            request_id=request_id,
            model_version=predictor.model_version,
            model_name=predictor.model_name,
            features=features,
            prediction=pred,
            probability=prob,
            latency_ms=lat,
        )

        predictions.append(PredictionResponse(
            request_id=request_id,
            prediction=pred,
            fraud_probability=prob,
            is_fraud=pred == 1,
            model_version=predictor.model_version,
            model_name=predictor.model_name,
            latency_ms=round(lat, 3),
        ))

    MODEL_PREDICTIONS_SERVED.set(predictor.prediction_count)

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=sum(1 for p in predictions if p.is_fraud),
        avg_latency_ms=round(total_latency / len(predictions), 3),
    )


# ──────────────────────────────────────────────
# Health & Info Endpoints
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """API health check."""
    uptime = time.time() - START_TIME if START_TIME else 0
    API_UPTIME.set(uptime)

    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        model_name=predictor.model_name if predictor.is_loaded else None,
        model_version=predictor.model_version if predictor.is_loaded else None,
        database_connected=prediction_logger.is_connected,
        mlflow_connected=predictor.is_loaded,
        uptime_seconds=round(uptime, 2),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Get current model information."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        loaded_at=predictor.loaded_at,
        total_predictions=predictor.prediction_count,
    )


# ──────────────────────────────────────────────
# Drift Detection Endpoints
# ──────────────────────────────────────────────

@app.get("/drift/status", response_model=DriftStatusResponse, tags=["Drift Detection"])
async def drift_status():
    """Get latest drift detection status."""
    report = prediction_logger.get_latest_drift_report()

    if not report:
        return DriftStatusResponse(status="no_reports")

    score = report.get("drift_score", 0)
    settings = get_settings()
    if score > settings.drift_threshold:
        status = "drifted"
    elif score > settings.drift_threshold * 0.7:
        status = "warning"
    else:
        status = "stable"

    return DriftStatusResponse(
        timestamp=report.get("timestamp"),
        dataset_drift=report.get("dataset_drift", False),
        drift_score=score,
        n_drifted_features=report.get("n_drifted_features", 0),
        n_total_features=report.get("n_total_features", 0),
        feature_scores=report.get("feature_scores"),
        status=status,
    )


@app.post("/drift/report", response_model=DriftReportResponse, tags=["Drift Detection"])
async def trigger_drift_report():
    """Trigger an on-demand drift analysis."""
    drift_detected, details = drift_detector.detect_drift()

    return DriftReportResponse(
        status="completed",
        drift_detected=drift_detected,
        drift_score=details.get("drift_score", 0),
        details=details,
        retraining_triggered=False,
    )


# ──────────────────────────────────────────────
# Retraining Endpoints
# ──────────────────────────────────────────────

@app.post("/retrain", tags=["Retraining"])
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Manually trigger model retraining."""
    if retraining_pipeline.is_running:
        raise HTTPException(status_code=409, detail="Retraining already in progress")

    background_tasks.add_task(retraining_pipeline.run, trigger_reason="manual")
    return {"status": "retraining_started", "trigger": "manual"}


@app.get("/retrain/history", tags=["Retraining"])
async def retraining_history():
    """Get retraining event history."""
    events = prediction_logger.get_retraining_history(limit=20)
    return {"events": events, "total": len(events)}
