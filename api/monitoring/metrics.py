"""
ML Sentinel — Prometheus Custom Metrics
Defines ML-specific metrics for monitoring model performance and system health.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import logging

logger = logging.getLogger(__name__)

# ================================================
# Prediction Metrics
# ================================================

# Total predictions counter with class labels
PREDICTION_TOTAL = Counter(
    "ml_sentinel_predictions_total",
    "Total number of predictions made",
    ["predicted_class", "model_version"]
)

# Prediction latency histogram (in seconds)
PREDICTION_LATENCY = Histogram(
    "ml_sentinel_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Fraud probability distribution histogram
PREDICTION_PROBABILITY = Histogram(
    "ml_sentinel_prediction_probability",
    "Distribution of fraud probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Transaction amount histogram
TRANSACTION_AMOUNT = Histogram(
    "ml_sentinel_transaction_amount",
    "Distribution of transaction amounts",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000]
)

# ================================================
# Model Metrics
# ================================================

# Current model version info
MODEL_INFO = Info(
    "ml_sentinel_model",
    "Current model information"
)

# Model version as a gauge (for Grafana queries)
MODEL_VERSION_GAUGE = Gauge(
    "ml_sentinel_model_version",
    "Current model version number"
)

# Total predictions served by current model
MODEL_PREDICTIONS_SERVED = Gauge(
    "ml_sentinel_model_predictions_served",
    "Total predictions served by the current model"
)

# ================================================
# Drift Metrics
# ================================================

# Overall drift score
DRIFT_SCORE = Gauge(
    "ml_sentinel_drift_score",
    "Latest overall drift score"
)

# Number of drifted features
DRIFT_FEATURES_COUNT = Gauge(
    "ml_sentinel_drift_features_count",
    "Number of features with detected drift"
)

# Whether dataset drift is detected (binary)
DRIFT_DETECTED = Gauge(
    "ml_sentinel_drift_detected",
    "Whether data drift is currently detected (1=yes, 0=no)"
)

# Per-feature drift scores
FEATURE_DRIFT_SCORE = Gauge(
    "ml_sentinel_feature_drift_score",
    "Drift score per feature",
    ["feature_name"]
)

# ================================================
# Retraining Metrics
# ================================================

# Total retraining events triggered
RETRAINING_TRIGGERED = Counter(
    "ml_sentinel_retraining_triggered_total",
    "Total number of retraining events triggered",
    ["trigger_reason"]
)

# Retraining outcome
RETRAINING_OUTCOME = Counter(
    "ml_sentinel_retraining_outcome_total",
    "Outcome of retraining events",
    ["status"]  # success, rejected, failed
)

# Latest retrained model AUC
RETRAINED_MODEL_AUC = Gauge(
    "ml_sentinel_retrained_model_auc",
    "AUC-ROC of the latest retrained model"
)

# ================================================
# System Metrics
# ================================================

# API uptime
API_UPTIME = Gauge(
    "ml_sentinel_api_uptime_seconds",
    "API uptime in seconds"
)

# Database connection status
DB_CONNECTED = Gauge(
    "ml_sentinel_db_connected",
    "Database connection status (1=connected, 0=disconnected)"
)

# MLflow connection status
MLFLOW_CONNECTED = Gauge(
    "ml_sentinel_mlflow_connected",
    "MLflow connection status (1=connected, 0=disconnected)"
)


def update_model_info(model_name: str, model_version: str):
    """Update model info metrics."""
    MODEL_INFO.info({
        "name": model_name,
        "version": model_version,
    })
    try:
        MODEL_VERSION_GAUGE.set(float(model_version))
    except (ValueError, TypeError):
        MODEL_VERSION_GAUGE.set(0)


def record_prediction(predicted_class: int, probability: float, 
                      latency_seconds: float, amount: float, model_version: str):
    """Record metrics for a single prediction."""
    PREDICTION_TOTAL.labels(
        predicted_class=str(predicted_class),
        model_version=model_version
    ).inc()
    PREDICTION_LATENCY.observe(latency_seconds)
    PREDICTION_PROBABILITY.observe(probability)
    TRANSACTION_AMOUNT.observe(amount)


def record_drift(drift_score: float, n_drifted: int, is_drifted: bool,
                 feature_scores: dict = None):
    """Record drift detection metrics."""
    DRIFT_SCORE.set(drift_score)
    DRIFT_FEATURES_COUNT.set(n_drifted)
    DRIFT_DETECTED.set(1 if is_drifted else 0)
    
    if feature_scores:
        for feature_name, score in feature_scores.items():
            if isinstance(score, (int, float)):
                FEATURE_DRIFT_SCORE.labels(feature_name=feature_name).set(score)
