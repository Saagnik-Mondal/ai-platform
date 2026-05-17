"""
ML Sentinel — Pydantic Request/Response Schemas
Defines the data contracts for the inference API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Input features for fraud prediction.
    Credit card dataset has V1-V28 (PCA components) + Amount.
    """
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., ge=0, description="Transaction amount")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "V1": -1.3598071336738,
                    "V2": -0.0727811733098497,
                    "V3": 2.53634673796914,
                    "V4": 1.37815522427443,
                    "V5": -0.338320769942518,
                    "V6": 0.462387777762292,
                    "V7": 0.239598554061257,
                    "V8": 0.0986979012610507,
                    "V9": 0.363786969611213,
                    "V10": 0.0907941719789316,
                    "V11": -0.551599533260813,
                    "V12": -0.617800855762348,
                    "V13": -0.991389847235408,
                    "V14": -0.311169353699879,
                    "V15": 1.46817697209427,
                    "V16": -0.470400525259478,
                    "V17": 0.207971241929242,
                    "V18": 0.0257905801985591,
                    "V19": 0.403992960255733,
                    "V20": 0.251412098239705,
                    "V21": -0.018306777944153,
                    "V22": 0.277837575558899,
                    "V23": -0.110473910188767,
                    "V24": 0.0669280749146731,
                    "V25": 0.128539358273528,
                    "V26": -0.189114843888824,
                    "V27": 0.133558376740387,
                    "V28": -0.0210530534538215,
                    "Amount": 149.62
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response from fraud prediction endpoint."""
    request_id: str = Field(..., description="Unique request identifier")
    prediction: int = Field(..., description="0 = legitimate, 1 = fraud")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud")
    is_fraud: bool = Field(..., description="Whether the transaction is classified as fraud")
    model_version: str = Field(..., description="Model version used for prediction")
    model_name: str = Field(..., description="Model name")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    transactions: List[PredictionRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total: int
    fraud_count: int
    avg_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    database_connected: bool
    mlflow_connected: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Current model information."""
    model_name: str
    model_version: str
    loaded_at: datetime
    total_predictions: int
    stage: str = "Production"


class DriftStatusResponse(BaseModel):
    """Latest drift detection status."""
    timestamp: Optional[datetime] = None
    dataset_drift: bool = False
    drift_score: float = 0.0
    n_drifted_features: int = 0
    n_total_features: int = 0
    feature_scores: Optional[Dict[str, Any]] = None
    status: str = "no_reports"  # no_reports, stable, warning, drifted


class DriftReportResponse(BaseModel):
    """Response after triggering drift analysis."""
    status: str
    drift_detected: bool
    drift_score: float
    details: Dict[str, Any]
    retraining_triggered: bool


class RetrainingStatusResponse(BaseModel):
    """Retraining event information."""
    id: int
    timestamp: datetime
    trigger_reason: str
    old_model_version: Optional[str]
    new_model_version: Optional[str]
    status: str
    auc_roc: Optional[float]
    improvement: Optional[float]
    deployed: bool
