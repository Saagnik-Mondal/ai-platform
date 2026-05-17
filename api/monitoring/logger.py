"""
ML Sentinel — Prediction Logger
Asynchronous logging of features and predictions to PostgreSQL.
Uses background tasks for non-blocking writes.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from config import get_settings

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logs prediction requests and responses to PostgreSQL.
    Uses connection pooling for efficient database access.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._engine = None
        self._connected = False
    
    def connect(self):
        """Initialize database connection pool."""
        try:
            self._engine = create_engine(
                self.settings.database_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800
            )
            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._connected = True
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def log_prediction(self, request_id: str, model_version: str, model_name: str,
                       features: dict, prediction: int, probability: float,
                       latency_ms: float):
        """Log a single prediction to the database."""
        if not self._connected:
            logger.warning("Database not connected, skipping prediction log")
            return
        
        try:
            with self._engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO prediction_logs 
                        (request_id, model_version, model_name, features, 
                         prediction, probability, latency_ms)
                        VALUES (:request_id, :model_version, :model_name, 
                                :features, :prediction, :probability, :latency_ms)
                    """),
                    {
                        "request_id": request_id,
                        "model_version": model_version,
                        "model_name": model_name,
                        "features": json.dumps(features),
                        "prediction": prediction,
                        "probability": probability,
                        "latency_ms": latency_ms
                    }
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def get_recent_predictions(self, limit: int = 1000) -> list:
        """Fetch recent predictions for drift detection."""
        if not self._connected:
            return []
        
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT features, prediction, probability
                        FROM prediction_logs
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                )
                rows = result.fetchall()
                return [
                    {
                        "features": json.loads(row[0]) if isinstance(row[0], str) else row[0],
                        "prediction": row[1],
                        "probability": row[2]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch recent predictions: {e}")
            return []
    
    def get_prediction_count(self) -> int:
        """Get total number of logged predictions."""
        if not self._connected:
            return 0
        
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM prediction_logs"))
                return result.scalar()
        except Exception as e:
            logger.error(f"Failed to get prediction count: {e}")
            return 0
    
    def get_reference_data(self, model_version: Optional[str] = None) -> list:
        """Fetch reference data for drift detection."""
        if not self._connected:
            return []
        
        try:
            with self._engine.connect() as conn:
                if model_version:
                    result = conn.execute(
                        text("""
                            SELECT features, label FROM reference_data
                            WHERE model_version = :version
                            ORDER BY id
                        """),
                        {"version": model_version}
                    )
                else:
                    result = conn.execute(
                        text("SELECT features, label FROM reference_data ORDER BY id")
                    )
                rows = result.fetchall()
                return [
                    {
                        "features": json.loads(row[0]) if isinstance(row[0], str) else row[0],
                        "label": row[1]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch reference data: {e}")
            return []
    
    def log_drift_report(self, report_type: str, dataset_drift: bool, 
                         drift_score: float, n_drifted: int, n_total: int,
                         feature_scores: dict, reference_size: int,
                         current_size: int, triggered_retraining: bool):
        """Log drift detection results."""
        if not self._connected:
            return
        
        try:
            with self._engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO drift_reports
                        (report_type, dataset_drift, drift_score, n_drifted_features,
                         n_total_features, feature_scores, reference_size, 
                         current_size, triggered_retraining)
                        VALUES (:report_type, :dataset_drift, :drift_score, 
                                :n_drifted, :n_total, :feature_scores,
                                :ref_size, :cur_size, :triggered)
                    """),
                    {
                        "report_type": report_type,
                        "dataset_drift": dataset_drift,
                        "drift_score": drift_score,
                        "n_drifted": n_drifted,
                        "n_total": n_total,
                        "feature_scores": json.dumps(feature_scores),
                        "ref_size": reference_size,
                        "cur_size": current_size,
                        "triggered": triggered_retraining
                    }
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log drift report: {e}")
    
    def get_latest_drift_report(self) -> Optional[dict]:
        """Get the most recent drift report."""
        if not self._connected:
            return None
        
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT timestamp, report_type, dataset_drift, drift_score,
                               n_drifted_features, n_total_features, feature_scores
                        FROM drift_reports
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """)
                )
                row = result.fetchone()
                if row:
                    return {
                        "timestamp": row[0],
                        "report_type": row[1],
                        "dataset_drift": row[2],
                        "drift_score": row[3],
                        "n_drifted_features": row[4],
                        "n_total_features": row[5],
                        "feature_scores": json.loads(row[6]) if isinstance(row[6], str) else row[6]
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to fetch latest drift report: {e}")
            return None
    
    def log_retraining_event(self, trigger_reason: str, old_version: str,
                             new_version: str, training_samples: int,
                             training_duration: float, metrics: dict,
                             old_auc: float, status: str, deployed: bool,
                             rejection_reason: str = None):
        """Log a retraining event."""
        if not self._connected:
            return
        
        try:
            with self._engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO retraining_events
                        (trigger_reason, old_model_version, new_model_version,
                         training_samples, training_duration_s, accuracy,
                         precision_score, recall, f1_score, auc_roc,
                         old_model_auc, improvement, status, deployed, rejection_reason)
                        VALUES (:trigger, :old_ver, :new_ver, :samples, :duration,
                                :accuracy, :precision, :recall, :f1, :auc,
                                :old_auc, :improvement, :status, :deployed, :reason)
                    """),
                    {
                        "trigger": trigger_reason,
                        "old_ver": old_version,
                        "new_ver": new_version,
                        "samples": training_samples,
                        "duration": training_duration,
                        "accuracy": metrics.get("accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1_score"),
                        "auc": metrics.get("auc_roc"),
                        "old_auc": old_auc,
                        "improvement": metrics.get("auc_roc", 0) - old_auc if old_auc else 0,
                        "status": status,
                        "deployed": deployed,
                        "reason": rejection_reason
                    }
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log retraining event: {e}")
    
    def get_retraining_history(self, limit: int = 10) -> list:
        """Get recent retraining events."""
        if not self._connected:
            return []
        
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT id, timestamp, trigger_reason, old_model_version,
                               new_model_version, status, auc_roc, improvement, deployed
                        FROM retraining_events
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                )
                return [
                    {
                        "id": row[0],
                        "timestamp": row[1],
                        "trigger_reason": row[2],
                        "old_model_version": row[3],
                        "new_model_version": row[4],
                        "status": row[5],
                        "auc_roc": row[6],
                        "improvement": row[7],
                        "deployed": row[8]
                    }
                    for row in result.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to fetch retraining history: {e}")
            return []


# Global logger instance
prediction_logger = PredictionLogger()
