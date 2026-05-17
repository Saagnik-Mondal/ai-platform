"""
ML Sentinel — Auto-Retraining Pipeline
Full retraining workflow: fetch data → train → evaluate → gate → register → deploy.
"""

import json
import logging
import threading
from typing import Optional

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from config import get_settings
from monitoring.logger import prediction_logger
from monitoring.metrics import (
    RETRAINING_TRIGGERED, RETRAINING_OUTCOME, RETRAINED_MODEL_AUC
)
from retraining.trainer import train_model, evaluate_model, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Automated retraining pipeline that:
    1. Fetches recent production data from PostgreSQL
    2. Combines with reference/historical data
    3. Trains a new model
    4. Evaluates against holdout test set
    5. Gates: only promotes if new model beats current
    6. Registers in MLflow and promotes to Production
    """

    def __init__(self):
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def run(self, trigger_reason: str = "drift_detected") -> bool:
        """
        Execute the full retraining pipeline.
        Returns True if a new model was deployed, False otherwise.
        """
        if self._is_running:
            logger.warning("Retraining already in progress, skipping")
            return False

        with self._lock:
            self._is_running = True

        try:
            logger.info("=" * 60)
            logger.info(f"RETRAINING PIPELINE — Trigger: {trigger_reason}")
            logger.info("=" * 60)

            RETRAINING_TRIGGERED.labels(trigger_reason=trigger_reason).inc()

            # Step 1: Gather training data
            X_train, X_test, y_train, y_test = self._prepare_data()
            if X_train is None:
                logger.warning("Insufficient data for retraining")
                RETRAINING_OUTCOME.labels(status="failed").inc()
                return False

            # Step 2: Get current model performance baseline
            old_auc = self._get_current_model_auc(X_test, y_test)

            # Step 3: Train new model
            model, duration = train_model(X_train, y_train)

            # Step 4: Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            new_auc = metrics["auc_roc"]
            RETRAINED_MODEL_AUC.set(new_auc)

            # Step 5: Gate — check improvement
            from models.predictor import predictor
            old_version = predictor.model_version or "0"
            improvement = new_auc - old_auc if old_auc else 0

            if old_auc and improvement < self.settings.retrain_improvement_threshold:
                logger.info(f"New model AUC ({new_auc:.4f}) does not sufficiently "
                           f"improve over current ({old_auc:.4f}). "
                           f"Improvement: {improvement:.4f} < "
                           f"threshold: {self.settings.retrain_improvement_threshold}")

                prediction_logger.log_retraining_event(
                    trigger_reason=trigger_reason,
                    old_version=old_version,
                    new_version=None,
                    training_samples=len(X_train),
                    training_duration=duration,
                    metrics=metrics,
                    old_auc=old_auc or 0,
                    status="rejected",
                    deployed=False,
                    rejection_reason=f"Insufficient improvement: {improvement:.4f}"
                )
                RETRAINING_OUTCOME.labels(status="rejected").inc()
                return False

            # Step 6: Register in MLflow
            new_version = self._register_model(model, metrics, trigger_reason)
            if not new_version:
                RETRAINING_OUTCOME.labels(status="failed").inc()
                return False

            # Step 7: Log event
            prediction_logger.log_retraining_event(
                trigger_reason=trigger_reason,
                old_version=old_version,
                new_version=new_version,
                training_samples=len(X_train),
                training_duration=duration,
                metrics=metrics,
                old_auc=old_auc or 0,
                status="success",
                deployed=True
            )

            RETRAINING_OUTCOME.labels(status="success").inc()
            logger.info(f"Retraining SUCCESS — New model v{new_version} deployed "
                       f"(AUC: {new_auc:.4f}, improvement: {improvement:+.4f})")
            return True

        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            RETRAINING_OUTCOME.labels(status="failed").inc()
            return False
        finally:
            self._is_running = False

    def _prepare_data(self):
        """Fetch and prepare training data from reference + recent predictions."""
        # Get reference data
        ref_data = prediction_logger.get_reference_data()
        if not ref_data:
            logger.error("No reference data available")
            return None, None, None, None

        ref_features = pd.DataFrame([r["features"] for r in ref_data])
        ref_labels = pd.Series([r["label"] for r in ref_data])

        # Get recent production data (if ground truth available)
        recent = prediction_logger.get_recent_predictions(
            limit=self.settings.retrain_min_samples * 2
        )

        if len(recent) >= self.settings.retrain_min_samples:
            recent_features = pd.DataFrame([r["features"] for r in recent])
            recent_labels = pd.Series([r["prediction"] for r in recent])

            # Combine reference + recent data
            cols = [c for c in FEATURE_COLUMNS
                    if c in ref_features.columns and c in recent_features.columns]
            X = pd.concat([ref_features[cols], recent_features[cols]], ignore_index=True)
            y = pd.concat([ref_labels, recent_labels], ignore_index=True)
        else:
            cols = [c for c in FEATURE_COLUMNS if c in ref_features.columns]
            X = ref_features[cols]
            y = ref_labels

        if len(X) < 100:
            logger.warning(f"Only {len(X)} samples available, need at least 100")
            return None, None, None, None

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training data: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test

    def _get_current_model_auc(self, X_test, y_test) -> Optional[float]:
        """Get current production model's AUC on the test set."""
        try:
            from models.predictor import predictor
            if not predictor.is_loaded:
                return None

            predictions = []
            for _, row in X_test.iterrows():
                _, prob, _ = predictor.predict(row.to_dict())
                predictions.append(prob)

            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, predictions)
            logger.info(f"Current model AUC on test set: {auc:.4f}")
            return auc
        except Exception as e:
            logger.warning(f"Could not evaluate current model: {e}")
            return None

    def _register_model(self, model, metrics: dict, trigger: str) -> Optional[str]:
        """Register new model in MLflow and promote to Production."""
        try:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment("fraud-detection")

            with mlflow.start_run(run_name=f"retrain-{trigger}") as run:
                mlflow.log_params({
                    "trigger_reason": trigger,
                    "model_type": "LightGBM",
                    "n_estimators": 200,
                })
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)

                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=self.settings.model_name
                )

            # Promote to Production
            client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
            versions = client.search_model_versions(
                f"name='{self.settings.model_name}'"
            )
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                client.transition_model_version_stage(
                    name=self.settings.model_name,
                    version=latest.version,
                    stage="Production"
                )
                logger.info(f"Model v{latest.version} promoted to Production")
                return latest.version

            return None
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None


# Global pipeline instance
retraining_pipeline = RetrainingPipeline()
