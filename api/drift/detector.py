"""
ML Sentinel — Drift Detector
Uses Evidently AI to detect data drift and prediction drift
by comparing reference data against recent production data.
"""

import logging
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from config import get_settings
from monitoring.logger import prediction_logger
from monitoring.metrics import record_drift, DRIFT_SCORE, DRIFT_DETECTED

logger = logging.getLogger(__name__)

# Feature columns
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class DriftDetector:
    """
    Detects data drift by comparing reference (training) data
    against recent production predictions using Evidently AI.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._last_result: Optional[Dict[str, Any]] = None
        self._reference_df: Optional[pd.DataFrame] = None
    
    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result
    
    def load_reference_data(self) -> bool:
        """Load reference data from the database."""
        try:
            ref_data = prediction_logger.get_reference_data()
            if not ref_data:
                logger.warning("No reference data found in database")
                return False
            
            # Convert to DataFrame
            features_list = [item["features"] for item in ref_data]
            self._reference_df = pd.DataFrame(features_list)
            
            # Ensure correct column order
            available_cols = [c for c in FEATURE_COLUMNS if c in self._reference_df.columns]
            self._reference_df = self._reference_df[available_cols]
            
            logger.info(f"Loaded {len(self._reference_df)} reference samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            return False
    
    def detect_drift(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run drift detection comparing reference data vs recent predictions.
        
        Returns:
            Tuple of (drift_detected, details_dict)
        """
        logger.info("Running drift detection...")
        
        # Load reference data if not loaded
        if self._reference_df is None:
            if not self.load_reference_data():
                return False, {"error": "No reference data available"}
        
        # Get recent prediction data
        recent_predictions = prediction_logger.get_recent_predictions(
            limit=self.settings.current_data_window
        )
        
        if len(recent_predictions) < 50:
            logger.info(f"Not enough recent predictions for drift detection "
                       f"({len(recent_predictions)} < 50 minimum)")
            return False, {
                "status": "insufficient_data",
                "current_size": len(recent_predictions),
                "minimum_required": 50
            }
        
        # Build current DataFrame from recent predictions
        current_features = [item["features"] for item in recent_predictions]
        current_df = pd.DataFrame(current_features)
        
        # Ensure same columns
        available_cols = [c for c in FEATURE_COLUMNS 
                         if c in current_df.columns and c in self._reference_df.columns]
        current_df = current_df[available_cols]
        reference_df = self._reference_df[available_cols]
        
        try:
            # Run Evidently drift report
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(
                reference_data=reference_df,
                current_data=current_df
            )
            
            # Extract results from the report
            report_dict = drift_report.as_dict()
            
            # Parse drift results
            drift_result = self._parse_drift_report(report_dict)
            
            # Determine if drift threshold exceeded
            drift_detected = drift_result["drift_score"] > self.settings.drift_threshold
            drift_result["threshold"] = self.settings.drift_threshold
            drift_result["drift_detected"] = drift_detected
            drift_result["reference_size"] = len(reference_df)
            drift_result["current_size"] = len(current_df)
            
            # Update metrics
            record_drift(
                drift_score=drift_result["drift_score"],
                n_drifted=drift_result.get("n_drifted_features", 0),
                is_drifted=drift_detected,
                feature_scores=drift_result.get("feature_drift_scores", {})
            )
            
            # Store result
            self._last_result = drift_result
            
            # Log to database
            prediction_logger.log_drift_report(
                report_type="data_drift",
                dataset_drift=drift_detected,
                drift_score=drift_result["drift_score"],
                n_drifted=drift_result.get("n_drifted_features", 0),
                n_total=drift_result.get("n_total_features", len(available_cols)),
                feature_scores=drift_result.get("feature_drift_scores", {}),
                reference_size=len(reference_df),
                current_size=len(current_df),
                triggered_retraining=False  # Updated later if retraining triggers
            )
            
            status = "DRIFT DETECTED" if drift_detected else "No drift"
            logger.info(f"Drift detection complete: {status} "
                       f"(score: {drift_result['drift_score']:.4f}, "
                       f"threshold: {self.settings.drift_threshold})")
            
            return drift_detected, drift_result
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return False, {"error": str(e)}
    
    def _parse_drift_report(self, report_dict: dict) -> Dict[str, Any]:
        """Parse the Evidently report dictionary into a clean result."""
        result = {
            "drift_score": 0.0,
            "n_drifted_features": 0,
            "n_total_features": 0,
            "dataset_drift": False,
            "feature_drift_scores": {}
        }
        
        try:
            metrics = report_dict.get("metrics", [])
            
            for metric in metrics:
                metric_result = metric.get("result", {})
                
                # Dataset drift metric
                if "share_of_drifted_columns" in metric_result:
                    result["drift_score"] = metric_result.get("share_of_drifted_columns", 0.0)
                    result["n_drifted_features"] = metric_result.get("number_of_drifted_columns", 0)
                    result["n_total_features"] = metric_result.get("number_of_columns", 0)
                    result["dataset_drift"] = metric_result.get("dataset_drift", False)
                    
                    # Per-feature drift
                    drift_by_columns = metric_result.get("drift_by_columns", {})
                    for col_name, col_data in drift_by_columns.items():
                        result["feature_drift_scores"][col_name] = {
                            "drift_score": col_data.get("drift_score", 0),
                            "is_drifted": col_data.get("column_drifted", False),
                            "stattest_name": col_data.get("stattest_name", "unknown")
                        }
            
        except Exception as e:
            logger.error(f"Error parsing drift report: {e}")
        
        return result


# Global drift detector instance
drift_detector = DriftDetector()
