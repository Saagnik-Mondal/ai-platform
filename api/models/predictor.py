"""
ML Sentinel — Model Predictor
Thread-safe model loading from MLflow with hot-reload capability.
"""

import time
import threading
import logging
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from config import get_settings

logger = logging.getLogger(__name__)

# Feature columns for the credit card dataset
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class ModelPredictor:
    """
    Thread-safe model predictor with hot-reload from MLflow Model Registry.
    
    - Loads the 'Production' stage model on startup
    - Periodically checks for new versions
    - Swaps model atomically using a read-write lock
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._model_version: Optional[str] = None
        self._model_name: str = self.settings.model_name
        self._loaded_at: Optional[datetime] = None
        self._lock = threading.RLock()
        self._prediction_count: int = 0
        self._client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
        
        # Set MLflow env
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def model_version(self) -> Optional[str]:
        return self._model_version
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at
    
    @property
    def prediction_count(self) -> int:
        return self._prediction_count
    
    def load_model(self) -> bool:
        """
        Load the production model from MLflow Model Registry.
        Returns True if model was loaded/updated, False otherwise.
        """
        try:
            logger.info(f"Loading model '{self._model_name}' from MLflow...")
            
            # Get the latest production version
            versions = self._client.get_latest_versions(
                name=self._model_name, 
                stages=["Production"]
            )
            
            if not versions:
                # Try to get any version
                versions = self._client.get_latest_versions(
                    name=self._model_name, 
                    stages=["None", "Staging", "Production"]
                )
            
            if not versions:
                logger.warning(f"No versions found for model '{self._model_name}'")
                return False
            
            latest = versions[0]
            new_version = latest.version
            
            # Skip if same version already loaded
            if self._model_version == new_version:
                logger.debug(f"Model v{new_version} already loaded, skipping")
                return False
            
            # Load the model
            model_uri = f"models:/{self._model_name}/{new_version}"
            new_model = mlflow.pyfunc.load_model(model_uri)
            
            # Atomic swap
            with self._lock:
                old_version = self._model_version
                self._model = new_model
                self._model_version = new_version
                self._loaded_at = datetime.utcnow()
            
            if old_version:
                logger.info(f"Model hot-reloaded: v{old_version} → v{new_version}")
            else:
                logger.info(f"Model loaded: v{new_version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: dict) -> Tuple[int, float, float]:
        """
        Make a prediction for a single transaction.
        
        Args:
            features: Dictionary of feature values (V1-V28, Amount)
            
        Returns:
            Tuple of (prediction, probability, latency_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.perf_counter()
        
        # Build feature vector
        feature_values = [features.get(col, 0.0) for col in FEATURE_COLUMNS]
        df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
        
        with self._lock:
            # MLflow pyfunc predict returns predictions directly
            raw_prediction = self._model.predict(df)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Handle different model output formats
        if hasattr(raw_prediction, 'ndim') and raw_prediction.ndim > 1:
            prediction = int(raw_prediction[0][0])
            probability = float(raw_prediction[0][0])
        else:
            prediction = int(raw_prediction[0])
            # For classification, we need probability — try predict_proba
            probability = self._get_probability(df)
        
        self._prediction_count += 1
        
        return prediction, probability, latency_ms
    
    def _get_probability(self, df: pd.DataFrame) -> float:
        """Get prediction probability using the unwrapped model."""
        try:
            with self._lock:
                unwrapped = self._model._model_impl
                if hasattr(unwrapped, 'predict_proba'):
                    proba = unwrapped.predict_proba(df)
                    return float(proba[0][1])  # Probability of class 1 (fraud)
        except Exception as e:
            logger.debug(f"Could not get probability: {e}")
        return 0.0
    
    def predict_batch(self, features_list: list) -> list:
        """
        Make predictions for a batch of transactions.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of (prediction, probability, latency_ms) tuples
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        
        return results
    
    def check_for_update(self) -> bool:
        """
        Check if a newer model version is available and load it.
        Called periodically by the scheduler.
        """
        try:
            versions = self._client.get_latest_versions(
                name=self._model_name,
                stages=["Production"]
            )
            
            if not versions:
                return False
            
            latest = versions[0]
            if latest.version != self._model_version:
                logger.info(f"New model version detected: v{latest.version} "
                          f"(current: v{self._model_version})")
                return self.load_model()
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for model update: {e}")
            return False


# Global predictor instance
predictor = ModelPredictor()
