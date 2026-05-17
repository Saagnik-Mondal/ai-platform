"""
ML Sentinel — Model Trainer
Encapsulated training logic for LightGBM fraud detection model.
"""

import time
import logging
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[LGBMClassifier, float]:
    """Train a LightGBM classifier for fraud detection."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    logger.info(f"Training LightGBM — {len(X_train)} samples, scale_pos_weight: {scale_pos_weight:.1f}")
    
    model = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    logger.info(f"Training completed in {duration:.2f}s")
    return model, duration


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }
    
    logger.info("Evaluation: " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    return metrics
