"""
ML Sentinel — Initial Model Training Script
Trains the first fraud detection model, logs to MLflow, registers in the Model Registry,
and stores reference data in PostgreSQL for drift detection.
"""

import os
import sys
import json
import time
import logging
import pickle

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sqlalchemy import create_engine, text

from preprocess import load_data, preprocess, split_data, FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def store_reference_data(X_ref: pd.DataFrame, y_ref: pd.Series, 
                         model_version: str, db_url: str):
    """Store reference data in PostgreSQL for drift detection baseline."""
    logger.info(f"Storing {len(X_ref)} reference samples in PostgreSQL...")
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        # Clear old reference data
        conn.execute(text("DELETE FROM reference_data"))
        conn.commit()
        
        # Insert reference data in batches
        batch_size = 500
        for i in range(0, len(X_ref), batch_size):
            batch_X = X_ref.iloc[i:i + batch_size]
            batch_y = y_ref.iloc[i:i + batch_size]
            
            for j, (idx, row) in enumerate(batch_X.iterrows()):
                features_dict = row.to_dict()
                conn.execute(
                    text("""
                        INSERT INTO reference_data (model_version, features, label)
                        VALUES (:model_version, :features, :label)
                    """),
                    {
                        "model_version": model_version,
                        "features": json.dumps(features_dict),
                        "label": int(batch_y.iloc[j])
                    }
                )
            conn.commit()
            logger.info(f"  Inserted batch {i // batch_size + 1}")
    
    logger.info("Reference data stored successfully")


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    """Train a LightGBM classifier optimized for fraud detection."""
    logger.info("Training LightGBM classifier...")
    
    # Calculate scale_pos_weight for imbalanced classes
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive
    logger.info(f"Class balance — Negative: {n_negative}, Positive: {n_positive}, "
                f"scale_pos_weight: {scale_pos_weight:.1f}")
    
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_duration = time.time() - start_time
    logger.info(f"Training completed in {training_duration:.2f}s")
    
    return model, training_duration


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model and return metrics dictionary."""
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }
    
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 60)
    for name, value in metrics.items():
        logger.info(f"  {name:>12}: {value:.4f}")
    logger.info("=" * 60)
    
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return metrics


def main():
    """Main training pipeline."""
    # Configuration
    data_path = os.getenv("DATA_PATH", "/app/data/creditcard.csv")
    db_url = os.getenv("DATABASE_URL")
    model_name = os.getenv("MODEL_NAME", "fraud-detector")
    reference_size = int(os.getenv("REFERENCE_DATA_SIZE", "5000"))
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    
    logger.info("=" * 60)
    logger.info("ML SENTINEL — INITIAL MODEL TRAINING")
    logger.info("=" * 60)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please mount the creditcard.csv file in the /app/data directory")
        sys.exit(1)
    
    # Load and preprocess data
    df = load_data(data_path)
    X, y, scaler = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test, X_ref, y_ref = split_data(
        X, y, reference_size=reference_size
    )
    
    # Start MLflow experiment
    mlflow.set_experiment("fraud-detection")
    
    with mlflow.start_run(run_name="initial-training") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("reference_size", len(X_ref))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("fraud_rate_train", f"{y_train.mean():.4%}")
        
        # Train
        model, training_duration = train_model(X_train, y_train)
        mlflow.log_param("training_duration_s", round(training_duration, 2))
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # Cross-validation
        logger.info("Running 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_scores.std())
        logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Log model artifact
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log the scaler as artifact
        scaler_path = "/tmp/scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessor")
        
        # Log feature names
        mlflow.log_dict(
            {"features": FEATURE_COLUMNS, "target": TARGET_COLUMN},
            artifact_file="feature_config.json"
        )
        
        logger.info(f"Model registered as '{model_name}'")
    
    # Get the registered model version
    client = mlflow.MlflowClient()
    
    # Get latest version
    versions = client.search_model_versions(f"name='{model_name}'")
    if versions:
        latest_version = max(versions, key=lambda v: int(v.version))
        version_num = latest_version.version
        
        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version_num,
            stage="Production"
        )
        logger.info(f"Model version {version_num} promoted to Production")
    else:
        version_num = "1"
        logger.warning("Could not find registered model version")
    
    # Store reference data for drift detection
    if db_url:
        store_reference_data(X_ref, y_ref, str(version_num), db_url)
    else:
        logger.warning("DATABASE_URL not set — skipping reference data storage")
    
    logger.info("=" * 60)
    logger.info("INITIAL TRAINING COMPLETE")
    logger.info(f"  Model: {model_name} v{version_num}")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
