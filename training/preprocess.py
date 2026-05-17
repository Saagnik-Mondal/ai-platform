"""
ML Sentinel — Data Preprocessing Module
Handles loading, cleaning, and splitting the credit card fraud dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Credit card dataset feature columns (V1-V28 + Amount)
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Class distribution:\n{df[TARGET_COLUMN].value_counts()}")
    logger.info(f"Fraud rate: {df[TARGET_COLUMN].mean():.4%}")
    return df


def preprocess(df: pd.DataFrame, fit_scaler: bool = True, scaler: StandardScaler = None):
    """
    Preprocess the dataset:
    - Scale the Amount feature (V1-V28 are already PCA-transformed)
    - Drop Time column (not useful for fraud detection)
    
    Returns:
        X: Feature matrix
        y: Target vector
        scaler: Fitted StandardScaler (for Amount)
    """
    logger.info("Preprocessing data...")
    
    # Drop Time column if present
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Scale Amount (V1-V28 are already normalized via PCA)
    if fit_scaler:
        scaler = StandardScaler()
        X["Amount"] = scaler.fit_transform(X[["Amount"]])
    elif scaler is not None:
        X["Amount"] = scaler.transform(X[["Amount"]])
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               reference_size: int = 5000, random_state: int = 42):
    """
    Split data into train, test, and reference sets.
    Reference set is used as baseline for drift detection.
    
    Returns:
        X_train, X_test, y_train, y_test, X_reference, y_reference
    """
    logger.info(f"Splitting data: test_size={test_size}, reference_size={reference_size}")
    
    # First split: train+reference vs test
    X_train_ref, X_test, y_train_ref, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs reference (from the training portion)
    if reference_size < len(X_train_ref):
        X_train, X_reference, y_train, y_reference = train_test_split(
            X_train_ref, y_train_ref, 
            test_size=reference_size, 
            random_state=random_state, 
            stratify=y_train_ref
        )
    else:
        X_train = X_train_ref
        y_train = y_train_ref
        X_reference = X_train_ref.sample(n=reference_size, random_state=random_state)
        y_reference = y_train_ref.loc[X_reference.index]
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Reference: {len(X_reference)}")
    return X_train, X_test, y_train, y_test, X_reference, y_reference
