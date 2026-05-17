"""
ML Sentinel — Configuration Module
Centralized configuration using pydantic-settings for environment variable management.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql+psycopg2://sentinel:sentinel_secret_2024@postgres:5432/ml_sentinel"
    
    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_s3_endpoint_url: str = "http://minio:9000"
    aws_access_key_id: str = "minio_admin"
    aws_secret_access_key: str = "minio_secret_2024"
    
    # Model
    model_name: str = "fraud-detector"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Drift Detection
    drift_check_interval_minutes: int = 5
    drift_threshold: float = 0.5
    reference_data_size: int = 5000
    current_data_window: int = 1000
    
    # Retraining
    retrain_min_samples: int = 500
    retrain_improvement_threshold: float = 0.01
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
