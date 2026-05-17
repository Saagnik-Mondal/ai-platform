-- ================================================
-- ML Sentinel — Database Schema
-- ================================================

-- Create separate database for MLflow if it doesn't exist
-- (handled by Docker entrypoint scripts)

-- ================================================
-- Prediction Logs
-- Stores every prediction made by the inference API
-- ================================================
CREATE TABLE IF NOT EXISTS prediction_logs (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(50) NOT NULL,
    model_name      VARCHAR(100) NOT NULL DEFAULT 'fraud-detector',
    
    -- Input features (credit card dataset has V1-V28 + Amount)
    features        JSONB NOT NULL,
    
    -- Model output
    prediction      INTEGER NOT NULL,          -- 0 or 1
    probability     DOUBLE PRECISION NOT NULL,  -- fraud probability
    
    -- Performance
    latency_ms      DOUBLE PRECISION,
    
    -- Optional ground truth (filled later if available)
    actual_label    INTEGER,
    label_timestamp TIMESTAMPTZ
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp 
    ON prediction_logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version 
    ON prediction_logs (model_version);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_prediction 
    ON prediction_logs (prediction);

-- ================================================
-- Drift Reports
-- Stores results of each drift detection run
-- ================================================
CREATE TABLE IF NOT EXISTS drift_reports (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    report_type     VARCHAR(50) NOT NULL,       -- 'data_drift', 'prediction_drift'
    
    -- Overall drift metrics
    dataset_drift   BOOLEAN NOT NULL DEFAULT FALSE,
    drift_score     DOUBLE PRECISION NOT NULL,
    n_drifted_features INTEGER DEFAULT 0,
    n_total_features   INTEGER DEFAULT 0,
    
    -- Per-feature drift details
    feature_scores  JSONB,                      -- {feature_name: {score, is_drifted, test_name}}
    
    -- Data window info
    reference_size  INTEGER,
    current_size    INTEGER,
    
    -- Action taken
    triggered_retraining BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_timestamp 
    ON drift_reports (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_drift_reports_drift 
    ON drift_reports (dataset_drift);

-- ================================================
-- Retraining Events
-- Tracks every retraining attempt and its outcome
-- ================================================
CREATE TABLE IF NOT EXISTS retraining_events (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_reason      VARCHAR(100) NOT NULL,   -- 'drift_detected', 'scheduled', 'manual'
    
    -- Model versions
    old_model_version   VARCHAR(50),
    new_model_version   VARCHAR(50),
    
    -- Training metrics
    training_samples    INTEGER,
    training_duration_s DOUBLE PRECISION,
    
    -- Evaluation metrics (new model)
    accuracy            DOUBLE PRECISION,
    precision_score     DOUBLE PRECISION,
    recall              DOUBLE PRECISION,
    f1_score            DOUBLE PRECISION,
    auc_roc             DOUBLE PRECISION,
    
    -- Comparison with old model
    old_model_auc       DOUBLE PRECISION,
    improvement         DOUBLE PRECISION,        -- new_auc - old_auc
    
    -- Outcome
    status              VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, success, rejected, failed
    rejection_reason    TEXT,
    deployed            BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_retraining_events_timestamp 
    ON retraining_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_retraining_events_status 
    ON retraining_events (status);

-- ================================================
-- Reference Data
-- Stores the reference/baseline data for drift detection
-- ================================================
CREATE TABLE IF NOT EXISTS reference_data (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(50) NOT NULL,
    features        JSONB NOT NULL,
    label           INTEGER
);

CREATE INDEX IF NOT EXISTS idx_reference_data_model_version 
    ON reference_data (model_version);

-- ================================================
-- Create MLflow database
-- ================================================
SELECT 'ML Sentinel schema initialized successfully' AS status;
