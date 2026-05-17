# 🛡️ ML Sentinel — ML Observability & Auto-Retraining Platform

> Built an end-to-end ML observability platform with automated drift detection, retraining pipelines, model versioning, and deployment monitoring for production inference services.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-2.16-0194E2?logo=mlflow)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?logo=grafana)
![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🏗️ Architecture

```
Data → Preprocessing → Model Training → API Serving → Logging → Monitoring → Drift Detection → Retraining → Model Registry → Redeploy
```

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   Client     │────▶│  FastAPI API  │────▶│ PostgreSQL │
│  (Requests)  │     │  /predict    │     │  (Logging)  │
└─────────────┘     └──────┬───────┘     └─────┬──────┘
                           │                    │
                    ┌──────▼───────┐     ┌─────▼──────┐
                    │  Prometheus   │     │   Drift     │
                    │  (Metrics)    │     │  Detector   │
                    └──────┬───────┘     │ (Evidently) │
                           │             └─────┬──────┘
                    ┌──────▼───────┐           │
                    │   Grafana     │     ┌─────▼──────┐
                    │ (Dashboards)  │     │ Retraining  │
                    └──────────────┘     │  Pipeline   │
                                         └─────┬──────┘
                    ┌──────────────┐           │
                    │    MLflow     │◀──────────┘
                    │   (Registry)  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    MinIO      │
                    │ (Artifacts)   │
                    └──────────────┘
```

## ✨ Features

### Core Inference
- **Real-time fraud detection** via REST API (FastAPI)
- **Batch prediction** support for bulk transactions
- **Sub-10ms latency** per prediction

### Observability
- **Prometheus metrics** — latency percentiles, throughput, prediction distributions
- **Grafana dashboards** — 16 pre-built panels auto-provisioned on startup
- **Prediction logging** — every prediction stored with features, output, and metadata

### Drift Detection
- **Automated data drift monitoring** using Evidently AI
- **Per-feature drift scores** with statistical tests (KS, Chi-Square)
- **Configurable thresholds** and check intervals

### Auto-Retraining
- **Drift-triggered retraining** — automatically retrain when drift exceeds threshold
- **Scheduled retraining** — periodic retraining every 6 hours
- **Quality gate** — new model only deployed if it outperforms current model
- **Manual trigger** — on-demand retraining via API endpoint

### Model Management
- **MLflow Model Registry** — full experiment tracking and model versioning
- **Hot-reload** — API automatically picks up new model versions without restart
- **Artifact storage** — MinIO (S3-compatible) for model artifacts

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| API | FastAPI + Uvicorn | High-performance inference serving |
| ML Model | LightGBM | Fraud detection classifier |
| Database | PostgreSQL 17 | Prediction logs, drift reports, reference data |
| Model Registry | MLflow 2.16 | Experiment tracking, model versioning |
| Artifact Store | MinIO | S3-compatible model artifact storage |
| Metrics | Prometheus | Time-series metrics collection |
| Dashboards | Grafana | Real-time visualization and alerting |
| Drift Detection | Evidently AI | Statistical drift analysis |
| Scheduler | APScheduler | Background job scheduling |
| Containerization | Docker Compose | Multi-service orchestration |
| Dataset | Credit Card Fraud | 284,807 transactions, 492 frauds |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- ~2GB disk space for containers and data
- The `creditcard.csv` dataset in the `data/` directory

### 1. Clone and Setup

```bash
git clone https://github.com/Saagnik-Mondal/ai-platform.git
cd ai-platform
```

### 2. Add the Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/` directory:

```bash
cp /path/to/creditcard.csv ./data/
```

### 3. Launch All Services

```bash
docker compose up -d --build
```

This starts all 7 services:
- **PostgreSQL** → `localhost:5432`
- **MinIO Console** → `localhost:9001`
- **MLflow UI** → `localhost:5000`
- **Inference API** → `localhost:8000`
- **Prometheus** → `localhost:9090`
- **Grafana** → `localhost:3000`

### 4. Train the Initial Model

The trainer container runs automatically on first startup. Check progress:

```bash
docker logs -f sentinel-trainer
```

### 5. Make Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/predict` | Single transaction fraud prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 100) |
| `GET` | `/health` | Service health check |
| `GET` | `/model/info` | Current model version and stats |
| `GET` | `/metrics` | Prometheus metrics endpoint |
| `GET` | `/drift/status` | Latest drift detection status |
| `POST` | `/drift/report` | Trigger on-demand drift analysis |
| `POST` | `/retrain` | Manually trigger retraining |
| `GET` | `/retrain/history` | View retraining event history |
| `GET` | `/docs` | Interactive Swagger documentation |

---

## 📊 Grafana Dashboard

The dashboard is **auto-provisioned** on startup with 16 panels:

- **Predictions/sec** — Real-time throughput
- **Latency percentiles** — p50, p95, p99
- **Fraud probability distribution** — Model confidence histogram
- **Drift score timeline** — With threshold markers
- **Retraining events** — Trigger history and outcomes
- **Model version tracking** — Current deployment info
- **System health** — DB/MLflow connection status

Access at `http://localhost:3000` (admin/sentinel_grafana)

---

## 🔄 How Auto-Retraining Works

```
1. Scheduler runs drift check every 5 minutes
2. Evidently AI compares reference data vs recent predictions
3. If drift_score > 0.5 (configurable threshold):
   a. Fetch training data from PostgreSQL
   b. Train new LightGBM model
   c. Evaluate on holdout test set
   d. GATE: Only deploy if AUC improves by > 0.01
   e. Register in MLflow Model Registry
   f. Promote to "Production" stage
   g. API hot-reloads new model automatically
4. Additionally, scheduled retraining runs every 6 hours
```

---

## 📁 Project Structure

```
ml-sentinel/
├── docker-compose.yml          # Service orchestration
├── .env                        # Configuration
├── api/                        # FastAPI inference service
│   ├── main.py                 # Application + endpoints
│   ├── config.py               # Settings management
│   ├── models/                 # ML model loading
│   ├── monitoring/             # Metrics + logging
│   ├── drift/                  # Drift detection + scheduler
│   └── retraining/             # Auto-retraining pipeline
├── training/                   # Initial model training
├── db/                         # Database schema
├── monitoring/                 # Prometheus + Grafana configs
│   ├── prometheus/
│   └── grafana/provisioning/
└── data/                       # Dataset directory
```

---

## ⚙️ Configuration

Key environment variables in `.env`:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `DRIFT_CHECK_INTERVAL_MINUTES` | 5 | How often to check for drift |
| `DRIFT_THRESHOLD` | 0.5 | Drift score threshold for retraining |
| `RETRAIN_IMPROVEMENT_THRESHOLD` | 0.01 | Minimum AUC improvement required |
| `RETRAIN_MIN_SAMPLES` | 500 | Minimum samples needed to retrain |

---

## 🧹 Cleanup

```bash
# Stop all services
docker compose down

# Stop and remove volumes (reset all data)
docker compose down -v
```

---

## 📝 License

MIT License — feel free to use this project for learning and portfolio purposes.

---

*Built as a demonstration of production ML engineering — covering the full lifecycle from model training to automated monitoring and self-healing deployment.*
