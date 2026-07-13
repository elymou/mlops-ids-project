# MLOps Pipeline for Cybersecurity — Intrusion Detection System (IDS)

End-to-end MLOps pipeline that trains, deploys, monitors, and tracks a machine learning model for detecting network intrusions.

---

## Overview

Traditional signature-based Intrusion Detection Systems (IDS) struggle against zero-day attacks and evolving anomalous behavior. This project builds a **ML-based IDS** wrapped in a full MLOps pipeline — covering the entire lifecycle from raw data to a monitored production API.

**Pipeline capabilities:**
1. Preprocesses raw network traffic data (NSL-KDD)
2. Trains and compares supervised ML models (Random Forest, XGBoost, SVM)
3. Tracks experiments and model versions with **MLflow**
4. Serves real-time predictions via a **FastAPI** REST API
5. Containerizes all services with **Docker** / **docker-compose**
6. Automates testing and deployment with **GitHub Actions (CI/CD)**
7. Monitors live API performance with **Prometheus** and **Grafana**
8. Detects data drift between training and production data with **Evidently AI**

---

## Architecture

```
                ┌─────────────────────┐
                │   Raw Network Data   │
                │   (NSL-KDD dataset)  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Data Preprocessing   │
                │ (encode, scale)      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Model Training      │
                │ (RF, XGBoost, SVM)    │
                │  + MLflow tracking    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Best Model Saved     │
                │  (best_model.pkl)     │
                └──────────┬───────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │          FastAPI REST API         │
        │   /predict  /health  /model-info  │
        │   exposes Prometheus metrics      │
        └───────┬───────────────┬──────────┘
                │               │
                ▼               ▼
      ┌──────────────┐   ┌──────────────────┐
      │  Prometheus   │──▶│      Grafana      │
      │ (scrapes API) │   │ (live dashboards) │
      └──────────────┘   └──────────────────┘

      ┌───────────────────────────────────┐
      │           Evidently AI              │
      │  Compares training vs test data     │
      │  → drift report (HTML)              │
      └───────────────────────────────────┘

        All services orchestrated together
              via docker-compose.yml
```

**Services:**
| Service | Role |
|---|---|
| `api` | Serves intrusion predictions via FastAPI |
| `mlflow` | Tracks experiments, metrics, and model versions |
| `prometheus` | Scrapes live metrics from the API |
| `grafana` | Visualizes metrics on real-time dashboards |
| `evidently` (script) | Generates a data drift report between train/test sets |

---

## Project Structure

```
mlops-ids-project/
│
├── data/
│   └── raw/
│       ├── KDDTrain_.txt
│       └── KDDTest_.txt
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── api/
│   └── main.py                  # FastAPI app (/predict, /health, /model-info)
│
├── models/
│   ├── best_model.pkl            # trained XGBoost classifier
│   ├── scaler.pkl                 # fitted StandardScaler (41 features)
│   └── encoders.pkl               # fitted LabelEncoders (protocol_type, service, flag)
│
├── tests/
│   └── test_api.py
│
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana/
│   └── evidently_report.py       # data drift analysis
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Dataset

**NSL-KDD**, a standard benchmark for network intrusion detection research.

- Source: https://www.unb.ca/cic/datasets/nsl.html
- Files: `KDDTrain_.txt` (125,973 rows), `KDDTest_.txt` (22,544 rows)
- Place both files in `data/raw/`

---

## Requirements

```txt
pandas
numpy
scikit-learn
xgboost
mlflow
fastapi
uvicorn
prometheus-fastapi-instrumentator
evidently
pytest
requests
joblib
```

Install:
```bash
pip install -r requirements.txt
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/elymou/mlops-ids-project.git
cd mlops-ids-project

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Preprocess the data
python src/data_preprocessing.py

# 5. Train the models
python src/train.py

# 6. Launch all services
docker compose up --build
```

> Deactivate the virtual environment anytime with `deactivate`. The `venv/` folder should be excluded from Git via `.gitignore`.

Once running:
| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## API Usage

The `/predict` endpoint expects a single `features` field — an array of 41 preprocessed (encoded + scaled) numeric values:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.12, 0.0, 0.87, ... , 0.03]}'
```

`/model-info` and `/health` are available for status checks; interactive schema docs are at `/docs`.

---

## Monitoring

- **Prometheus** scrapes request counts, latency, and custom counters (`ids_normal_traffic_total`, `ids_attacks_detected_total`) from the API's `/metrics` endpoint.
- **Grafana** dashboards visualize total requests, request rate, normal traffic, and detected attacks in real time.
- **Evidently AI** (`monitoring/evidently_report.py`) compares the training and test datasets and generates an HTML drift report:

```bash
cd monitoring
python3 evidently_report.py
```

This produces `evidently_report.html`, flagging which features have drifted between the reference (training) and current (test) distributions using the Wasserstein distance test.

---

## Scope & Limitations

This project implements a **prediction API and MLOps lifecycle** around a trained IDS model — it does not perform live network packet capture. The API returns a prediction for whatever `features` array it receives; connecting it to a real-time packet sniffer (e.g., Zeek or Suricata) to extract features from live traffic automatically is outside this project's scope but would be the natural next step for production deployment.

---

## Author

Génie Informatique (CI2) — Spécialité Cybersécurité — Année 2025-2026
