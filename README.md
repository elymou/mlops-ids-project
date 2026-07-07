# MLOps IDS Project

End-to-end MLOps pipeline for a network intrusion detection system, built for the Machine Learning module (CI2 - Cybersecurity, Semester 4, 2025-2026).

The idea is pretty simple: train a model that can tell if network traffic is normal or an attack, then wrap the whole thing with the tooling you'd actually need in a real environment - experiment tracking, an API to serve predictions, containers, CI/CD, and monitoring.

## Dataset

NSL-KDD (`KDDTrain+` / `KDDTest+`). It's the classic benchmark for this kind of thing - 41 features per connection (protocol, service, byte counts, error rates, etc.) plus a label saying whether it's normal traffic or a specific attack type.

Not included in the repo directly due to size - download from https://www.unb.ca/cic/datasets/nsl.html and drop the files in `data/raw/`.

## What's in here

```
mlops-ids-project/
├── data/raw/              KDDTrain+ and KDDTest+ go here
├── src/
│   ├── data_preprocessing.py
│   └── train.py
├── api/
│   └── main.py            FastAPI app, serves predictions
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana/
│   └── evidently_report.py
├── tests/
├── .github/workflows/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Models

Trained and compared Random Forest, XGBoost, and SVM on the preprocessed dataset. XGBoost came out on top and is what's saved as `best_model.pkl`. All the runs (params, metrics) are logged in MLflow under the `IDS-Cybersecurity` experiment.

`scaler.pkl` and `encoders.pkl` are needed alongside the model - they hold the fitted preprocessing (StandardScaler + label encoders for `protocol_type`, `service`, `flag`) so incoming requests get transformed the same way the training data was.

## Running it

```bash
# preprocess
python src/data_preprocessing.py

# train + log to mlflow
python src/train.py

# spin up everything (api, mlflow, prometheus, grafana)
docker compose up --build
```

Once it's up:
- API: `localhost:8000` (docs at `/docs`)
- MLflow: `localhost:5000`
- Prometheus: `localhost:9090`
- Grafana: `localhost:3000`

## Calling the API

The `/predict` endpoint expects a single `features` array with 41 numbers (already encoded/scaled), not named fields:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.0, 0.87, ...]}'
```

Check `/openapi.json` if you're not sure about the exact schema, or just POST an empty body and read the validation error - FastAPI tells you what it expects.

## Monitoring

Two layers here:

- **Prometheus + Grafana** - operational metrics (request counts, latency, how many predictions were "normal" vs "attack"). Dashboard is in `monitoring/grafana/`.
- **Evidently AI** - data drift between train and test sets. Run `monitoring/evidently_report.py` to regenerate the HTML report. Currently shows drift on 17/41 features, mostly the error-rate columns (`serror_rate`, `rerror_rate`, etc.) - makes sense given how NSL-KDD's splits were built.

## Known limitations

This is a prediction service, not a live network sensor - it doesn't capture packets off an interface. It reacts to whatever sends it a `features` request. Plugging in something like Zeek or Suricata to extract features from real traffic and call the API automatically would be the next step for an actual production setup, but that's outside the scope of this project.

## Tests

```bash
pytest tests/
```

## Tech stack

Python, scikit-learn, XGBoost, FastAPI, MLflow, Docker, GitHub Actions, Prometheus, Grafana, Evidently AI.

---
Project for Institut Supérieur de Management, d'Administration et de Génie Informatique - Cybersecurity track, 2025-2026.