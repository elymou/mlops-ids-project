from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

app = FastAPI(title="IDS MLOps API", version="1.0")

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ✅ Custom ML metrics
ATTACK_COUNTER = Counter('ids_attacks_detected_total', 'Total attacks detected by IDS')
NORMAL_COUNTER = Counter('ids_normal_traffic_total', 'Total normal traffic detected by IDS')

Instrumentator().instrument(app).expose(app)

class NetworkTraffic(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "IDS MLOps API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(traffic: NetworkTraffic):
    data = np.array(traffic.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0].tolist()

    # ✅ Increment ML-specific counters
    if prediction == 1:
        ATTACK_COUNTER.inc()
    else:
        NORMAL_COUNTER.inc()

    return {
        "prediction": int(prediction),
        "label": "ATTACK 🚨" if prediction == 1 else "NORMAL ✅",
        "probability": {
            "normal": round(probability[0], 4),
            "attack": round(probability[1], 4)
        }
    }

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "features_expected": 41
    }