from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="IDS MLOps API", version="1.0")

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Prometheus monitoring
Instrumentator().instrument(app).expose(app)

class NetworkTraffic(BaseModel):
    features: list[float]  # 41 features from NSL-KDD

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