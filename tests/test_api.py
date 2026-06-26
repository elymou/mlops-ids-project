from fastapi.testclient import TestClient
import sys
import os

# ✅ Tell Python where to find the api folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.main import app  # ✅ correct path

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    features = [0.0] * 41
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "label" in response.json()

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["features_expected"] == 41