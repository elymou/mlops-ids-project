from fastapi.testclient import TestClient
import sys
import os
import joblib
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ✅ Mock the model and scaler so tests work without real .pkl files
mock_model = MagicMock()
mock_model.predict.return_value = np.array([0])
mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
mock_model.__class__.__name__ = "RandomForestClassifier"

mock_scaler = MagicMock()
mock_scaler.transform.return_value = np.zeros((1, 41))

with patch('joblib.load', side_effect=[mock_model, mock_scaler]):
    from api.main import app

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