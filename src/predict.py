import numpy as np
import joblib
import pandas as pd

def load_artifacts():
    model    = joblib.load('models/best_model.pkl')
    scaler   = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/encoders.pkl')
    return model, scaler, encoders

def predict_single(raw_input: dict):
    """
    raw_input example:
    {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'http',
        'flag': 'SF', 'src_bytes': 181, 'dst_bytes': 5450, ...
    }
    """
    model, scaler, encoders = load_artifacts()

    df = pd.DataFrame([raw_input])

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    X = scaler.transform(df.values)
    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    label = "ATTACK 🚨" if prediction == 1 else "NORMAL ✅"
    print(f"\n🔍 Prediction : {label}")
    print(f"   Confidence : Normal={probability[0]:.2%} | Attack={probability[1]:.2%}")

    return {
        "prediction": int(prediction),
        "label": label,
        "probability_normal": round(float(probability[0]), 4),
        "probability_attack": round(float(probability[1]), 4)
    }

def predict_from_features(features: list):
    """
    Use this when you have 41 pre-encoded numeric features directly.
    This is what the API uses.
    """
    model, scaler, _ = load_artifacts()

    data        = np.array(features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction  = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0]

    label = "ATTACK 🚨" if prediction == 1 else "NORMAL ✅"
    print(f"\n🔍 Prediction : {label}")
    print(f"   Confidence : Normal={probability[0]:.2%} | Attack={probability[1]:.2%}")

    return {
        "prediction": int(prediction),
        "label": label,
        "probability_normal": round(float(probability[0]), 4),
        "probability_attack": round(float(probability[1]), 4)
    }

if __name__ == "__main__":
    # Quick test with 41 zeros (dummy input)
    print("Testing with dummy input...")
    result = predict_from_features([0.0] * 41)
    print(result)