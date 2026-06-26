import numpy as np
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import pandas as pd

def evaluate_model(model_path='models/best_model.pkl'):
    # Load model
    model = joblib.load(model_path)

    # Load data
    X = np.load('data/processed/X_train.npy')
    y = np.load('data/processed/y_train.npy')

    # Same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preds = model.predict(X_test)

    # Metrics
    acc       = accuracy_score(y_test, preds)
    f1        = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall    = recall_score(y_test, preds)
    cm        = confusion_matrix(y_test, preds)

    print("=" * 40)
    print("📊 MODEL EVALUATION REPORT")
    print("=" * 40)
    print(f"  Model      : {type(model).__name__}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Normal', 'Attack']))

    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

if __name__ == "__main__":
    evaluate_model()