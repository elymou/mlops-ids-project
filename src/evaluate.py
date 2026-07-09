import numpy as np
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

COLUMNS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
'root_shell','su_attempted','num_root','num_file_creations','num_shells',
'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
'label','difficulty']


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
    print("MODEL EVALUATION REPORT (train-derived holdout)")
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


def evaluate_on_real_test_set(
    model_path='models/best_model.pkl',
    scaler_path='models/scaler.pkl',
    encoders_path='models/encoders.pkl',
    test_path='/home/drp53/Desktop/mlops-ids-project/data/raw/KDDTest+.txt'
):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)

    df = pd.read_csv(test_path, names=COLUMNS)
    y_true = df['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)

    X = df.drop(columns=['label', 'difficulty']).copy()
    for col, le in encoders.items():
        known = set(le.classes_)
        X[col] = X[col].apply(lambda x: x if x in known else le.classes_[0])
        X[col] = le.transform(X[col])

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 50)
    print("MODEL EVALUATION ON REAL TEST SET (KDDTest_.txt)")
    print("=" * 50)
    print(f"Total test samples : {len(y_true)}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print("                Predicted Normal   Predicted Attack")
    print(f"Actual Normal        {cm[0][0]:6d}             {cm[0][1]:6d}")
    print(f"Actual Attack        {cm[1][0]:6d}             {cm[1][1]:6d}")
    print()
    print("Full classification report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))

    return {"accuracy": accuracy, "f1_score": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    evaluate_model()               # existing: train-derived holdout
    print("\n\n")
    evaluate_on_real_test_set()    # new: real, independent NSL-KDD test set