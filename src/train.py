import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load data
X = np.load('data/processed/X_train.npy')
y = np.load('data/processed/y_train.npy')

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("IDS-Cybersecurity")

MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

best_model = None
best_f1 = 0
best_name = ""

for name, model in MODELS.items():
    with mlflow.start_run(run_name=name):
        print(f"🚀 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # ✅ Use correct logger per model type
        if name == "XGBoost":
            mlflow.xgboost.log_model(model, name)
        else:
            mlflow.sklearn.log_model(model, name)

        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\n✅ Best model: {best_name} (F1={best_f1:.4f}) saved!")