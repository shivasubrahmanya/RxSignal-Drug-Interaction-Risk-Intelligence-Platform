import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

def calculate_classification_metrics(y_true, y_pred, threshold=5.0):
    # Convert continuous scores to binary classification (1: Significant Risk, 0: Low Risk)
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    return acc, prec, rec, f1

def train_xgboost(data_file, model_file):
    print(f"Loading ML data from {data_file}...")
    df = pd.read_parquet(data_file)
    
    features = [
        'drug_a_freq', 'drug_b_freq', 'drug_a_mean_risk', 'drug_b_mean_risk',
        'combined_risk_interaction', 'risk_difference', 'freq_ratio'
    ]
    target = 'max_risk'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training DEEP XGBoost Regressor on {len(X_train)} samples...")
    # Optimized hyperparameters for higher accuracy
    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0, # L2 Regularization to prevent overfitting
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    print("\nEvaluating DEEP XGBoost Model...")
    y_test_pred = model.predict(X_test)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    acc, prec, rec, f1 = calculate_classification_metrics(y_test, y_test_pred, threshold=5.0)
    
    print(f"Regression Metrics -> MSE: {test_mse:.4f} | R2 Score: {test_r2:.4f}")
    print(f"Classification Metrics (Risk > 5.0) -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    # Save the metrics to a file so we can view them later
    metrics = f"XGBoost R2: {test_r2:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}"
    with open(os.path.join(os.path.dirname(model_file), "..", "xgb_metrics.txt"), "w") as f:
        f.write(metrics)
        
    print(f"Saving model to {model_file}...")
    joblib.dump(model, model_file)
    print("Done!")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "processed", "ml_training_data.parquet")
    model_file = os.path.join(base_dir, "models", "xgboost_risk_model.pkl")
    
    train_xgboost(data_file, model_file)
