import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost import XGBoost
import boto3

def load_data(data_path):
    """Load and preprocess the data."""
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """Preprocess the data for model training."""
    # Separate features and target
    X = df.drop('purchase_conversion', axis=1)
    y = df['purchase_conversion']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train the XGBoost model."""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
    
    return y_pred, y_pred_proba

def save_model(model, scaler, model_dir):
    """Save the model and scaler."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model.save_model(os.path.join(model_dir, 'model.json'))
    
    # Save scaler
    import joblib
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))

def main():
    # Set up SageMaker session
    session = sagemaker.Session()
    
    # Load and preprocess data
    data_path = 'data/processed/training_data.csv'
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Save model and scaler
    model_dir = 'models'
    save_model(model, scaler, model_dir)
    
    # Prepare for SageMaker deployment
    xgb_model = XGBoost(
        entry_point='train.py',
        role=sagemaker.get_execution_role(),
        instance_count=1,
        instance_type='ml.m5.xlarge',
        framework_version='1.7-1',
        py_version='py3',
        output_path=f's3://{session.default_bucket()}/models'
    )
    
    # Deploy the model
    predictor = xgb_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge'
    )
    
    print("Model deployed successfully!")

if __name__ == "__main__":
    main() 