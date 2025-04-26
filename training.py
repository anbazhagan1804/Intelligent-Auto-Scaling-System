import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processor import (
    clean_time_series, 
    add_seasonal_features, 
    create_lag_features, 
    create_rolling_features,
    prepare_train_test_data
)

def train_and_evaluate(data_path, model_type='random_forest', model_path=None):
    """Train and evaluate a model on the provided data"""
    # Load data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Process data
    df = clean_time_series(df)
    df = add_seasonal_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    # Prepare train/test data
    X_train, y_train, X_test, y_test, feature_columns = prepare_train_test_data(df)
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model: {model_type}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Importance:")
        for i in range(min(10, len(feature_columns))):
            idx = indices[i]
            print(f"{feature_columns[idx]}: {importances[idx]:.4f}")
    
    # Save model if path provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model, rmse, mae, r2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a predictive scaling model')
    parser.add_argument('--data', required=True, help='Path to the training data CSV')
    parser.add_argument('--model-type', default='random_forest', 
                        choices=['random_forest', 'gradient_boosting', 'linear'],
                        help='Type of model to train')
    parser.add_argument('--output', default='models/traffic_predictor.joblib',
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    
    train_and_evaluate(args.data, args.model_type, args.output)