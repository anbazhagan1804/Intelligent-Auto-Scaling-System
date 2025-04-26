import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from prometheus_api_client import PrometheusConnect
from flask import Flask, jsonify

app = Flask(__name__)

# Connect to Prometheus
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server.monitoring.svc.cluster.local:9090")
prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/traffic_predictor.joblib")

def get_historical_data(query, days=7):
    """Fetch historical metrics data from Prometheus"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Get metric data
    metric_data = prom.custom_query_range(
        query=query,
        start_time=start_time,
        end_time=end_time,
        step="1h"
    )
    
    if not metric_data:
        return None
    
    # Convert to DataFrame
    data_points = []
    for result in metric_data:
        for value in result["values"]:
            timestamp = datetime.fromtimestamp(value[0])
            data_points.append({
                "timestamp": timestamp,
                "value": float(value[1])
            })
    
    return pd.DataFrame(data_points)

def extract_features(df):
    """Extract time-based features for ML model"""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add lag features
    df['value_lag_1h'] = df['value'].shift(1)
    df['value_lag_24h'] = df['value'].shift(24)
    df['value_lag_7d'] = df['value'].shift(24*7)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def train_model(df):
    """Train a RandomForest model for prediction"""
    features = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 
                'value_lag_1h', 'value_lag_24h', 'value_lag_7d']
    
    X = df[features]
    y = df['value']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    return model

def predict_future(model, df, hours_ahead=24):
    """Predict future values based on historical data"""
    last_row = df.iloc[-1:].copy()
    predictions = []
    
    for i in range(hours_ahead):
        # Create a new row for prediction
        new_row = last_row.copy()
        
        # Update timestamp
        new_timestamp = last_row['timestamp'].iloc[0] + timedelta(hours=i+1)
        new_row['timestamp'] = new_timestamp
        
        # Update time features
        new_row['hour'] = new_timestamp.hour
        new_row['day_of_week'] = new_timestamp.dayofweek
        new_row['day_of_month'] = new_timestamp.day
        new_row['month'] = new_timestamp.month
        new_row['is_weekend'] = 1 if new_timestamp.dayofweek >= 5 else 0
        
        # Make prediction
        features = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 
                    'value_lag_1h', 'value_lag_24h', 'value_lag_7d']
        prediction = model.predict(new_row[features])[0]
        
        # Store prediction
        predictions.append({
            'timestamp': new_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_value': prediction
        })
        
        # Update lag values for next prediction
        new_row['value'] = prediction
        new_row['value_lag_1h'] = prediction
        
        if i >= 23:
            new_row['value_lag_24h'] = predictions[i-23]['predicted_value']
        
        if i >= 167:  # 24*7 - 1
            new_row['value_lag_7d'] = predictions[i-167]['predicted_value']
            
        last_row = new_row
    
    return predictions

@app.route('/predict/<service_name>', methods=['GET'])
def get_prediction(service_name):
    """API endpoint to get predictions for a service"""
    try:
        # Query for specific service
        query = f'sum(rate(http_requests_total{{service="{service_name}"}}[5m]))'
        
        # Get historical data
        df = get_historical_data(query)
        if df is None or df.empty:
            return jsonify({"error": "No data available for this service"}), 404
        
        # Process data
        processed_df = extract_features(df)
        
        # Load or train model
        try:
            model = joblib.load(MODEL_PATH)
        except:
            model = train_model(processed_df)
        
        # Make predictions
        predictions = predict_future(model, processed_df)
        
        return jsonify({
            "service": service_name,
            "predictions": predictions
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)