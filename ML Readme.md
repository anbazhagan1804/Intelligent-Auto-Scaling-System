# Machine Learning for Predictive Scaling

This directory contains Python code and ML models for predictive scaling in the Intelligent Auto-Scaling System. The ML service forecasts traffic and resource needs to enable proactive scaling decisions.

## Components

- `predictor.py`: Core ML service that fetches metrics from Prometheus, trains models, and makes predictions
- `models/`: Directory containing trained ML models
- `data_processor.py`: Utilities for processing and transforming time-series data
- `training.py`: Scripts for model training and evaluation
- `deployment.yaml`: Kubernetes deployment configuration for the ML service

## How It Works

1. Historical metrics are collected from Prometheus
2. Time-series data is processed and used to train prediction models
3. Models forecast future resource needs based on patterns and seasonality
4. Predictions are exposed via a REST API for the scaling controller
5. The system continuously retrains models as new data becomes available