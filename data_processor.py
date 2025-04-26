import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def clean_time_series(df, value_column='value'):
    """Clean time series data by removing outliers and filling missing values"""
    # Remove extreme outliers (3 standard deviations)
    mean = df[value_column].mean()
    std = df[value_column].std()
    df = df[(df[value_column] > mean - 3*std) & (df[value_column] < mean + 3*std)]
    
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp')
    
    # Resample to hourly data and interpolate missing values
    df.set_index('timestamp', inplace=True)
    df = df.resample('1H').mean()
    df = df.interpolate(method='time')
    df.reset_index(inplace=True)
    
    return df

def add_seasonal_features(df):
    """Add seasonal features to the dataframe"""
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    return df

def create_lag_features(df, value_column='value', lag_hours=[1, 2, 3, 24, 48, 168]):
    """Create lag features for time series prediction"""
    df = df.copy()
    
    for lag in lag_hours:
        df[f'{value_column}_lag_{lag}h'] = df[value_column].shift(lag)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def create_rolling_features(df, value_column='value', windows=[3, 6, 12, 24]):
    """Create rolling window features (mean, std, min, max)"""
    df = df.copy()
    
    for window in windows:
        df[f'{value_column}_rolling_mean_{window}h'] = df[value_column].rolling(window=window).mean()
        df[f'{value_column}_rolling_std_{window}h'] = df[value_column].rolling(window=window).std()
        df[f'{value_column}_rolling_min_{window}h'] = df[value_column].rolling(window=window).min()
        df[f'{value_column}_rolling_max_{window}h'] = df[value_column].rolling(window=window).max()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def prepare_train_test_data(df, value_column='value', test_size=0.2):
    """Split data into training and testing sets"""
    # Determine split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Define features and target
    feature_columns = [col for col in df.columns if col not in ['timestamp', value_column]]
    
    X_train = train_df[feature_columns]
    y_train = train_df[value_column]
    
    X_test = test_df[feature_columns]
    y_test = test_df[value_column]
    
    return X_train, y_train, X_test, y_test, feature_columns