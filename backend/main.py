import yfinance as yf
import pandas as pd
from fastapi import FastAPI
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
import talib as ta
from typing import List
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Download stock data
def download_data(ticker="GOOG", start="2015-01-01", end="2025-01-01"):
    df = yf.download(ticker, start=start, end=end)
    return df

# Feature engineering: Add lag features
def create_lag_features(data, lag=5):
    for i in range(1, lag + 1):
        data[f"lag_{i}"] = data['Close'].shift(i)
    data = data.dropna()
    return data

# Add technical indicators using TA-Lib
# Fix in the add_technical_indicators function
def add_technical_indicators(df):
    # Ensure the Close column is a 1D numpy array
    close_prices = df['Close'].values  # Extract values as a 1D numpy array

    # Calculate the Simple Moving Average (SMA) over 5 days
    df['SMA'] = ta.SMA(close_prices, timeperiod=5)

    # Calculate the Relative Strength Index (RSI) over 14 days
    df['RSI'] = ta.RSI(close_prices, timeperiod=14)

    return df

# Train the XGBoost model
def train_xgboost_model(df):
    # Use lag features and technical indicators as features
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'SMA', 'RSI', 'MACD', 'MACD_signal']
    target = 'Close'

    # Create features and target
    X = df[features]
    y = df[target]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the XGBoost model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    return model

# Predict using Prophet
@app.get("/predict/prophet")
async def predict_prophet():
    df = download_data()
    df_prophet = df[['Close']].reset_index()
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'})
    if isinstance(df_prophet['y'], pd.DataFrame):
        df_prophet['y'] = df_prophet['y'].squeeze()  # Convert to Series if it's a DataFrame
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')

    # Train Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Make a future dataframe for the next 30 days
    future = model.make_future_dataframe(df_prophet, periods=30)
    forecast = model.predict(future)

    # Return the forecast results as a dictionary
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient='records')
    return {"predictions": forecast_data}

# Predict using XGBoost
@app.get("/predict/xgboost")
async def predict_xgboost():
    df = download_data()

    # Add lag features and technical indicators
    df = create_lag_features(df)
    df = add_technical_indicators(df)

    # Train the XGBoost model
    model = train_xgboost_model(df)

    # Prepare the features for prediction
    future_data = df.tail(1)[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'SMA', 'RSI', 'MACD', 'MACD_signal']]

    # Make predictions for the next 30 days
    predictions = []
    for _ in range(30):
        pred = model.predict(future_data)
        predictions.append(pred[0])
        # Update the future_data with the new predicted value
        future_data = future_data.copy()
        future_data['lag_1'] = future_data['lag_2']
        future_data['lag_2'] = future_data['lag_3']
        future_data['lag_3'] = future_data['lag_4']
        future_data['lag_4'] = future_data['lag_5']
        future_data['lag_5'] = pred

        # Recalculate technical indicators for the new data
        future_data = add_technical_indicators(future_data)

    # Return the predictions as a dictionary
    prediction_data = [{"day": i+1, "prediction": float(pred)} for i, pred in enumerate(predictions)]
    return {"predictions": prediction_data}