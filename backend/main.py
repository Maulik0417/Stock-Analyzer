import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
import talib as ta
import numpy as np
import traceback
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay


app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_data(ticker="MSFT", start="2015-01-01", end=None):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')  # today, dynamic
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError("No data downloaded for the ticker.")

    # Flatten multiindex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Find the close column by searching possible names
    possible_close_cols = [col for col in df.columns if 'close' in col.lower()]
    if not possible_close_cols:
        raise ValueError("No close price column found in the data")
    # Rename the first found close column to 'Close'
    df = df.rename(columns={possible_close_cols[0]: 'Close'})

    return df


def create_lag_features(data, lag=5):
    for i in range(1, lag + 1):
        data[f"lag_{i}"] = data['Close'].shift(i)
    data = data.dropna()
    return data

def add_technical_indicators(df):
    if 'Close' not in df.columns or df['Close'].empty:
        raise ValueError("Missing or empty 'Close' column for technical indicators")

    close_prices = np.asarray(df['Close']).ravel()

    df['SMA'] = ta.SMA(close_prices, timeperiod=5)
    df['RSI'] = ta.RSI(close_prices, timeperiod=14)
    macd, macd_signal, _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal

    return df

def train_xgboost_model(df):
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    target = 'Close'

    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    return model

from datetime import timedelta

@app.get("/predict/prophet")
async def predict_prophet():
    try:
        df = download_data()

        # Prepare df for Prophet
        df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet = df_prophet.dropna(subset=['y'])

        model = Prophet()
        model.fit(df_prophet)

        # Get last date from data
        last_date = df_prophet['ds'].max()

        # Select past 10 days (including last_date)
        past_start_date = last_date - BDay(9)
        past_data = df_prophet[df_prophet['ds'] >= past_start_date].copy()
        past_data['type'] = 'actual'  # mark as actual

        # Predict next 20 days
        future = model.make_future_dataframe(periods=20, freq='B')  # 'B' = business day frequency
        forecast = model.predict(future)
        future_forecast = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_forecast['type'] = 'forecast'  # mark as forecast

        # Combine past actual and future forecast
        combined = pd.concat([past_data[['ds', 'y', 'type']], 
                              future_forecast.rename(columns={'yhat': 'y'})[['ds', 'y', 'type']]], ignore_index=True)

        # Convert to records and return
        results = combined.to_dict(orient='records')
        return {"predictions": results}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/xgboost")
async def predict_xgboost():
    try:
        df = download_data()

        df = create_lag_features(df)
        df = add_technical_indicators(df)
        df = df.dropna(subset=['SMA', 'RSI', 'MACD', 'MACD_signal'])

        model = train_xgboost_model(df)

        # Get last 10 business days of actual data
        df = df.reset_index()
        df['ds'] = pd.to_datetime(df['Date'])
        last_actuals = df[['ds', 'Close']].dropna().sort_values('ds').tail(10)
        last_actuals = last_actuals.rename(columns={'Close': 'prediction'})
        last_actuals['type'] = 'actual'

        # Start from the latest known row to predict future
        last_row = df.iloc[-1]
        last_date = last_row['ds']
        future_data = last_row[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'SMA', 'RSI', 'MACD', 'MACD_signal']].to_frame().T

        # Predict next 20 business days
        future_predictions = []
        for i in range(20):
            feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
            future_data = future_data[feature_cols].astype(float)
            pred = model.predict(future_data)[0]
            pred_date = last_date + BDay(i + 1)  # +1 to skip current day
            future_predictions.append({
                "ds": pred_date.strftime('%Y-%m-%d'),
                "prediction": float(pred),
                "type": "forecast"
            })

            # Shift lags
            future_data = future_data.copy()
            future_data['lag_1'] = future_data['lag_2']
            future_data['lag_2'] = future_data['lag_3']
            future_data['lag_3'] = future_data['lag_4']
            future_data['lag_4'] = future_data['lag_5']
            future_data['lag_5'] = pred  # use new prediction as latest lag

        # Combine and return
        combined = pd.concat([last_actuals, pd.DataFrame(future_predictions)], ignore_index=True)
        return {"predictions": combined.to_dict(orient="records")}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))