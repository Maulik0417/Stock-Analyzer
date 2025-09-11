# backend/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pytz
import os

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Prophet import (try both names)
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

app = FastAPI(title="Stock Predictor API")

EAST = pytz.timezone("America/New_York")

# Helper functions
def get_intraday_data(symbol: str, interval="1m"):
    """
    Fetch intraday data for the current trading day.
    Returns a DataFrame indexed by timezone-aware timestamps in America/New_York.
    """
    # yfinance might restrict 1m in some cases; fall back to 5m if necessary.
    try:
        df = yf.Ticker(symbol).history(period="1d", interval=interval, actions=False)
    except Exception:
        df = pd.DataFrame()

    if df.empty and interval != "5m":
        df = yf.Ticker(symbol).history(period="2d", interval="5m", actions=False)

    if df.empty:
        raise RuntimeError("No intraday data returned by yfinance for symbol " + symbol)

    # ensure index tz-aware and in Eastern
    if df.index.tz is None:
        # yfinance usually returns UTC-aware; but handle naive -> assume UTC then convert
        df.index = df.index.tz_localize("UTC").tz_convert(EAST)
    else:
        df.index = df.index.tz_convert(EAST)

    # keep only today's data (market local date)
    now_east = datetime.datetime.now(EAST)
    today = now_east.date()
    df = df[df.index.date == today]
    return df

def market_open_close_for_date(date: datetime.date):
    # NYSE open 9:30, close 16:00 Eastern
    open_dt = EAST.localize(datetime.datetime.combine(date, datetime.time(9,30)))
    close_dt = EAST.localize(datetime.datetime.combine(date, datetime.time(16,0)))
    return open_dt, close_dt

def create_minute_index(open_dt, close_dt):
    # minute-resolution timestamps inclusive of open to close
    idx = pd.date_range(start=open_dt, end=close_dt, freq="1T", tz=EAST)
    return idx

def prepare_features_for_ml(df: pd.DataFrame, lookback=60):
    # df expected to have 'Close'
    # Build simple lag features + moving averages
    prices = df['Close'].fillna(method="ffill").values
    X = []
    for i in range(lookback, len(prices)):
        window = prices[i-lookback:i]
        features = []
        features.append(window[-1])  # last price
        features.append(window.mean())
        features.append(window.std())
        # last 3 lags
        features.extend(list(window[-3:]))
        X.append(features)
    X = np.array(X)
    return X

def train_random_forest(prices):
    # Create a dataset where we try to predict price one hour ahead using last 60 minutes
    lookback = 60  # last 60 minutes features
    if len(prices) < lookback + 61:
        # Not enough data; reduce lookback
        lookback = max(5, len(prices)//2)

    # Build X, y where y is price at t + 60 minutes (1 hour ahead)
    X = []
    y = []
    for i in range(0, len(prices) - lookback - 60):
        window = prices[i:i+lookback]
        target = prices[i+lookback+60-1]  # index at +60 minutes ahead
        feats = [window[-1], np.mean(window), np.std(window)]
        feats += list(window[-3:])
        X.append(feats)
        y.append(target)

    if len(X) < 10:
        # Not enough data to train RF meaningfully; fallback to naive predictor (last price)
        return None, None, None

    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(Xs, y)
    return rf, scaler, lookback

def predict_with_rf(rf, scaler, lookback, recent_prices):
    if rf is None:
        return None
    window = recent_prices[-lookback:]
    feat = [window[-1], np.mean(window), np.std(window)]
    # pad if needed
    last_three = list(window[-3:]) if len(window) >=3 else [window[-1]]*3
    feat += last_three
    X = scaler.transform([feat])
    return float(rf.predict(X)[0])

def train_lstm(prices):
    # Very small LSTM that predicts 60-min ahead via sequence-to-scalar
    seq_len = min(60, max(5, len(prices)//4))
    if len(prices) < seq_len + 61:
        seq_len = max(5, len(prices)//4)
    if seq_len < 5:
        return None, None, None

    X = []
    y = []
    for i in range(0, len(prices) - seq_len - 60):
        X.append(prices[i:i+seq_len])
        y.append(prices[i+seq_len+60-1])
    X = np.array(X)
    y = np.array(y)
    # normalize
    mean = X.mean()
    std = X.std() if X.std() != 0 else 1.0
    Xs = (X - mean) / std
    ys = (y - mean) / std

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len,1)),
        tf.keras.layers.LSTM(32, activation='tanh'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    # reshape
    Xs = Xs.reshape(Xs.shape[0], Xs.shape[1], 1)
    # quick train with small epochs for speed
    model.fit(Xs, ys, epochs=10, batch_size=16, verbose=0)
    return model, mean, std, seq_len

def predict_with_lstm(model, mean, std, seq_len, recent_prices):
    if model is None:
        return None
    window = recent_prices[-seq_len:]
    x = (np.array(window) - mean) / std
    x = x.reshape(1, seq_len, 1)
    pred_scaled = model.predict(x, verbose=0)[0,0]
    pred = pred_scaled * std + mean
    return float(pred)

def predict_with_prophet(df):
    if Prophet is None:
        return None
    # Prophet expects ds (datetime) and y (value)
    # We'll use minute-level data and build a dataset with Close values
    ds = df.index.tz_convert("UTC")  # prophet prefers naive or UTC
    tmp = pd.DataFrame({"ds": ds, "y": df['Close'].values})
    # Resample to 1-minute as needed
    tmp = tmp.reset_index(drop=True)
    # Train quick prophet
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    try:
        m.fit(tmp)
        future = m.make_future_dataframe(periods=60, freq='min')  # 60 minutes ahead
        forecast = m.predict(future)
        # the last forecasted point is price 60 minutes ahead
        pred = forecast.iloc[-1]['yhat']
        return float(pred)
    except Exception:
        return None

class PredictResponse(BaseModel):
    symbol: str
    market_open: datetime.datetime
    market_close: datetime.datetime
    timestamps: List[str]   # ISO strings for every minute from open to close
    prices: List[Optional[float]]  # actual prices or null
    predicted: Dict[str, Any]     # includes model predictions and ensemble

@app.get("/predict", response_model=PredictResponse)
def predict(symbol: str = Query("GOOG")):
    # 1) fetch intraday
    symbol = symbol.upper()
    try:
        df = get_intraday_data(symbol, interval="1m")
    except Exception as e:
        # fallback to 5m
        df = get_intraday_data(symbol, interval="5m")

    now_east = datetime.datetime.now(EAST)
    today = now_east.date()
    open_dt, close_dt = market_open_close_for_date(today)
    minute_index = create_minute_index(open_dt, close_dt)

    # build arrays of prices aligned to minute_index
    price_map = {ts: None for ts in minute_index}
    # yfinance data index may be irregular (every 1m when market open)
    # We'll map available closes to the nearest minute in our index
    for ts, row in df.iterrows():
        # round ts to minute (it should already be minute-aligned)
        ts_r = ts.floor('T')
        if ts_r in price_map:
            price_map[ts_r] = float(row['Close'])

    # Build price list
    prices_list = [price_map[ts] for ts in minute_index]

    # Prepare recent_prices array (use last available closes, forward-fill)
    prices_series = pd.Series(prices_list, index=minute_index)
    prices_series = prices_series.fillna(method="ffill")
    # If still NaN at start (market not open yet), leave as NaN
    recent_prices = prices_series.dropna().values
    if len(recent_prices) == 0:
        raise RuntimeError("No price data available yet for today.")

    # Train models
    # Use the non-NaN portion of the series
    prices_for_ml = recent_prices

    # Random Forest
    rf, rf_scaler, rf_lookback = train_random_forest(prices_for_ml)

    rf_pred = predict_with_rf(rf, rf_scaler, rf_lookback, prices_for_ml) if rf is not None else None

    # LSTM
    try:
        lstm_model, lstm_mean, lstm_std, lstm_seq_len = train_lstm(prices_for_ml)
        lstm_pred = predict_with_lstm(lstm_model, lstm_mean, lstm_std, lstm_seq_len, prices_for_ml) if lstm_model is not None else None
    except Exception:
        lstm_pred = None

    # Prophet
    try:
        prophet_pred = predict_with_prophet(df)
    except Exception:
        prophet_pred = None

    # Naive fallback: last price as prediction
    naive_pred = float(recent_prices[-1])

    # Build ensemble: average of available model predictions (exclude None)
    preds = [p for p in [prophet_pred, rf_pred, lstm_pred] if p is not None]
    if len(preds) == 0:
        ensemble = naive_pred
    else:
        ensemble = float(np.mean(preds + [naive_pred]))  # include naive to stabilize

    # predicted timestamp is now + 1 hour, but clamp to market close if beyond close
    pred_dt = now_east + datetime.timedelta(hours=1)
    if pred_dt > close_dt:
        pred_dt = close_dt

    # convert minute_index to ISO strings
    timestamps_iso = [ts.isoformat() for ts in minute_index]
    prices_out = [None if np.isnan(p) else float(p) for p in prices_series.values]

    result = {
        "symbol": symbol,
        "market_open": open_dt.isoformat(),
        "market_close": close_dt.isoformat(),
        "timestamps": timestamps_iso,
        "prices": prices_out,
        "predicted": {
            "timestamp": pred_dt.isoformat(),
            "ensemble": float(ensemble),
            "models": {
                "prophet": float(prophet_pred) if prophet_pred is not None else None,
                "random_forest": float(rf_pred) if rf_pred is not None else None,
                "lstm": float(lstm_pred) if lstm_pred is not None else None,
                "naive_last_price": float(naive_pred)
            }
        }
    }
    return result