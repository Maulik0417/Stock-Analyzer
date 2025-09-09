from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/{ticker}")
def predict_stock(ticker: str):
    # Fetch 1-day intraday data (1m interval)
    data = yf.download(tickers=ticker, period="1d", interval="5m")
    data = data.reset_index()

    if data.empty:
        return {"error": "No data available"}

    # Prepare data for Prophet
    df = pd.DataFrame({
        "ds": data["Datetime"],
        "y": data["Close"]
    })

    # Prophet Model
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(df)
    future = prophet.make_future_dataframe(periods=12, freq="5min")  # 1 hour = 12 * 5min
    forecast = prophet.predict(future)

    # Prophet Prediction (last 1 hour)
    prophet_pred = forecast.tail(12)[["ds", "yhat"]].to_dict(orient="records")

    # Linear Regression as second model
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["y"].values
    model = LinearRegression()
    model.fit(X, y)
    future_idx = np.arange(len(df), len(df) + 12).reshape(-1, 1)
    lr_pred_values = model.predict(future_idx)

    lr_pred = [{"ds": df["ds"].iloc[-1] + timedelta(minutes=5 * (i + 1)),
                "yhat": val} for i, val in enumerate(lr_pred_values)]

    # Combine predictions (average Prophet + LR)
    combined_pred = []
    for i in range(12):
        avg_val = (prophet_pred[i]["yhat"] + lr_pred[i]["yhat"]) / 2
        combined_pred.append({
            "ds": prophet_pred[i]["ds"],
            "yhat": avg_val
        })

    # Format response
    response = {
        "historical": df.tail(78).to_dict(orient="records"),  # last ~6.5h trading data
        "prediction": combined_pred
    }
    return response