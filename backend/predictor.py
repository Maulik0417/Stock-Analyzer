from prophet import Prophet
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import talib as ta
import numpy as np

def add_technical_indicators(df):
    if 'Close' not in df.columns or df['Close'].empty:
        raise ValueError("Missing or empty 'Close' column for technical indicators")

    close_prices = np.asarray(df['Close']).ravel()  # Ensure 1D numpy array

    df['SMA'] = ta.SMA(close_prices, timeperiod=5)
    df['RSI'] = ta.RSI(close_prices, timeperiod=14)

    macd, macd_signal, _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal

    return df

def prophet_forecast(df: pd.DataFrame, days: int = 30):
    df_prophet = df.copy()

    if not isinstance(df_prophet.index, pd.DatetimeIndex):
        df_prophet.index = pd.to_datetime(df_prophet.index)

    df_prophet = df_prophet[['Close']].rename(columns={'Close': 'y'})
    df_prophet['ds'] = df_prophet.index
    df_prophet = df_prophet[['ds', 'y']]

    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')

    if df_prophet['y'].isnull().all():
        raise ValueError("Prophet forecast: all target values are NaN.")

    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

def xgboost_forecast(df: pd.DataFrame, days: int = 30):
    df = df.copy()
    df["y"] = df["Close"]
    df["y_shifted"] = df["y"].shift(-1)

    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    features = ['y', 'SMA', 'RSI', 'MACD', 'MACD_signal']
    X = df[features]
    y = df["y_shifted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # Start from the last known price
    full_close_series = df['y'].tolist()
    preds = []

    for _ in range(days):
        # Create a DataFrame with the latest prices
        temp_df = pd.DataFrame({'Close': full_close_series})
        temp_df = add_technical_indicators(temp_df)
        temp_df.dropna(inplace=True)

        last_features = temp_df.iloc[-1][['Close', 'SMA', 'RSI', 'MACD', 'MACD_signal']].values
        next_pred = model.predict([last_features])[0]
        preds.append(next_pred)

        # Append the prediction to the price series
        full_close_series.append(next_pred)

    return preds