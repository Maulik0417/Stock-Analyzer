from prophet import Prophet
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import talib as ta

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
    X = df[features].values
    y = df["y_shifted"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    preds = []
    last_value = X[-1]

    for _ in range(days):
        next_pred = model.predict([last_value])[0]
        preds.append(next_pred)
        # Update last_value input: shift predicted y, keep other indicators the same
        last_value = [next_pred] + list(last_value[1:])

    return preds