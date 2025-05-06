from prophet import Prophet
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import talib as ta

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

# Prophet Forecast Function
# Fix in the prophet_forecast function
def prophet_forecast(df: pd.DataFrame, days: int = 30):
    df_prophet = df[['Close']].reset_index()
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'})
    if isinstance(df_prophet['y'], pd.DataFrame):
        df_prophet['y'] = df_prophet['y'].squeeze()  # Convert to Series if it's a DataFrame
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')

    # Check if 'y' is 1D and avoid incorrect format
    if df_prophet['y'].ndim > 1:
        df_prophet['y'] = df_prophet['y'].squeeze()

    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(df_prophet, periods=days)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

# XGBoost Forecast Function
def xgboost_forecast(df: pd.DataFrame, days: int = 30):
    df = df.copy()
    df["y_shifted"] = df["y"].shift(-1)
    df = add_technical_indicators(df)  # Include the technical indicators
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
        last_value = [next_pred]

    return preds