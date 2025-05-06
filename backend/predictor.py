from prophet import Prophet
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

def prophet_forecast(df: pd.DataFrame, days: int = 30):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

def xgboost_forecast(df: pd.DataFrame, days: int = 30):
    df = df.copy()
    df["y_shifted"] = df["y"].shift(-1)
    df.dropna(inplace=True)

    X = df[["y"]].values
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