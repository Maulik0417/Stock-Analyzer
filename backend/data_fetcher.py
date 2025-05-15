import yfinance as yf
import pandas as pd

def get_stock_data():
    df = yf.download('GOOG', period='1y', auto_adjust=False)
    df = df[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    return df