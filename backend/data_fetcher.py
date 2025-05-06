import yfinance as yf
import pandas as pd

def get_stock_data():
    df = yf.download("GOOG", start="2015-01-01", progress=False)
    df = df[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    return df