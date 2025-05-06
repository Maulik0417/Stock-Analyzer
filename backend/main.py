from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predictor import prophet_forecast, xgboost_forecast
from data_fetcher import get_stock_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/prophet")
def predict_prophet():
    df = get_stock_data()
    forecast = prophet_forecast(df)
    return forecast.to_dict(orient="records")

@app.get("/predict/xgboost")
def predict_xgboost():
    df = get_stock_data()
    preds = xgboost_forecast(df)
    return {"predictions": preds}