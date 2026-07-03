from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os

app = FastAPI(title="Telco Customer Churn Prediction Service (MLOps)")

# پیدا کردن آدرس آخرین مدل ذخیره شده در MLflow
# برای سادگی در داکر، مدل محلی ثبت شده در پوشه mlflow_runs یا آخرین مدل ترکینگ را لود می‌کنیم
MODEL_PATH = "mlruns/0/" # پیش‌فرض اولین اکسپریمنت

def get_latest_model_uri():
    # پیدا کردن آخرین run_id به صورت خودکار
    if os.path.exists("mlruns/0"):
        runs = [d for d in os.listdir("mlruns/0") if os.path.isdir(os.path.join("mlruns/0", d)) and d != "meta.yaml"]
        if runs:
            # مرتب‌سازی بر اساس زمان ایجاد پوشه برای یافتن آخرین اجرا
            runs.sort(key=lambda x: os.path.getmtime(os.path.join("mlruns/0", x)), reverse=True)
            return f"mlruns/0/{runs[0]}/artifacts/model"
    return None

class CustomerData(BaseModel):
    # یک نمونه ساده از ماتیریس ویژگی‌ها برای ورودی مدل
    features: list

@app.on_event("startup")
def load_model():
    global model
    model_uri = get_latest_model_uri()
    if model_uri and os.path.exists(model_uri):
        print(f"--- Loading model from: {model_uri} ---")
        model = mlflow.pyfunc.load_model(model_uri)
    else:
        print("--- Model URI not found, tracking fallback ---")
        model = None

@app.get("/")
def home():
    return {"message": "Welcome to Telco Churn MLOps Docker Service", "status": "Running"}

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        return {"error": "Model is not loaded or not found in MLflow Registry."}
    
    # انجام پیش‌بینی
    prediction = model.predict([data.features])
    return {
        "churn_prediction": int(prediction[0]),
        "result": "Customer will Churn" if prediction[0] == 1 else "Customer will Stay"
    }
