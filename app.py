from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os

app = FastAPI(title="Telco Customer Churn Prediction Service (MLOps)")

# مسیر ثابت و استاندارد لود مدل نهایی ثبت شده در پایپ‌لاین
# این ساختار پایداری کامل سرویس را در کانتینر داکر تضمین می‌کند
MODEL_PATH = "best_model"

def get_best_model_uri():
    """
    یافتن هوشمند بهترین مدل بر اساس بالاترین مقدار متریک ثبت شده در MLflow
    """
    # در صورت وجود مسیر ثابت و پورت شده در داکر، از آن استفاده می‌شود
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
        
    try:
        # جستجوی هوشمند در ران‌های ثبت شده برای یافتن بهترین مدل بر اساس F1-score
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Telco_Churn_Project")
        
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.F1-score DESC"],
                max_results=1
            )
            if runs:
                best_run_id = runs[0].info.run_id
                return f"runs:/{best_run_id}/model"
    except Exception:
        pass
        
    # روش جایگزین سنتی در صورت عدم دسترسی به کلاینت ترکینگ
    if os.path.exists("mlruns/0"):
        runs = [d for d in os.listdir("mlruns/0") if os.path.isdir(os.path.join("mlruns/0", d)) and d != "meta.yaml"]
        if runs:
            runs.sort(key=lambda x: os.path.getmtime(os.path.join("mlruns/0", x)), reverse=True)
            return f"mlruns/0/{runs[0]}/artifacts/model"
            
    return None

class CustomerData(BaseModel):
    # ساختار استاندارد جی‌سان ورودی برای ماتریس ویژگی‌ها
    features: list

@app.on_event("startup")
def load_model():
    global model
    model_uri = get_best_model_uri()
    
    if model_uri:
        print(f"--- Loading best model from verified URI: {model_uri} ---")
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print("مدل نهایی با موفقیت لود شد و آماده پاسخگویی است.")
        except Exception as e:
            print(f"--- Error loading model: {e} ---")
            model = None
    else:
        print("--- Best model URI not found, tracking fallback ---")
        model = None

@app.get("/")
def home():
    return {"message": "Welcome to Telco Churn MLOps Docker Service", "status": "Running"}

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        return {"error": "Model is not loaded properly or not found in MLflow Registry."}
    
    # انجام پیش‌بینی آنی بر روی ویژگی‌های ارسالی کاربر
    try:
        prediction = model.predict([data.features])
        return {
            "churn_prediction": int(prediction[0]),
            "result": "Customer will Churn" if prediction[0] == 1 else "Customer will Stay"
        }
    except Exception as e:
        return {"error": f"Error during inference process: {str(e)}"}