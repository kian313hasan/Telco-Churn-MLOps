import mlflow
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def init_mlflow(experiment_name="Telco_Churn_Project"):
    """راه‌اندازی آزمایش در MLflow"""
    mlflow.set_experiment(experiment_name)

def log_experiment(model_name, data_version, params, metrics, y_true, y_pred, model=None):
    """ثبت کامل اطلاعات یک اجرا در MLflow"""
    with mlflow.start_run(run_name=f"{model_name}_{data_version}"):
        # ۱. ثبت پارامترها و اطلاعات نسخه داده
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_version", data_version)
        for p_name, p_val in params.items():
            mlflow.log_param(p_name, p_val)
            
        # ۲. ثبت معیارهای ارزیابی
        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, m_val)
            
        # ۳. ساخت و ثبت ماتریس آشفتگی (Confusion Matrix) به عنوان Artifact
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        
        cm_path = f"confusion_matrix_{model_name}_{data_version}.png"
        plt.savefig(cm_path)
        plt.close()
        
        mlflow.log_artifact(cm_path)
        
        # ۴. ثبت خود مدل در صورت وجود (اختیاری برای مراحل بعد)
        if model is not None:
            if "Logistic" in model_name:
                mlflow.sklearn.log_model(model, "model")
            elif "Forest" in model_name:
                mlflow.sklearn.log_model(model, "model")
            elif "XGBoost" in model_name:
                mlflow.xgboost.log_model(model, "model")
            elif "CatBoost" in model_name:
                mlflow.catboost.log_model(model, "model")
                
        print(f"اطلاعات مدل {model_name} برای نسخه {data_version} با موفقیت در MLflow ثبت شد.")
