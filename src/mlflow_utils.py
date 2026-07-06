import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os

def init_mlflow(experiment_name="Telco_Churn_Project"):
    """راه‌اندازی آزمایش در MLflow با ساختار استاندارد"""
    mlflow.set_experiment(experiment_name)

def log_experiment(model_name, data_version, params, metrics, plot_path, model=None):
    """
    ثبت کامل و بدون تکرار اطلاعات آزمایش در MLflow با تفکیک هوشمند انواع مدل‌ها برای امنیت سایبری
    """
    with mlflow.start_run(run_name=f"{model_name}_{data_version}"):
        # ۱. ثبت پارامترها و اطلاعات نسخه داده
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_version", data_version)
        for p_name, p_val in params.items():
            mlflow.log_param(p_name, p_val)
            
        # ۲. ثبت معیارهای ارزیابی پنج‌گانه خواسته شده
        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, m_val)
            
        # ۳. ثبت مستقیم ماتریس آشفتگی دریافتی از ماژول ارزیابی به عنوان Artifact
        if plot_path and os.path.exists(plot_path):
            mlflow.log_artifact(plot_path)
            try:
                os.remove(plot_path)  # پاک‌سازی فایل موقت محلی
            except OSError:
                pass
                
        # ۴. ثبت هوشمند مدل بر اساس فلیور واقعی آن و رفع خطای امنیتی skops برای CatBoost و XGBoost
        if model is not None:
            if "XGBoost" in model_name:
                mlflow.xgboost.log_model(model, "model")
            elif "CatBoost" in model_name:
                # کتبست به دلیل سازگاری با API اسکایلرن با تراست کردن تایپ‌های آن ذخیره می‌شود
                mlflow.sklearn.log_model(model, "model", skops_trusted_types=["catboost.core.CatBoostClassifier"])
            else:
                mlflow.sklearn.log_model(model, "model")
                
        print(f"اطلاعات مدل {model_name} برای نسخه {data_version} با موفقیت در MLflow ثبت شد.")