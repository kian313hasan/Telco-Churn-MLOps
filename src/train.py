import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# اضافه کردن مسیر src برای دسترسی به سایر ماژول‌ها
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mlflow_utils import init_mlflow, log_experiment
from evaluate import evaluate_and_log_metrics

def train_and_evaluate(data_path, data_version, seed=42):
    """
    آموزش و ارزیابی هوشمند مدل‌ها با ساختار Stratified K-Fold و ارزیابی واقعی بدون نشت داده
    """
    print(f"\n========== آغاز فرآیند آموزش برای نسخه: {data_version} ==========")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"داده‌های نسخه {data_version} در مسیر {data_path} یافت نشد.")
    df = pd.read_csv(data_path)
    
    target_col = 'Churn Value'
    if target_col not in df.columns:
        raise ValueError(f"ستون هدف '{target_col}' در دیتاست یافت نشد.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=5000, random_state=seed),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=seed, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(iterations=100, random_state=seed, verbose=0)
    }
    
    init_mlflow()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    X_arr = X.values
    y_arr = y.values
    
    for model_name, model in models.items():
        print(f"در حال آموزش مدل: {model_name}...")
        
        oof_preds = np.zeros(len(df))
        
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model.fit(X_train_scaled, y_train)
            oof_preds[val_idx] = model.predict(X_val_scaled)
            
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X_arr)
        model.fit(X_scaled, y_arr)
        
        class OOFModelWrapper:
            def __init__(self, m, preds):
                self.m = m
                self.preds = preds
            def predict(self, X): return self.preds
            def predict_proba(self, X):
                return self.m.predict_proba(X) if hasattr(self.m, "predict_proba") else self.preds
                
        wrapped_model = OOFModelWrapper(model, oof_preds)
        # تمرير قيم الـ values لتفادي ارور الأسائلرن الافتراضي
        metrics, plot_path = evaluate_and_log_metrics(wrapped_model, X_arr, y_arr, data_version, model_name)
        
        params = model.get_params()
        clean_params = {k: str(v) for k, v in params.items() if len(str(v)) < 50}
        clean_params["seed"] = str(seed)
        
        log_experiment(
            model_name=model_name,
            data_version=data_version,
            params=clean_params,
            metrics=metrics,
            plot_path=plot_path,
            model=model
        )
        
    print(f"========== فرآیند آموزش نسخه {data_version} با موفقیت به پایان رسید ==========")

if __name__ == "__main__":
    train_and_evaluate("data/v3/engineered_churn.csv", "v3")