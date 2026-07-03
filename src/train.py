import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# اضافه کردن مسیر src برای دسترسی به mlflow_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mlflow_utils import init_mlflow, log_experiment

def train_and_evaluate(data_path, data_version, seed=42):
    """
    آموزش و ارزیابی مدل‌ها روی نسخه مشخصی از داده‌ها با استانداردسازی کامل و رفع خطای همگرایی
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
    
    # تعریف مدل‌ها با افزایش max_iter برای لجستیک جهت رفع کامل تحذير همگرایی
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=5000, random_state=seed),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=seed, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(iterations=100, random_state=seed, verbose=0)
    }
    
    init_mlflow()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for model_name, model in models.items():
        print(f"در حال آموزش مدل: {model_name}...")
        
        oof_preds = np.zeros(len(df))
        oof_pred_probs = np.zeros(len(df))
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # اعمال استانداردسازی کامل روی کل ماتریکس ویژگی‌ها به صورت محلی در هر Fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model.fit(X_train_scaled, y_train)
            
            oof_preds[val_idx] = model.predict(X_val_scaled)
            if hasattr(model, "predict_proba"):
                oof_pred_probs[val_idx] = model.predict_proba(X_val_scaled)[:, 1]
            else:
                oof_pred_probs[val_idx] = oof_preds[val_idx]
                
        metrics = {
            "Accuracy": accuracy_score(y, oof_preds),
            "Precision": precision_score(y, oof_preds, zero_division=0),
            "Recall": recall_score(y, oof_preds, zero_division=0),
            "F1-Score": f1_score(y, oof_preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y, oof_pred_probs)
        }
        
        params = model.get_params()
        clean_params = {k: str(v) for k, v in params.items() if len(str(v)) < 50}
        clean_params["seed"] = str(seed)
        
        log_experiment(
            model_name=model_name,
            data_version=data_version,
            params=clean_params,
            metrics=metrics,
            y_true=y,
            y_pred=oof_preds,
            model=model
        )
        
    print(f"========== فرآیند آموزش نسخه {data_version} با موفقیت و بدون خطا به پایان رسید ==========")

if __name__ == "__main__":
    train_and_evaluate("project/data/v3/engineered_churn.csv", "v3")
