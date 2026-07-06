import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_and_log_metrics(model, X_test, y_test, dataset_version, model_name):
    """
    ماژول استاندارد و واحد برای محاسبه معیارهای پنج‌گانه ارزیابی و رسم ماتریس آشفتگی مطابق با اصول Clean Code
    """
    # ۱. انجام پیش‌بینی‌ها بر روی داده‌های تست
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # ۲. محاسبه معیارهای پنج‌گانه ارزیابی خواسته شده توسط استاد
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba))
    }
    
    # چاپ نتایج در ترمینال برای تایید فرآیند
    for metric_name, value in metrics.items():
        print(f"[{model_name} - {dataset_version}] {metric_name}: {value:.4f}")
        
    # ۳. رسم ساختار یافته ماتریس آشفتگی (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"]).plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name} ({dataset_version})")
    
    # ۴. ذخیره موقت تصویر نمودار جهت انتقال به ماژول مدیریت MLOps
    plot_path = f"confusion_matrix_{model_name}_{dataset_version}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
        
    # خروجی دادن متریک‌ها و مسیر تصویر بدون شکستن ساختار ماژولار پروژه
    return metrics, plot_path