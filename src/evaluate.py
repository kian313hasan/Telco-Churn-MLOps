import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow

def evaluate_and_log_metrics(model, X_test, y_test, dataset_version, model_name):
    """
    حساب المقاييس الخمسة المطلوبة ورسم Confusion Matrix ولصقها في MLflow
    """
    # التنبؤ
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # حساب المتریکس
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
    
    # تسجيل المتریکس في MLflow
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{metric_name}_{dataset_version}", value)
        print(f"[{model_name} - {dataset_version}] {metric_name}: {value:.4f}")
        
    # رسم وحفظ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"]).plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name} ({dataset_version})")
    
    # حفظ الصورة كـ Artifact
    plot_path = f"confusion_matrix_{model_name}_{dataset_version}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    mlflow.log_artifact(plot_path, artifact_path="artifacts")
    if os.path.exists(plot_path):
        os.remove(plot_path)
        
    return metrics