import os
import sys

# اضافه کردن پوشه src به مسیر پایتون برای دسترسی به ماژول‌ها
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from preprocessing import preprocess_data
from features import engineer_features
from train import train_and_evaluate
from evaluate import evaluate_and_log_metrics  # <--- إضافة سكريبت التقييم المنفصل هنا لحساب المتریکس

def main():
    print("=========================================================")
    print("   آغاز اجرای خط لوله جامع MLOps پروژه Telco Churn    ")
    print("=========================================================\n")
    
    # تعریف مسیرهای فایل‌ها
    raw_data_path = "project/data/v1/raw_churn.csv"
    cleaned_data_path = "project/data/v2/cleaned_churn.csv"
    engineered_data_path = "project/data/v3/engineered_churn.csv"
    
    # مرحله ۱: اجرای پیش‌پردازش و پاک‌سازی (تولید نسخه v2)
    print("[گام ۱/۳] آغاز فرآیند پاک‌سازی و تولید نسخه v2...")
    preprocess_data(raw_data_path, cleaned_data_path)
    
    # مرحله ۲: اجرای مهندسی ویژگی‌ها و نرمال‌سازی (تولید نسخه v3)
    print("\n[گام ۲/۳] آغاز فرآیند مهندسی ویژگی‌ها و تولید نسخه v3...")
    engineer_features(cleaned_data_path, engineered_data_path)
    
    # مرحله ۳: آموزش مدل‌ها و ثبت در MLflow روی نسخه v3
    print("\n[گام ۳/۳] آغاز فرآیند آموزش مدل‌ها و ثبت نتایج در MLflow...")
    train_and_evaluate(engineered_data_path, data_version="v3_final")
    
    print("\n=========================================================")
    print("   اجرای خط لوله اتوماتیک با موفقیت کامل به پایان رسید!   ")
    print("=========================================================")

if __name__ == "__main__":
    main()