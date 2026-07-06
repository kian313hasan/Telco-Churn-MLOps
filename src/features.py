import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def engineer_features(input_path, output_path):
    """
    مرحله ۳: ساخت ویژگی‌های جدید و نرمال‌سازی داده‌ها (v3) مطابق با اصول جامع مهندسی ویژگی‌ها
    """
    print(f"--- در حال بارگذاری داده‌های نسخه v2 از: {input_path} ---")
    df = pd.read_csv(input_path)
    
    print("\n--- آغاز فرآیند Feature Engineering (v3) ---")
    
    # ۱. ساخت ویژگی‌های جدید (Feature Engineering)
    # ویژگی اول: نسبت هزینه ماهانه به تعداد ماه‌های حضور (برای سنجش وفاداری و هزینه)
    if 'Monthly Charges' in df.columns and 'Tenure Months' in df.columns:
        df['Charges_Per_Month'] = df['Monthly Charges'] / (df['Tenure Months'] + 1)
        
    # ویژگی دوم: جمع‌بندی تعداد خدماتی که کاربر آنلاین فعال کرده است
    service_cols = ['Phone Service', 'Multiple Lines', 'Internet Service', 
                    'Online Security', 'Online Backup', 'Device Protection', 
                    'Tech Support', 'Streaming TV', 'Streaming Movies']
    existing_services = [col for col in service_cols if col in df.columns]
    if existing_services:
        # چون انکودینگ شده، چک می‌کنیم مقادیر بزرگتر از 0 (یعنی سرویس فعال است) جمع شوند
        df['Total_Services_Count'] = df[existing_services].gt(0).sum(axis=1)

    print(f"ویژگی‌های جدید با موفقیت ساخته شدند. تعداد کل ستون‌ها: {df.shape[1]}")
    
    # ۲. نرمال‌سازی داده‌های عددی (Scaling) شامل تمام مژول‌های عددی و ویژگی‌های جدید ساخته شده
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Charges_Per_Month', 'Total_Services_Count']
    num_cols = [col for col in num_cols if col in df.columns]
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"نرمال‌سازی (StandardScaler) برای ستون‌های عددی انجام شد: {num_cols}")
    
    # ۳. ذخیره نسخه v3
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"نسخه نهایی ویژگی‌ها (v3) با موفقیت در مسیر زیر ذخیره شد:\n{output_path}")
    print(f"ابعاد نهایی نسخه v3: {df.shape} (سطر، ستون)")
    return df

if __name__ == "__main__":
    input_file = "project/data/v2/cleaned_churn.csv"
    output_file = "project/data/v3/engineered_churn.csv"
    
    try:
        engineer_features(input_file, output_file)
    except Exception as e:
        print(f"خطا در اجرای فرآیند مهندسی ویژگی‌ها: {e}")