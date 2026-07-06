import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder

# اضافه کردن مسیر src به سیستم برای دسترسی به data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

def preprocess_data(input_path, output_path):
    """
    مرحله ۲: پاک‌سازی داده‌ها، مدیریت مقادیر مفقوده و Encoding استاندارد برای نسخه v2
    """
    # ۱. بارگذاری داده‌ها با استفاده از ماژول مرحله قبل
    df = load_data(input_path)
    
    print("\n--- آغاز فرآیند پاک‌سازی داده‌ها (v2) ---")
    
    # ۲. لیست ستون‌های غیرضروری برای حذف
    columns_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Reason'
    ]
    
    # حذف ستون‌ها در صورت وجود در دیتاست
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"تعداد {len(existing_drops)} ستون غیرضروری حذف شدند. ستون‌های باقی‌مانده: {df.shape[1]}")
    
    # ۳. رفع مشکل مقادیر مفقوده در Total Charges
    if 'Total Charges' in df.columns:
        # تنظیم آپشن پاندا برای جلوگیری از FutureWarning در داون‌کستینگ
        pd.set_option('future.no_silent_downcasting', True)
        
        # تبدیل فضاهای خالی متنی به NaN
        df['Total Charges'] = df['Total Charges'].replace(r'^\s*$', np.nan, regex=True)
        # تبدیل کل ستون به فرمت عددی
        df['Total Charges'] = pd.to_numeric(df['Total Charges'])
        # جایگزینی مقادیر مفقوده با صفر (چون کارکرد ماه‌های حضور آن‌ها صفر بوده است)
        df['Total Charges'] = df['Total Charges'].fillna(0)
        print("مقادیر خالی متنی در ستون Total Charges به عدد صفر تبدیل و اصلاح شدند.")
    
    # حذف هرگونه سطر با مقدار مفقوده احتمالی دیگر
    df = df.dropna()
    
    # ۴. تبدیل داده‌های متنی (Categorical) به عددی با استفاده از LabelEncoder استاندارد
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"ستون‌های متنی پیدا شده برای Encoding: {list(categorical_cols)}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    print(f"فرآیند Encoding استاندارد برای {len(categorical_cols)} ستون با موفقیت انجام شد.")
    
    # ۵. ذخیره نسخه v2
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"نسخه پاک‌سازی شده (v2) با موفقیت در مسیر زیر ذخیره شد:\n{output_path}")
    print(f"ابعاد نهایی نسخه v2: {df.shape} (سطر، ستون)")
    return df

if __name__ == "__main__":
    input_file = "project/data/v1/raw_churn.csv"
    output_file = "project/data/v2/cleaned_churn.csv"
    
    try:
        preprocess_data(input_file, output_file)
    except Exception as e:
        print(f"خطا در اجرای فرآیند پیش‌پردازش: {e}")