import pandas as pd
import os

def load_data(file_path):
    """
    تابع بارگذاری داده‌های اکسل پروژه Telco Churn
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"فایل مورد نظر در مسیر {file_path} یافت نشد.")
    
    print(f"--- در حال بارگذاری داده‌ها از: {file_path} ---")
    try:
        # خواندن فایل به صورت اکسل واقعی
        df = pd.read_excel(file_path)
        print("داده‌ها با موفقیت به عنوان فایل Excel بارگذاری شدند.")
        print(f"ابعاد دیتاست: {df.shape} (سطر، ستون)")
        return df
    except Exception as e:
        raise ValueError(f"خطا در خواندن فایل اکسل: {e}")

if __name__ == "__main__":
    try:
        test_df = load_data("project/data/v1/raw_churn.csv")
        print("\n۵ ستون اول دیتاست:")
        print(list(test_df.columns)[:5])
    except Exception as e:
        print(f"خطا در تست ماژول: {e}")
