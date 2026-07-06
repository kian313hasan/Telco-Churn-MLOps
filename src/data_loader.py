import pandas as pd
import os

def load_data(file_path):
    """
    تابع بارگذاری هوشمند و چندلایه داده‌های پروژه Telco Churn 
    (مقاوم در برابر خطاهای ساختاری، تغییر امتداد و انکودینگ)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"فایل مورد نظر در مسیر {file_path} یافت نشد.")
    
    print(f"--- در حال بارگذاری داده‌ها از: {file_path} ---")
    
    # تلاش اول: خواندن به عنوان CSV استاندارد با مدیریت انکودینگ
    try:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
            
        # بررسی اینکه آیا فایل واقعاً CSV است یا یک فایل اکسل تغییر نام یافته
        # اگر فقط ۱ ستون بخواند و خطا داده باشد، احتمالاً ساختار تفکیک نشده است
        if df.shape[1] <= 1:
            raise pd.errors.ParserError("ساختار فایل با CSV استاندارد همخوانی ندارد.")
            
        print("داده‌ها با موفقیت به عنوان فایل CSV واقعی بارگذاری شدند.")
        print(f"ابعاد دیتاست: {df.shape} (سطر، ستون)")
        return df

    # تلاش پشتیبان و نجات‌دهنده: اگر فایل در اصل اکسل باشد ولی پسوند CSV داشته باشد
    except (pd.errors.ParserError, Exception) as e:
        print("هشدار: ساختار CSV نامعتبر است. در حال تلاش برای بارگذاری به عنوان ساختار اکسل (Excel Fallback)...")
        try:
            df = pd.read_excel(file_path)
            print("داده‌ها با موفقیت از لایه پشتیبان به عنوان فایل Excel واقعی بارگذاری شدند.")
            print(f"ابعاد دیتاست: {df.shape} (سطر، ستون)")
            return df
        except Exception as excel_error:
            raise ValueError(f"خطا در خواندن فایل داده (پایپ‌لاین در هر دو فرمت شکست خورد): {excel_error}")

if __name__ == "__main__":
    # تست ماژول به صورت مستقل
    try:
        test_df = load_data("data/v1/raw_churn.csv")
        print("\n۵ ستون اول دیتاست:")
        print(list(test_df.columns)[:5])
    except Exception as e:
        print(f"خطا در تست ماژول: {e}")