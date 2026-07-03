# استفاده از نسخه سبک پایتون
FROM python:3.12-slim

# تعیین پوشه کاری داخل داکر
WORKDIR /app

# آپدیت کردن خود pip برای جلوگیری از خطای نصب
RUN pip install --no-cache-dir --upgrade pip

# نصب مقطعی کتابخانه‌ها برای پایداری در دانلود
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy openpyxl
RUN pip install --no-cache-dir scikit-learn mlflow
RUN pip install --no-cache-dir xgboost catboost

# کپی کردن کدهای پروژه و دیتابیس MLflow به داخل کانتینر
COPY ./src /app/src
COPY ./mlruns /app/mlruns
COPY ./app.py /app/app.py

# باز کردن پورت ۸۰۰۰ برای سرویس API
EXPOSE 8000

# دستور اجرای سرویس هنگام روشن شدن کانتینر
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]