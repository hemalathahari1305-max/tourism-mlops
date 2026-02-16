FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY ../artifacts/best_pipeline.joblib /app/best_pipeline.joblib

EXPOSE 7860
CMD ["python", "app.py"]
