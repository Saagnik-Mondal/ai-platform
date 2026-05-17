FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    lightgbm==4.5.0 \
    mlflow==2.16.2 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.32 \
    boto3==1.35.0

COPY . .

CMD ["python", "train_initial.py"]
