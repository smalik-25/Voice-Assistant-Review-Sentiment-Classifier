# Slim inference image. Installs only the serving runtime (requirements-serve.txt), not the
# training or deep-learning stack. Build context must contain models/model.joblib, produced
# by `python -m src.serving.export_model` before building.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/serving/app.py .
COPY models/model.joblib models/model.joblib
ENV MODEL_PATH=models/model.joblib

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
