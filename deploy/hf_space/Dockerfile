# Dockerfile for the Hugging Face Space. Same slim runtime as the main image, but listens on
# port 7860, which is what Spaces expect. The Space repo also holds app.py, requirements.txt
# (a copy of requirements-serve.txt), and models/model.joblib (see DEPLOY.md).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/model.joblib models/model.joblib
ENV MODEL_PATH=models/model.joblib

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
