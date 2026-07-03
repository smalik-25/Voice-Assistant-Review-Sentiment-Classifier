# Serving

A FastAPI service that scores review text with the selected model at its operating
threshold.

## Rebuild the model and run it

The model artifact is not committed, so I rebuild it from config and seed, then serve:

```bash
# curated data has to exist first
python -m src.data.build_curated

# export the serving bundle to models/model.joblib
# (deterministic from configs/serve_model.yaml plus the seed)
python -m src.serving.export_model

# run the API directly
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# or in the container, which installs only requirements-serve.txt
docker build -t alexa-sentiment .
docker run -p 8000:8000 alexa-sentiment
```

## Endpoints

`GET /health` returns the service status, whether the model loaded, and the operating
threshold.

`POST /predict` takes `{"text": "..."}` and returns the predicted `feedback` (1 positive, 0
negative), a `label`, the `negative_probability`, and the `threshold` it applied.

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "it stopped working after a week and support was useless"}'
```

## Why it is reproducible

The served model is logistic regression on TF-IDF (L2, C=10, unweighted), which is the
configuration the PR-AUC ablation picked. `configs/serve_model.yaml` freezes those
hyperparameters as single values, and `export_model.py` fits them on the curated data with a
fixed seed and computes the operating threshold on out-of-fold predictions. Same config, same
seed, same artifact. The container pins its dependencies in `requirements-serve.txt`, and it
carries none of the training or deep-learning stack.
