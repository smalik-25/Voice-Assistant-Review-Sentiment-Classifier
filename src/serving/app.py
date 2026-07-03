"""FastAPI inference service for the Alexa review sentiment model.

Loads the serving bundle exported by ``src.serving.export_model`` and exposes ``/health``
and ``/predict``. Predictions apply the model's operating threshold (chosen against the
negative-recall target), not the default 0.5, so a review is flagged negative when the
predicted negative probability clears that threshold.

Deliberately standalone: it imports only the slim serving stack (fastapi, pydantic, joblib,
numpy) plus a plain scikit-learn pipeline inside the bundle. No project code or MLflow is
needed at serve time.
"""
from __future__ import annotations

import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

NEG = 0
DEFAULT_MODEL_PATH = "models/model.joblib"

_DEMO_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Alexa Review Sentiment</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:48px auto;padding:0 16px;color:#1a1a1a}
 textarea{width:100%;height:96px;font-size:15px;padding:8px;box-sizing:border-box}
 button{margin-top:8px;padding:8px 16px;font-size:15px;cursor:pointer}
 .out{margin-top:20px;font-size:18px;min-height:24px}
 .muted{color:#666;font-size:14px}
</style></head>
<body>
<h2>Alexa review sentiment</h2>
<p class="muted">Paste a review. The model returns positive or negative at its operating
threshold (tuned for negative-class recall), and the probability it assigns to negative.</p>
<textarea id="t" placeholder="it stopped working after a week and support was useless"></textarea>
<br><button onclick="go()">Predict</button>
<div class="out" id="o"></div>
<script>
async function go(){
  const t = document.getElementById('t').value;
  const o = document.getElementById('o');
  o.textContent = 'scoring...';
  try{
    const r = await fetch('predict', {method:'POST',
      headers:{'content-type':'application/json'}, body: JSON.stringify({text: t})});
    if(!r.ok){ o.textContent = 'error: ' + r.status; return; }
    const d = await r.json();
    o.textContent = d.label.toUpperCase() +
      '  (p_negative=' + d.negative_probability.toFixed(3) +
      ', threshold=' + d.threshold.toFixed(3) + ')';
  }catch(e){ o.textContent = 'error: ' + e; }
}
</script>
</body></html>
"""


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="the review text to score")


class PredictResponse(BaseModel):
    feedback: int = Field(..., description="1 positive, 0 negative")
    label: str
    negative_probability: float
    threshold: float


def _load_bundle(path: str):
    try:
        return joblib.load(path)
    except (FileNotFoundError, OSError):
        return None


def create_app(model_path: str | None = None) -> FastAPI:
    path = model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    bundle = _load_bundle(path)
    app = FastAPI(title="Alexa Review Sentiment Classifier", version="1.0")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _DEMO_HTML

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if bundle is not None else "no_model",
            "model_loaded": bundle is not None,
            "threshold": bundle["threshold"] if bundle else None,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if bundle is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        pipeline = bundle["pipeline"]
        neg_col = list(pipeline.classes_).index(NEG)
        neg_prob = float(pipeline.predict_proba([req.text])[0][neg_col])
        threshold = float(bundle["threshold"])
        feedback = NEG if neg_prob >= threshold else 1
        return PredictResponse(
            feedback=feedback,
            label="negative" if feedback == NEG else "positive",
            negative_probability=neg_prob,
            threshold=threshold,
        )

    return app


app = create_app()
