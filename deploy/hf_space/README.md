---
title: Alexa Review Sentiment
emoji: 🗣️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Alexa Review Sentiment

A live FastAPI service that scores Amazon Alexa review text as positive or negative. The
model is logistic regression on TF-IDF, applied at an operating threshold tuned for
negative-class recall rather than the default 0.5.

- `/` a small demo page: paste a review, get the prediction.
- `/docs` the interactive API docs.
- `POST /predict` with `{"text": "..."}` returns the label, the negative probability, and the
  threshold.

Source and the full write-up: https://github.com/smalik-25/Voice-Assistant-Review-Sentiment-Classifier
