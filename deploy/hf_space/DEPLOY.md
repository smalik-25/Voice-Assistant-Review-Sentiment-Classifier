# Deploy to a Hugging Face Space

A Space is its own git repo. The steps below assemble the four files it needs (Dockerfile, a
`requirements.txt`, `app.py`, and the model bundle) and push. Hugging Face builds the
Dockerfile and gives you a public URL.

## One-time setup

1. Create a Hugging Face account and, if you have not, run `huggingface-cli login` (or set up
   a git credential with a write token from https://huggingface.co/settings/tokens).
2. On huggingface.co, create a new Space: SDK = Docker, blank template. Note its URL, which
   looks like `https://huggingface.co/spaces/<user>/<space>`.

## Assemble and push

Run this from the repo root, in `.venv`, with `<user>/<space>` filled in:

```bash
# 1. build the model bundle locally (needs the dataset at data/raw/)
python -m src.serving.export_model

# 2. clone the empty Space repo next to this one
git clone https://huggingface.co/spaces/<user>/<space> hf-space
cd hf-space

# 3. copy in the four files the Space needs
cp ../deploy/hf_space/Dockerfile .
cp ../deploy/hf_space/README.md .
cp ../requirements-serve.txt requirements.txt
cp ../src/serving/app.py app.py
mkdir -p models && cp ../models/model.joblib models/model.joblib

# 4. push; the Space builds automatically
git add .
git commit -m "Alexa review sentiment inference service"
git push
```

When the build finishes, the Space serves at `https://<user>-<space>.hf.space`. Open `/` for
the demo page or `/docs` for the API. Link it from the repo README and your site.

## Notes

- The model bundle is small and gets committed to the Space repo, which is fine. The raw
  dataset is never involved; only the fitted pipeline ships.
- Free Spaces sleep after inactivity and cold-start on the next request, which is expected
  for a demo.
- To update the model later, rerun `export_model`, copy the new `models/model.joblib` into
  `hf-space`, and push again.
