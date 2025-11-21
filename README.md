# Brain Tumor Detection and Classification

## What this repo contains
- Code for inference and demo server.
- Pretrained models are tracked with Git LFS (`*.h5`, `*.pt`).
- Minimal web API to run single-image inference.

## Quick start (local)
1. Create a Python venv and install:
```bash
python -m venv .venv
# Git Bash / WSL:
source .venv/bin/activate
# Windows CMD / PowerShell:
# .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run server:
```bash
python app.py
# or in prod via gunicorn:
# gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

3. Send inference (example):
```bash
curl -X POST -F "image=@sample.jpg" http://127.0.0.1:8000/predict
```

## Notes
- Models are large; they are tracked with Git LFS.
- For sharing big models consider GitHub Releases or cloud storage (Drive / S3).
