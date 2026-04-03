# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — EasyVisa Visa Approval Prediction API (V6 — S3 Artifact Store)
#
# BUILD:   docker build -t visa-api:v6 .
#
# RUN (all five -e flags required):
#   docker run -d -p 8000:8000 `
#     -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
#     -e MLFLOW_MODEL_URI=models:/easyvisa_gbm/Production `
#     -e AWS_ACCESS_KEY_ID=AKIA... `
#     -e AWS_SECRET_ACCESS_KEY=... `
#     -e AWS_DEFAULT_REGION=us-east-1 `
#     --name visa-api-v6 visa-api:v6
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — pip install only reruns if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# ── ENV VARS ──────────────────────────────────────────────────────────────────
# MLflow — empty = container exits immediately (fail-fast)
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_URI="models:/easyvisa_gbm/Production"

# AWS — required for S3 artifact download via boto3
# Never hardcode real values here — always pass via docker run -e flags
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_DEFAULT_REGION="us-east-1"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD python -c "\
import urllib.request, json, sys; \
resp = urllib.request.urlopen('http://localhost:8000/health', timeout=10); \
data = json.loads(resp.read()); \
print('Health OK:', data.get('status'), '| version:', data.get('model_version')); \
sys.exit(0) if data.get('status') == 'healthy' else sys.exit(1)"

# src.api:app — correct module path since api.py lives in src/ folder
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]