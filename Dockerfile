FROM python:3.10.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api.py .

# V10: Default to MLflow Docker Compose service name
# On EKS, overridden via Kubernetes secret
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"
ENV MLFLOW_MODEL_URI="models:/easyvisa_gbm/Production"
ENV AWS_DEFAULT_REGION="us-east-1"

# AWS credentials injected at runtime via Docker Compose (.env) or Kubernetes Secret
# Do NOT bake AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY into the image

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD python -c "\
import urllib.request, json, sys; \
resp = urllib.request.urlopen('http://localhost:8000/health', timeout=10); \
data = json.loads(resp.read()); \
print('Health OK:', data.get('status'), '| version:', data.get('model_version')); \
sys.exit(0) if data.get('status') == 'healthy' else sys.exit(1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]