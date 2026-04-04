"""
api.py — EasyVisa Visa Approval Prediction API  (V6 — Registry-Only + Monitoring)
FastAPI serving layer — loads model AND feature_names exclusively from MLflow

WHAT'S NEW IN V6 MONITORING:
  - Prometheus metrics: prediction counter, latency histogram, request counter
  - /metrics endpoint for Prometheus scraping
  - Structured per-request logging with continent + confidence
  - Error rate tracking by endpoint

REQUIRED ENV VARS:
  $env:MLFLOW_TRACKING_URI    = "http://<your-ip>:5000"
  $env:MLFLOW_MODEL_URI       = "models:/easyvisa_gbm/Production"
  $env:AWS_ACCESS_KEY_ID      = "your_key"
  $env:AWS_SECRET_ACCESS_KEY  = "your_secret"
  $env:AWS_DEFAULT_REGION     = "us-east-1"
"""

# ─── Standard library ────────────────────────────────────────────────────────
import logging
import os
import sys
import tempfile
import time

# ─── Third-party ─────────────────────────────────────────────────────────────
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel

# ─── Load .env (dev only — prod uses real env vars) ──────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────────────────────
prediction_counter = Counter(
    "easyvisa_predictions_total",
    "Total visa predictions made",
    ["prediction"],           # label: Certified / Denied
)

prediction_latency = Histogram(
    "easyvisa_prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

api_requests = Counter(
    "easyvisa_api_requests_total",
    "Total API requests by endpoint and status",
    ["endpoint", "status"],   # labels: endpoint=/predict, status=success|error
)

model_confidence = Histogram(
    "easyvisa_model_confidence",
    "Distribution of model confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

certified_ratio = Gauge(
    "easyvisa_certified_ratio",
    "Rolling ratio of Certified predictions (last 100)",
)

# Simple rolling window for certified ratio
_recent_predictions: list = []


def _update_certified_ratio(prediction: str) -> None:
    """Track rolling certified ratio over last 100 predictions."""
    global _recent_predictions
    _recent_predictions.append(1 if prediction == "Certified" else 0)
    _recent_predictions = _recent_predictions[-100:]   # keep last 100
    ratio = sum(_recent_predictions) / len(_recent_predictions)
    certified_ratio.set(ratio)


# ─────────────────────────────────────────────────────────────────────────────
# REQUIRE ENV VARS — fail-fast with clear fix instructions
# ─────────────────────────────────────────────────────────────────────────────
def require_env_vars() -> tuple:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    model_uri    = os.getenv("MLFLOW_MODEL_URI", "").strip()
    errors = []

    if not tracking_uri:
        errors.append(
            "  MLFLOW_TRACKING_URI is not set.\n"
            "  Fix: $env:MLFLOW_TRACKING_URI = 'http://192.168.x.x:5000'"
        )
    if not model_uri:
        errors.append(
            "  MLFLOW_MODEL_URI is not set.\n"
            "  Fix: $env:MLFLOW_MODEL_URI = 'models:/easyvisa_gbm/Production'"
        )
    if errors:
        log.error("=" * 65)
        log.error("API STARTUP FAILED — required env vars missing:")
        for e in errors:
            log.error(e)
        log.error("=" * 65)
        sys.exit(1)

    log.info("MLFLOW_TRACKING_URI : %s", tracking_uri)
    log.info("MLFLOW_MODEL_URI    : %s", model_uri)
    return tracking_uri, model_uri


# ─────────────────────────────────────────────────────────────────────────────
# RESOLVE MODEL URI → RUN_ID
# ─────────────────────────────────────────────────────────────────────────────
def resolve_run_id(
    client: mlflow.tracking.MlflowClient, model_uri: str
) -> tuple:
    uri_parts = model_uri.replace("models:/", "").split("/")
    reg_name  = uri_parts[0]
    stage_ver = uri_parts[1] if len(uri_parts) > 1 else "Production"

    try:
        versions = client.get_latest_versions(reg_name, stages=[stage_ver])
        if versions:
            return versions[0].run_id, versions[0].version
    except Exception:
        pass

    try:
        v = client.get_model_version(reg_name, stage_ver)
        return v.run_id, v.version
    except Exception:
        pass

    all_versions = client.search_model_versions(f"name='{reg_name}'")
    if not all_versions:
        raise ValueError(
            f"No versions found for model '{reg_name}'. "
            "Run train.py first."
        )
    latest = sorted(all_versions, key=lambda x: int(x.version))[-1]
    return latest.run_id, latest.version


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER — MLflow Registry Only
# ─────────────────────────────────────────────────────────────────────────────
def load_from_registry(tracking_uri: str, model_uri: str) -> tuple:
    mlflow.set_tracking_uri(tracking_uri)
    log.info("Connecting to MLflow: %s", tracking_uri)

    log.info("Loading model from registry: %s", model_uri)
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        log.error("FAILED to load model: %s", e)
        log.error("Fix: python -m mlflow server --host 0.0.0.0 --port 5000")
        sys.exit(1)

    log.info("✅ Model loaded: %s", type(model).__name__)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run_id, model_version = resolve_run_id(client, model_uri)
    log.info("Resolved run_id=%s  version=%s", run_id, model_version)

    feature_artifact_uri = f"runs:/{run_id}/model/feature_names.pkl"
    log.info("Downloading feature_names: %s", feature_artifact_uri)

    try:
        tmp_dir    = tempfile.mkdtemp(prefix="easyvisa_features_")
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=feature_artifact_uri,
            dst_path=tmp_dir,
        )
        feature_names = joblib.load(local_path)
    except Exception as e:
        log.error("FAILED to download feature_names.pkl: %s", e)
        log.error("Fix: python src/train.py --data-path data/EasyVisa.csv")
        sys.exit(1)

    log.info("✅ feature_names loaded: %d features", len(feature_names))

    model_source = (
        f"MLflow Registry | {model_uri} | "
        f"version={model_version} | run_id={run_id[:8]}..."
    )
    return model, feature_names, model_source, run_id, str(model_version)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
TRACKING_URI, MODEL_URI = require_env_vars()

model, feature_names, MODEL_SOURCE, RUN_ID, MODEL_VERSION = load_from_registry(
    TRACKING_URI, MODEL_URI
)

log.info("=" * 65)
log.info("✅ API STARTUP COMPLETE")
log.info("   Model type    : %s", type(model).__name__)
log.info("   Model version : %s", MODEL_VERSION)
log.info("   Features      : %d", len(feature_names))
log.info("   Source        : %s", MODEL_SOURCE)
log.info("   Docs          : http://localhost:8000/docs")
log.info("   Metrics       : http://localhost:8000/metrics")
log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visa Approval Prediction API  (V6 — Registry-Only + Monitoring)",
    description=(
        "Predicts US visa approval probability using a GradientBoosting model "
        "trained on OFLC historical data (25,480 applications). "
        "Model and feature schema loaded exclusively from MLflow Model Registry. "
        "Prometheus metrics available at /metrics."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
class VisaApplication(BaseModel):
    continent:             str
    education_of_employee: str
    has_job_experience:    str
    requires_job_training: str
    no_of_employees:       int
    yr_of_estab:           int
    region_of_employment:  str
    prevailing_wage:       float
    unit_of_wage:          str
    full_time_position:    str

    model_config = {
        "json_schema_extra": {
            "example": {
                "continent":             "Asia",
                "education_of_employee": "Master's",
                "has_job_experience":    "Y",
                "requires_job_training": "N",
                "no_of_employees":       500,
                "yr_of_estab":           2010,
                "region_of_employment":  "West",
                "prevailing_wage":       85000.0,
                "unit_of_wage":          "Yearly",
                "full_time_position":    "Y",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def home():
    return {
        "message": "Visa Approval Prediction API  (V6 — Registry-Only + Monitoring)",
        "status":  "running",
        "version": "4.0.0",
        "docs":    "/docs",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["Status"])
def health():
    api_requests.labels(endpoint="/health", status="success").inc()
    return {
        "status":        "healthy",
        "model_source":  MODEL_SOURCE,
        "model_type":    type(model).__name__,
        "model_version": MODEL_VERSION,
        "run_id":        RUN_ID,
        "n_features":    len(feature_names),
        "tracking_uri":  TRACKING_URI,
    }


@app.post("/predict", tags=["Prediction"])
def predict_visa(application: VisaApplication):
    start_time = time.time()

    try:
        # ── Preprocessing ─────────────────────────────────────────
        input_df = pd.DataFrame([application.model_dump()])
        encoded  = pd.get_dummies(input_df, drop_first=True)

        for col in feature_names:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded[feature_names]

        # ── Predict ───────────────────────────────────────────────
        pred  = model.predict(encoded)[0]
        proba = model.predict_proba(encoded)[0]

        result = {
            "prediction":            "Certified" if pred == 1 else "Denied",
            "probability_certified": round(float(proba[1]), 4),
            "probability_denied":    round(float(proba[0]), 4),
            "confidence":            round(float(max(proba)), 4),
            "model_version":         MODEL_VERSION,
        }

        # ── Metrics ───────────────────────────────────────────────
        prediction_counter.labels(prediction=result["prediction"]).inc()
        model_confidence.observe(result["confidence"])
        api_requests.labels(endpoint="/predict", status="success").inc()
        _update_certified_ratio(result["prediction"])

        # ── Structured log ────────────────────────────────────────
        log.info(
            "PREDICT | continent=%-12s | result=%-9s | "
            "confidence=%.4f | latency=%.3fs",
            application.continent,
            result["prediction"],
            result["confidence"],
            time.time() - start_time,
        )

        return result

    except KeyError as e:
        api_requests.labels(endpoint="/predict", status="error").inc()
        log.error("Feature mismatch: %s", e)
        raise HTTPException(
            status_code=422,
            detail=f"Feature mismatch — '{e}' not found. Check preprocessing.",
        )
    except Exception as e:
        api_requests.labels(endpoint="/predict", status="error").inc()
        log.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        prediction_latency.observe(time.time() - start_time)


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics scraping endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info", tags=["Model"])
def model_info():
    """Full model metadata for auditing and CI/CD verification."""
    return {
        "model_type":    type(model).__name__,
        "model_source":  MODEL_SOURCE,
        "model_version": MODEL_VERSION,
        "run_id":        RUN_ID,
        "tracking_uri":  TRACKING_URI,
        "model_uri":     MODEL_URI,
        "n_features":    len(feature_names),
        "features":      feature_names,
        "api_version":   "4.0.0",
    }