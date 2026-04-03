"""
api.py — EasyVisa Visa Approval Prediction API  (V5 — Registry-Only)
FastAPI serving layer — loads model AND feature_names exclusively from MLflow

REQUIRED ENV VARS (set before starting):
  $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
  $env:MLFLOW_MODEL_URI    = "models:/easyvisa_gbm/Production"

  # AWS (optional — only needed if using S3 artifact store)
  $env:AWS_ACCESS_KEY_ID     = "your_key"
  $env:AWS_SECRET_ACCESS_KEY = "your_secret"
"""

# ─── Standard library ────────────────────────────────────────────────────────
import logging
import os
import sys
import tempfile

# ─── Third-party ─────────────────────────────────────────────────────────────
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── Load .env file (dev only — prod uses real env vars) ─────────────────────
load_dotenv()

# ─── AWS credentials (from environment only — never hardcoded) ───────────────
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Propagate to boto3/MLflow S3 client if present
if AWS_ACCESS_KEY_ID:
    os.environ["AWS_ACCESS_KEY_ID"]     = AWS_ACCESS_KEY_ID
if AWS_SECRET_ACCESS_KEY:
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# REQUIRE BOTH ENV VARS — No defaults, no fallback
# ─────────────────────────────────────────────────────────────────────────────
def require_env_vars() -> tuple:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    model_uri    = os.getenv("MLFLOW_MODEL_URI", "").strip()
    errors = []

    if not tracking_uri:
        errors.append(
            "  MLFLOW_TRACKING_URI is not set.\n"
            "  Fix: $env:MLFLOW_TRACKING_URI = 'http://localhost:5000'"
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
        log.error("")
        log.error("Full startup sequence:")
        log.error("  1. python -m mlflow server --host 0.0.0.0 --port 5000")
        log.error("  2. $env:MLFLOW_TRACKING_URI = 'http://localhost:5000'")
        log.error("  3. $env:MLFLOW_MODEL_URI    = 'models:/easyvisa_gbm/Production'")
        log.error("  4. python -m uvicorn src.api:app --host 0.0.0.0 --port 8000")
        log.error("=" * 65)
        sys.exit(1)

    log.info("MLFLOW_TRACKING_URI : %s", tracking_uri)
    log.info("MLFLOW_MODEL_URI    : %s", model_uri)
    return tracking_uri, model_uri


# ─────────────────────────────────────────────────────────────────────────────
# RESOLVE MODEL URI TO RUN_ID
# ─────────────────────────────────────────────────────────────────────────────
def resolve_run_id(client: mlflow.tracking.MlflowClient, model_uri: str) -> tuple:
    uri_parts  = model_uri.replace("models:/", "").split("/")
    reg_name   = uri_parts[0]
    stage_ver  = uri_parts[1] if len(uri_parts) > 1 else "Production"

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
            f"Run train.py first: python src/train.py --data-path data/EasyVisa.csv"
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
        log.error("=" * 65)
        log.error("FAILED to load model from MLflow registry.")
        log.error("  model_uri : %s", model_uri)
        log.error("  error     : %s", e)
        log.error("  Fix: python -m mlflow server --host 0.0.0.0 --port 5000")
        log.error("=" * 65)
        sys.exit(1)

    log.info("✅ Model loaded: %s", type(model).__name__)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run_id, model_version = resolve_run_id(client, model_uri)
    log.info("Resolved run_id=%s  version=%s", run_id, model_version)

    feature_artifact_uri = f"runs:/{run_id}/model/feature_names.pkl"
    log.info("Downloading feature_names artifact: %s", feature_artifact_uri)

    try:
        tmp_dir    = tempfile.mkdtemp(prefix="easyvisa_features_")
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=feature_artifact_uri,
            dst_path=tmp_dir,
        )
        feature_names = joblib.load(local_path)
    except Exception as e:
        log.error("=" * 65)
        log.error("FAILED to download feature_names.pkl from MLflow artifacts.")
        log.error("  artifact_uri : %s", feature_artifact_uri)
        log.error("  run_id       : %s", run_id)
        log.error("  error        : %s", e)
        log.error("  Fix: python src/train.py --data-path data/EasyVisa.csv")
        log.error("=" * 65)
        sys.exit(1)

    log.info("✅ feature_names loaded: %d features", len(feature_names))

    model_source = (
        f"MLflow Registry | {model_uri} | "
        f"version={model_version} | run_id={run_id[:8]}..."
    )
    log.info("✅ Model source: %s", model_source)

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
log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visa Approval Prediction API  (V5 — Registry-Only)",
    description=(
        "Predicts US visa approval probability using a GradientBoosting model "
        "trained on OFLC historical data (25,480 applications). "
        "Model and feature schema loaded exclusively from MLflow Model Registry."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SCHEMA — Pydantic v2
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
        "message": "Visa Approval Prediction API  (V5 — Registry-Only)",
        "status":  "running",
        "version": "3.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Status"])
def health():
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
    try:
        input_df = pd.DataFrame([application.model_dump()])
        encoded  = pd.get_dummies(input_df, drop_first=True)

        for col in feature_names:
            if col not in encoded.columns:
                encoded[col] = 0

        encoded = encoded[feature_names]
        pred    = model.predict(encoded)[0]
        proba   = model.predict_proba(encoded)[0]

        return {
            "prediction":            "Certified" if pred == 1 else "Denied",
            "probability_certified": round(float(proba[1]), 4),
            "probability_denied":    round(float(proba[0]), 4),
            "confidence":            round(float(max(proba)), 4),
            "model_version":         MODEL_VERSION,
        }

    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Feature mismatch — column '{e}' not found. "
                "Preprocessing in api.py must match train.py exactly."
            )
        )
    except Exception as e:
        log.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_type":    type(model).__name__,
        "model_source":  MODEL_SOURCE,
        "model_version": MODEL_VERSION,
        "run_id":        RUN_ID,
        "tracking_uri":  TRACKING_URI,
        "model_uri":     MODEL_URI,
        "n_features":    len(feature_names),
        "features":      feature_names,
        "api_version":   "3.0.0",
    }