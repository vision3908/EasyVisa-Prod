"""
api.py — EasyVisa Visa Approval Prediction API (V6 — S3 Artifact Store)
FastAPI serving layer — loads model AND feature_names exclusively from MLflow (S3 backend)

REQUIRED ENV VARS (all five must be set before starting):
  $env:MLFLOW_TRACKING_URI    = "http://localhost:5000"
  $env:MLFLOW_MODEL_URI       = "models:/easyvisa_gbm/Production"
  $env:AWS_ACCESS_KEY_ID      = "AKIASOG3HEM2K52VFYNM"
  $env:AWS_SECRET_ACCESS_KEY  = "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"
  $env:AWS_DEFAULT_REGION     = "us-east-1"
"""

import logging, os, sys, tempfile
import joblib, mlflow, mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def require_env_vars() -> tuple:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    model_uri    = os.getenv("MLFLOW_MODEL_URI", "").strip()
    aws_key      = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    aws_secret   = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    aws_region   = os.getenv("AWS_DEFAULT_REGION", "").strip()

    errors = []
    if not tracking_uri:
        errors.append("  MLFLOW_TRACKING_URI not set.\n  Fix: $env:MLFLOW_TRACKING_URI = 'http://localhost:5000'")
    if not model_uri:
        errors.append("  MLFLOW_MODEL_URI not set.\n  Fix: $env:MLFLOW_MODEL_URI = 'models:/easyvisa_gbm/Production'")
    if not aws_key:
        errors.append("  AWS_ACCESS_KEY_ID not set.")
    if not aws_secret:
        errors.append("  AWS_SECRET_ACCESS_KEY not set.")
    if not aws_region:
        errors.append("  AWS_DEFAULT_REGION not set.")

    if errors:
        log.error("=" * 65)
        log.error("API STARTUP FAILED — required environment variables missing:")
        for e in errors:
            log.error(e)
        log.error("=" * 65)
        sys.exit(1)

    log.info("MLFLOW_TRACKING_URI : %s", tracking_uri)
    log.info("MLFLOW_MODEL_URI    : %s", model_uri)
    log.info("AWS_DEFAULT_REGION  : %s", aws_region)
    log.info("AWS_ACCESS_KEY_ID   : %s***", aws_key[:6])
    return tracking_uri, model_uri


def resolve_run_id(client: mlflow.tracking.MlflowClient, model_uri: str) -> tuple:
    """
    Resolve models:/easyvisa_gbm/Production → actual run_id + version.
    REQUIRED because MLflow cannot append file paths to a models:/ stage URI.
    runs:/<run_id>/model/feature_names.pkl is the correct download path.
    """
    uri_parts = model_uri.replace("models:/", "").split("/")
    reg_name  = uri_parts[0]
    stage_ver = uri_parts[1] if len(uri_parts) > 1 else "Production"

    # Try by stage name (Production / Staging)
    try:
        versions = client.get_latest_versions(reg_name, stages=[stage_ver])
        if versions:
            return versions[0].run_id, versions[0].version
    except Exception:
        pass

    # Try by version number
    try:
        v = client.get_model_version(reg_name, stage_ver)
        return v.run_id, v.version
    except Exception:
        pass

    # Fallback — latest registered version
    all_versions = client.search_model_versions(f"name='{reg_name}'")
    if not all_versions:
        raise ValueError(f"No versions found for '{reg_name}'. Run train.py first.")
    latest = sorted(all_versions, key=lambda x: int(x.version))[-1]
    return latest.run_id, latest.version


def load_from_registry(tracking_uri: str, model_uri: str) -> tuple:
    mlflow.set_tracking_uri(tracking_uri)
    log.info("Connecting to MLflow: %s", tracking_uri)

    # Step 1: Load model
    log.info("Loading model from registry: %s", model_uri)
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        log.error("FAILED to load model: %s", e)
        log.error("Causes: MLflow not running | model not registered | no Production version | bad AWS creds")
        sys.exit(1)
    log.info("✅ Model loaded: %s", type(model).__name__)

    # Step 2: Resolve run_id FIRST — must happen before feature_names download
    client = mlflow.tracking.MlflowClient()
    try:
        run_id, model_version = resolve_run_id(client, model_uri)
        log.info("Resolved run_id: %s | version: %s", run_id[:12], model_version)
    except Exception as e:
        log.error("FAILED to resolve run_id: %s", e)
        sys.exit(1)

    # Step 3: Download feature_names.pkl using runs:/ URI (NOT models:/)
    feature_artifact_uri = f"runs:/{run_id}/model/feature_names.pkl"
    log.info("Downloading feature_names from S3: %s", feature_artifact_uri)
    try:
        tmp_dir    = tempfile.mkdtemp(prefix="easyvisa_features_")
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=feature_artifact_uri,
            dst_path=tmp_dir,
        )
        feature_names = joblib.load(local_path)
    except Exception as e:
        log.error("FAILED to download feature_names.pkl: %s", e)
        log.error("artifact_uri: %s", feature_artifact_uri)
        log.error("Fix: re-run train.py — python src/train.py --data-path data/EasyVisa.csv")
        sys.exit(1)

    log.info("✅ feature_names loaded: %d features", len(feature_names))

    model_source = (
        f"MLflow Registry | {model_uri} | "
        f"version={model_version} | run_id={run_id[:8]}... | artifact_store=S3"
    )
    log.info("✅ %s", model_source)
    return model, feature_names, model_source, run_id, str(model_version)


# ── Startup ───────────────────────────────────────────────────────────────────
TRACKING_URI, MODEL_URI = require_env_vars()
model, feature_names, MODEL_SOURCE, RUN_ID, MODEL_VERSION = load_from_registry(
    TRACKING_URI, MODEL_URI
)

log.info("=" * 65)
log.info("✅ V6 API STARTUP COMPLETE")
log.info("   Model type    : %s", type(model).__name__)
log.info("   Model version : %s", MODEL_VERSION)
log.info("   Features      : %d", len(feature_names))
log.info("   Source        : %s", MODEL_SOURCE)
log.info("   Artifact store: S3 (via MLflow)")
log.info("   Docs          : http://localhost:8000/docs")
log.info("=" * 65)


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visa Approval Prediction API (V6 — S3 Artifact Store)",
    description=(
        "Predicts US visa approval using GradientBoosting on OFLC data. "
        "Model loaded from MLflow Registry backed by S3."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


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


@app.get("/", tags=["Status"])
def home():
    return {
        "message": "Visa Approval Prediction API (V6 — S3 Artifact Store)",
        "status":  "running",
        "version": "4.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Status"])
def health():
    return {
        "status":         "healthy",
        "model_source":   MODEL_SOURCE,
        "model_type":     type(model).__name__,
        "model_version":  MODEL_VERSION,
        "run_id":         RUN_ID,
        "n_features":     len(feature_names),
        "tracking_uri":   TRACKING_URI,
        "artifact_store": "S3",
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
            detail=f"Feature mismatch — column '{e}' not found. Preprocessing must match train.py."
        )
    except Exception as e:
        log.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_type":     type(model).__name__,
        "model_source":   MODEL_SOURCE,
        "model_version":  MODEL_VERSION,
        "run_id":         RUN_ID,
        "tracking_uri":   TRACKING_URI,
        "model_uri":      MODEL_URI,
        "n_features":     len(feature_names),
        "features":       feature_names,
        "api_version":    "4.0.0",
        "artifact_store": "S3",
    }