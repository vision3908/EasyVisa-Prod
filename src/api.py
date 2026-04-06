"""
api.py â€” EasyVisa Visa Approval Prediction API  (V9 â€” S3 Artifact Store + Inference Logging)
FastAPI serving layer â€” loads model AND feature_names exclusively from MLflow (S3 backend)

WHAT THIS DOES:
  - Loads the registered sklearn model from MLflow Model Registry (S3 backend) at startup
  - Downloads feature_names.pkl from S3 via MLflow to enforce preprocessing consistency
  - Serves predictions via POST /predict with full Pydantic request validation
  - Exposes /health (for Kubernetes probes) and /metrics (for Prometheus scraping)
  - V9 NEW: Logs every raw prediction input to monitoring/inference_log.csv
    so the Evidently drift service can compute live data drift metrics
  - Fails fast at startup if any required env var is missing â€” no silent bad state

REQUIRED ENV VARS (all five must be set before starting):
  $env:MLFLOW_TRACKING_URI   = "http://localhost:5000"
  $env:MLFLOW_MODEL_URI      = "models:/easyvisa_gbm/Production"
  $env:AWS_ACCESS_KEY_ID     = "AKIA..."
  $env:AWS_SECRET_ACCESS_KEY = "..."
  $env:AWS_DEFAULT_REGION    = "us-east-1"

STARTUP SEQUENCE:
  # Window 1 â€” MLflow server with S3 backend (keep running):
  python -m mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root s3://easyvisa-mlflow-vision-2025/mlflow-artifacts \
      --host 0.0.0.0 --port 5000

  # Window 2 â€” start API:
  $env:MLFLOW_TRACKING_URI   = "http://localhost:5000"
  $env:MLFLOW_MODEL_URI      = "models:/easyvisa_gbm/Production"
  $env:AWS_ACCESS_KEY_ID     = "AKIA..."
  $env:AWS_SECRET_ACCESS_KEY = "..."
  $env:AWS_DEFAULT_REGION    = "us-east-1"
  python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

DOCKER RUN (use your actual Windows IPv4, not localhost):
  docker run -d -p 8000:8000 \
    -e MLFLOW_TRACKING_URI=http://192.168.12.157:5000 \
    -e MLFLOW_MODEL_URI=models:/easyvisa_gbm/Production \
    -e AWS_ACCESS_KEY_ID=AKIA... \
    -e AWS_SECRET_ACCESS_KEY=... \
    -e AWS_DEFAULT_REGION=us-east-1 \
    -v $(pwd)/monitoring:/app/monitoring \
    --name visa-api-v9 visa-api:v9

ENDPOINTS:
  GET  /           -> root health check (API alive)
  GET  /health     -> detailed health: model version, run_id, feature count
  POST /predict    -> main prediction endpoint (V9: also logs input to CSV)
  GET  /model-info -> full model metadata for auditing
  GET  /metrics    -> Prometheus metrics scrape endpoint
"""

# --- Standard library --------------------------------------------------------
import csv        # V9: Write inference inputs to CSV for Evidently drift detection
import logging    # Structured timestamped logs
import os         # Read environment variables + check file existence
import sys        # sys.exit() for fail-fast startup
import tempfile   # Temp directory for artifact download
import time       # Latency tracking for Prometheus histogram
from pathlib import Path  # Cross-platform path handling for log directory creation

# --- Third-party -------------------------------------------------------------
import joblib              # Deserialize feature_names.pkl downloaded from MLflow
import mlflow              # MLflow client for registry and artifact access
import mlflow.sklearn      # Load sklearn models from registry
import pandas as pd        # DataFrame for preprocessing (must match train.py exactly)
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

# --- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# V9: INFERENCE LOG PATH
# Every prediction input is appended here for Evidently drift detection.
# In Docker Compose, visa-api and evidently-service share ./monitoring as a
# volume so evidently_service.py can read what api.py writes here.
# -----------------------------------------------------------------------------
INFERENCE_LOG_PATH = "monitoring/inference_log.csv"


# -----------------------------------------------------------------------------
# PROMETHEUS METRICS â€” defined at module level (created once at startup)
# -----------------------------------------------------------------------------
prediction_counter = Counter(
    "predictions_total",
    "Total predictions made",
    ["prediction"],           # Label: "Certified" or "Denied"
)
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Time to generate a prediction (seconds)",
)
api_requests = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "status"],   # Labels: endpoint path + "success"/"error"
)


# -----------------------------------------------------------------------------
# STARTUP VALIDATION â€” all five env vars required, no defaults
# -----------------------------------------------------------------------------
def require_env_vars() -> tuple[str, str]:
    """
    Read all five required environment variables from the environment.
    Exit immediately with clear fix instructions if any are missing.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    model_uri    = os.getenv("MLFLOW_MODEL_URI",    "").strip()
    aws_key      = os.getenv("AWS_ACCESS_KEY_ID",   "").strip()
    aws_secret   = os.getenv("AWS_SECRET_ACCESS_KEY","").strip()
    aws_region   = os.getenv("AWS_DEFAULT_REGION",  "").strip()

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
    if not aws_key:
        errors.append(
            "  AWS_ACCESS_KEY_ID is not set.\n"
            "  Fix: $env:AWS_ACCESS_KEY_ID = 'AKIA...'"
        )
    if not aws_secret:
        errors.append(
            "  AWS_SECRET_ACCESS_KEY is not set.\n"
            "  Fix: $env:AWS_SECRET_ACCESS_KEY = '...'"
        )
    if not aws_region:
        errors.append(
            "  AWS_DEFAULT_REGION is not set.\n"
            "  Fix: $env:AWS_DEFAULT_REGION = 'us-east-1'"
        )

    if errors:
        log.error("=" * 65)
        log.error("API STARTUP FAILED â€” required environment variables missing:")
        log.error("")
        for e in errors:
            log.error(e)
        log.error("")
        log.error("=" * 65)
        sys.exit(1)

    log.info("MLFLOW_TRACKING_URI : %s", tracking_uri)
    log.info("MLFLOW_MODEL_URI    : %s", model_uri)
    log.info("AWS_DEFAULT_REGION  : %s", aws_region)
    log.info("AWS_ACCESS_KEY_ID   : %s***", aws_key[:6])
    return tracking_uri, model_uri


# -----------------------------------------------------------------------------
# MODEL LOADER â€” downloads model + feature_names from MLflow (S3 backend)
# -----------------------------------------------------------------------------
def load_from_registry(tracking_uri: str, model_uri: str) -> tuple:
    """
    Load the sklearn model AND feature_names exclusively from MLflow registry.
    No local .pkl fallback. If loading fails, the process exits immediately.
    """
    mlflow.set_tracking_uri(tracking_uri)
    log.info("Loading model from registry: %s", model_uri)

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        log.error("FAILED to load model from MLflow registry: %s", e)
        log.error("Possible causes:")
        log.error("  1. MLflow server not running at %s", tracking_uri)
        log.error("  2. Model '%s' not registered yet â€” run train.py", model_uri)
        log.error("  3. No version in 'Production' stage â€” promote in MLflow UI")
        sys.exit(1)

    log.info("âœ… Model loaded: %s", type(model).__name__)

    feature_artifact_uri = "runs:/f289aac02a2249119e4c3bf4a8e3ad7e/model/feature_names.pkl"
    try:
        tmp_dir = tempfile.mkdtemp(prefix="easyvisa_features_")
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=feature_artifact_uri,
            dst_path=tmp_dir,
        )
        feature_names = joblib.load(local_path)
    except Exception as e:
        log.error("FAILED to download feature_names.pkl: %s", e)
        log.error("  artifact_uri: %s", feature_artifact_uri)
        log.error("  Re-run train.py to create a new version with feature_names logged.")
        sys.exit(1)

    log.info("âœ… feature_names loaded: %d features", len(feature_names))

    # Retrieve run metadata for auditability
    try:
        client    = mlflow.tracking.MlflowClient()
        uri_parts = model_uri.replace("models:/", "").split("/")
        reg_name  = uri_parts[0]
        stage     = uri_parts[1] if len(uri_parts) > 1 else "latest"
        versions  = client.get_latest_versions(reg_name, stages=[stage])
        if versions:
            run_id, model_ver = versions[0].run_id, versions[0].version
        else:
            all_v  = client.search_model_versions(f"name='{reg_name}'")
            latest = sorted(all_v, key=lambda x: int(x.version))[-1]
            run_id, model_ver = latest.run_id, latest.version
    except Exception:
        run_id, model_ver = "unknown", "unknown"

    model_source = f"MLflow Registry | {model_uri} | version={model_ver} | run_id={run_id[:8]}..."
    log.info("âœ… Model source: %s", model_source)
    return model, feature_names, model_source, run_id, str(model_ver)


# -----------------------------------------------------------------------------
# STARTUP â€” runs once when uvicorn imports this module
# -----------------------------------------------------------------------------
TRACKING_URI, MODEL_URI = require_env_vars()
model, feature_names, MODEL_SOURCE, RUN_ID, MODEL_VERSION = load_from_registry(
    TRACKING_URI, MODEL_URI
)

log.info("=" * 65)
log.info("âœ… V9 API STARTUP COMPLETE")
log.info("   Model type    : %s", type(model).__name__)
log.info("   Model version : %s", MODEL_VERSION)
log.info("   Features      : %d", len(feature_names))
log.info("   Inference log : %s", INFERENCE_LOG_PATH)
log.info("   Artifact store: S3 (via MLflow)")
log.info("   Docs          : http://localhost:8000/docs")
log.info("=" * 65)


# -----------------------------------------------------------------------------
# FASTAPI APPLICATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Visa Approval Prediction API  (V9 â€” S3 Artifact Store + Inference Logging)",
    description=(
        "Predicts US visa approval probability using a GradientBoosting model "
        "trained on OFLC historical data (25,480 applications). "
        "Model and feature schema loaded exclusively from MLflow Model Registry (S3 backend). "
        "V9: Every prediction input logged to CSV for live Evidently drift detection."
    ),
    version="9.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# -----------------------------------------------------------------------------
# INPUT SCHEMA â€” Pydantic v2 BaseModel
# -----------------------------------------------------------------------------
class VisaApplication(BaseModel):
    continent:             str    # "Asia" | "Europe" | "Africa" | "North America" | "South America" | "Oceania"
    education_of_employee: str    # "High School" | "Bachelor's" | "Master's" | "Doctorate"
    has_job_experience:    str    # "Y" | "N"
    requires_job_training: str    # "Y" | "N"
    no_of_employees:       int    # Company headcount (positive integer)
    yr_of_estab:           int    # Year company was established (e.g. 2005)
    region_of_employment:  str    # "Northeast" | "South" | "Midwest" | "West" | "Island"
    prevailing_wage:       float  # Wage offered in USD (e.g. 85000.0)
    unit_of_wage:          str    # "Yearly" | "Monthly" | "Weekly" | "Hourly"
    full_time_position:    str    # "Y" | "N"

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


# -----------------------------------------------------------------------------
# V9: INFERENCE LOGGING HELPER
# -----------------------------------------------------------------------------
def log_inference_input(application: VisaApplication) -> None:
    """
    Append the raw prediction input to monitoring/inference_log.csv.

    Non-fatal: wrapped in try/except so logging failure never breaks predictions.
    Both visa-api and evidently-service mount ./monitoring as a shared volume,
    so evidently_service.py can read what api.py writes here.
    """
    try:
        log_dir = Path(INFERENCE_LOG_PATH).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        write_header = not os.path.exists(INFERENCE_LOG_PATH)

        with open(INFERENCE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=application.model_dump().keys())
            if write_header:
                writer.writeheader()
            writer.writerow(application.model_dump())

    except Exception as log_err:
        log.warning("Inference logging failed (non-fatal): %s", log_err)


# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Root endpoint â€” confirms the API is alive."""
    return {
        "message": "Visa Approval Prediction API",
        "status":  "running",
        "version": "9.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """
    Detailed health check â€” used by Docker HEALTHCHECK and Kubernetes probes.
    Returns model version, run_id, artifact source, and inference log path.
    """
    return {
        "status":        "healthy",
        "model_source":  MODEL_SOURCE,
        "model_type":    type(model).__name__,
        "model_version": MODEL_VERSION,
        "run_id":        RUN_ID,
        "n_features":    len(feature_names),
        "tracking_uri":  TRACKING_URI,
        "inference_log": INFERENCE_LOG_PATH,
    }


@app.post("/predict", tags=["Prediction"])
def predict_visa(application: VisaApplication):
    """
    Main prediction endpoint.

    Preprocessing pipeline (must match train.py exactly):
      Step 0: Log raw input to CSV for Evidently drift detection (V9 â€” non-fatal)
      Step 1: Convert Pydantic model -> single-row DataFrame
      Step 2: pd.get_dummies(drop_first=True) â€” same as train.py
      Step 3: Add missing columns (categories absent in this row -> fill with 0)
      Step 4: Reorder to exactly match training column order (feature_names)
      Step 5: model.predict() + model.predict_proba()

    Returns:
      prediction            : "Certified" or "Denied"
      probability_certified : float
      probability_denied    : float
      confidence            : float
      model_version         : str
    """
    start_time = time.time()

    try:
        # Step 0 (V9): Log raw input for drift detection â€” non-fatal
        log_inference_input(application)

        # Step 1: Convert request to DataFrame
        input_df = pd.DataFrame([application.model_dump()])

        # Step 2: One-hot encode â€” MUST use same settings as train.py
        encoded = pd.get_dummies(input_df, drop_first=True)

        # Step 3: Add columns that exist in training but not in this single row
        for col in feature_names:
            if col not in encoded.columns:
                encoded[col] = 0

        # Step 4: Reorder to exactly match training column order
        encoded = encoded[feature_names]

        # Step 5: Run inference
        pred  = model.predict(encoded)[0]
        proba = model.predict_proba(encoded)[0]

        result = {
            "prediction":            "Certified" if pred == 1 else "Denied",
            "probability_certified": round(float(proba[1]), 4),
            "probability_denied":    round(float(proba[0]), 4),
            "confidence":            round(float(max(proba)), 4),
            "model_version":         MODEL_VERSION,
        }

        prediction_counter.labels(prediction=result["prediction"]).inc()
        api_requests.labels(endpoint="/predict", status="success").inc()

        return result

    except KeyError as e:
        api_requests.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(
            status_code=422,
            detail=(
                f"Feature mismatch â€” column '{e}' not found. "
                "Preprocessing in api.py must match train.py exactly "
                "(both use pd.get_dummies with drop_first=True)."
            ),
        )
    except Exception as e:
        api_requests.labels(endpoint="/predict", status="error").inc()
        log.error("Prediction error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
    finally:
        prediction_latency.observe(time.time() - start_time)


@app.get("/model-info", tags=["Model"])
def model_info():
    """Full model metadata â€” for auditing, CI/CD checks, and debugging."""
    return {
        "model_type":    type(model).__name__,
        "model_source":  MODEL_SOURCE,
        "model_version": MODEL_VERSION,
        "run_id":        RUN_ID,
        "tracking_uri":  TRACKING_URI,
        "model_uri":     MODEL_URI,
        "n_features":    len(feature_names),
        "features":      feature_names,
        "api_version":   "9.0.0",
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Prometheus metrics scrape endpoint.

    Exposes:
      predictions_total{prediction="Certified"|"Denied"}
      prediction_latency_seconds (histogram)
      api_requests_total{endpoint, status}
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
