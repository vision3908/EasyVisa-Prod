"""
api.py — EasyVisa Visa Approval Prediction API  (V10 — EKS + Cloud Observability)

WHAT THIS DOES:
  - Loads model AND feature_names from MLflow Model Registry (S3 backend) at startup
  - V9: Logs every prediction input to monitoring/inference_log.csv for Evidently drift
  - V10 FIX 1: feature_artifact_uri resolved dynamically via MlflowClient.get_latest_versions()
               → uses runs:/<run_id>/model/feature_names.pkl (correct S3 subfolder, no hardcoding)
  - V10 FIX 2: MLFLOW_TRACKING_URI defaults to http://mlflow:5000 (Docker Compose service name)
  - Fails fast at startup if any required env var is missing — no silent bad state

REQUIRED ENV VARS (all five must be set before starting):
  MLFLOW_TRACKING_URI   = "http://mlflow:5000"       ← Docker Compose service (V10)
  MLFLOW_MODEL_URI      = "models:/easyvisa_gbm/Production"
  AWS_ACCESS_KEY_ID     = "AKIA..."
  AWS_SECRET_ACCESS_KEY = "..."
  AWS_DEFAULT_REGION    = "us-east-1"

DOCKER COMPOSE (V10 — 5 services):
  docker compose up -d
  # visa-api connects to http://mlflow:5000 automatically via Docker network

LOCAL DEV (outside Docker):
  $env:MLFLOW_TRACKING_URI   = "http://localhost:5000"
  $env:MLFLOW_MODEL_URI      = "models:/easyvisa_gbm/Production"
  $env:AWS_ACCESS_KEY_ID     = "AKIA..."
  $env:AWS_SECRET_ACCESS_KEY = "..."
  $env:AWS_DEFAULT_REGION    = "us-east-1"
  python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

ENDPOINTS:
  GET  /           -> root health check (API alive)
  GET  /health     -> detailed health: model version, run_id, feature count
  POST /predict    -> main prediction endpoint (V9+: also logs input to CSV)
  GET  /model-info -> full model metadata for auditing
  GET  /metrics    -> Prometheus metrics scrape endpoint
"""

# --- Standard library --------------------------------------------------------
import csv        # Write inference inputs to CSV for Evidently drift detection
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
# PROMETHEUS METRICS — defined at module level (created once at startup)
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
# STARTUP VALIDATION — all five env vars required, no defaults
# -----------------------------------------------------------------------------
def require_env_vars() -> tuple[str, str]:
    """
    Read all five required environment variables from the environment.
    Exit immediately with clear fix instructions if any are missing.

    Required vars:
      MLFLOW_TRACKING_URI   : MLflow server address
      MLFLOW_MODEL_URI      : Registry URI for the model to load
      AWS_ACCESS_KEY_ID     : IAM credentials for S3 artifact download
      AWS_SECRET_ACCESS_KEY : IAM credentials for S3 artifact download
      AWS_DEFAULT_REGION    : S3 bucket region

    Returns:
      tracking_uri (str) -- e.g. "http://mlflow:5000"
      model_uri    (str) -- e.g. "models:/easyvisa_gbm/Production"
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
            "  Fix: $env:MLFLOW_TRACKING_URI = 'http://mlflow:5000'  (Docker Compose)\n"
            "       $env:MLFLOW_TRACKING_URI = 'http://localhost:5000'  (local dev)"
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
        log.error("API STARTUP FAILED -- required environment variables missing:")
        log.error("")
        for e in errors:
            log.error(e)
        log.error("")
        log.error("Docker Compose startup (V10 — 5 services):")
        log.error("  docker compose up -d")
        log.error("")
        log.error("Local dev startup sequence:")
        log.error("  1. python -m mlflow server \\")
        log.error("         --backend-store-uri sqlite:///mlflow.db \\")
        log.error("         --default-artifact-root s3://easyvisa-mlflow-vision-2025/mlflow-artifacts \\")
        log.error("         --host 0.0.0.0 --port 5000")
        log.error("  2. $env:MLFLOW_TRACKING_URI   = 'http://localhost:5000'")
        log.error("  3. $env:MLFLOW_MODEL_URI       = 'models:/easyvisa_gbm/Production'")
        log.error("  4. $env:AWS_ACCESS_KEY_ID      = 'AKIA...'")
        log.error("  5. $env:AWS_SECRET_ACCESS_KEY  = '...'")
        log.error("  6. $env:AWS_DEFAULT_REGION     = 'us-east-1'")
        log.error("  7. python -m uvicorn src.api:app --host 0.0.0.0 --port 8000")
        log.error("=" * 65)
        sys.exit(1)

    log.info("MLFLOW_TRACKING_URI : %s", tracking_uri)
    log.info("MLFLOW_MODEL_URI    : %s", model_uri)
    log.info("AWS_DEFAULT_REGION  : %s", aws_region)
    log.info("AWS_ACCESS_KEY_ID   : %s***", aws_key[:6])  # Partial -- never log full key
    return tracking_uri, model_uri


# -----------------------------------------------------------------------------
# MODEL LOADER — downloads model + feature_names from MLflow (S3 backend)
# -----------------------------------------------------------------------------
def load_from_registry(tracking_uri: str, model_uri: str) -> tuple:
    """
    Load the sklearn model AND feature_names exclusively from MLflow registry.
    No local .pkl fallback. If loading fails, the process exits immediately.

    HOW THE MODEL IS RETRIEVED:
      mlflow.sklearn.load_model(model_uri) resolves "models:/easyvisa_gbm/Production"
      to the actual run in the registry, downloads the model artifact from S3,
      and deserializes it into a fitted sklearn estimator object.

    HOW feature_names IS RETRIEVED (V10 FIX — no hardcoding):
      1. MlflowClient.get_latest_versions() resolves the Production stage → run_id
      2. Build artifact URI as runs:/<run_id>/model/feature_names.pkl
      3. mlflow.artifacts.download_artifacts() fetches it from S3

      WHY runs:/ INSTEAD OF models:/:
        models:/easyvisa_gbm/Production/model/feature_names.pkl resolves to a
        different S3 sub-path than where train.py actually wrote the file.
        runs:/<run_id>/model/feature_names.pkl resolves to the exact S3 location.
        The run_id is resolved dynamically at startup — never hardcoded.

    WHY feature_names IS CRITICAL:
      pd.get_dummies() produces different columns depending on which categorical
      values appear in the data. feature_names.pkl is the contract that ensures
      api.py builds the exact same column set and order as train.py did.

    Returns:
      model         : fitted sklearn model (GradientBoostingClassifier, etc.)
      feature_names : list[str] -- ordered column names from training
      model_source  : str -- human-readable description for /health endpoint
      run_id        : str -- MLflow run ID for auditability
      model_version : str -- registry version number for auditability
    """
    mlflow.set_tracking_uri(tracking_uri)
    log.info("Connecting to MLflow: %s", tracking_uri)
    log.info("Loading model from registry: %s", model_uri)

    # ── Step 1: Load sklearn model ────────────────────────────────────────────
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        log.error("=" * 65)
        log.error("FAILED to load model from MLflow registry.")
        log.error("  model_uri : %s", model_uri)
        log.error("  error     : %s", e)
        log.error("")
        log.error("Possible causes:")
        log.error("  1. MLflow service not healthy yet — wait 30s and retry")
        log.error("     (visa-api depends_on mlflow with service_healthy condition)")
        log.error("  2. Model '%s' not registered yet.", model_uri)
        log.error("     Fix: python src/train.py --data-path data/EasyVisa.csv")
        log.error("  3. No version in 'Production' stage.")
        log.error("     Fix: promote a version in the MLflow UI -> Models tab")
        log.error("  4. AWS credentials missing or invalid (S3 access denied).")
        log.error("     Fix: verify AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        log.error("=" * 65)
        sys.exit(1)

    log.info("✅ Model loaded: %s", type(model).__name__)

    # ── Step 2: Resolve run_id dynamically via MlflowClient (V10 FIX) ────────
    #
    # WHY: models:/ URI resolves to a different S3 sub-path than where
    #      train.py actually stored feature_names.pkl. The runs:/ URI hits
    #      the exact S3 location. We resolve run_id from the registry at
    #      startup — zero hardcoding, auto-updates on every new promotion.
    #
    try:
        client       = mlflow.tracking.MlflowClient()
        uri_parts    = model_uri.replace("models:/", "").split("/")
        model_name   = uri_parts[0]            # e.g. "easyvisa_gbm"
        stage        = uri_parts[1] if len(uri_parts) > 1 else "Production"

        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            log.error("No model version found in stage '%s' for model '%s'.", stage, model_name)
            log.error("Fix: promote a version in the MLflow UI -> Models tab -> set stage to %s", stage)
            sys.exit(1)

        run_id    = versions[0].run_id
        model_ver = versions[0].version
        log.info("Resolved run_id: %s  (version: %s)", run_id, model_ver)

    except SystemExit:
        raise
    except Exception as e:
        log.error("Failed to resolve run_id from registry: %s", e)
        sys.exit(1)

    # ── Step 3: Download feature_names.pkl using resolved runs:/ URI ─────────
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
        log.error("=" * 65)
        log.error("FAILED to download feature_names.pkl from S3.")
        log.error("  artifact_uri : %s", feature_artifact_uri)
        log.error("  run_id       : %s", run_id)
        log.error("  error        : %s", e)
        log.error("")
        log.error("Possible causes:")
        log.error("  1. train.py did not log feature_names.pkl as an artifact")
        log.error("     Fix: re-run python src/train.py --data-path data/EasyVisa.csv")
        log.error("  2. AWS credentials invalid — S3 read access denied")
        log.error("     Fix: verify AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env")
        log.error("  3. The artifact path inside the run is not model/feature_names.pkl")
        log.error("     Fix: check MLflow UI -> Experiments -> run -> Artifacts tab")
        log.error("=" * 65)
        sys.exit(1)

    log.info("✅ Feature names loaded: %d features", len(feature_names))

    model_source = (
        f"MLflow Registry | {model_uri} | "
        f"version={model_ver} | run_id={run_id[:8]}..."
    )
    log.info("Model source: %s", model_source)

    return model, feature_names, model_source, run_id, str(model_ver)


# -----------------------------------------------------------------------------
# STARTUP — runs once when uvicorn imports this module
# -----------------------------------------------------------------------------
TRACKING_URI, MODEL_URI = require_env_vars()

model, feature_names, MODEL_SOURCE, RUN_ID, MODEL_VERSION = load_from_registry(
    TRACKING_URI, MODEL_URI
)

log.info("=" * 65)
log.info("✅ V10 API STARTUP COMPLETE")
log.info("   Model type    : %s", type(model).__name__)
log.info("   Model version : %s", MODEL_VERSION)
log.info("   Run ID        : %s", RUN_ID)
log.info("   Features      : %d", len(feature_names))
log.info("   Source        : %s", MODEL_SOURCE)
log.info("   Artifact store: S3 (via MLflow)")
log.info("   Inference log : %s  (written on every /predict call)", INFERENCE_LOG_PATH)
log.info("   Docs          : http://localhost:8000/docs")
log.info("=" * 65)


# -----------------------------------------------------------------------------
# FASTAPI APPLICATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Visa Approval Prediction API  (V10 — EKS + Cloud Observability)",
    description=(
        "Predicts US visa approval probability using a GradientBoosting model "
        "trained on OFLC historical data (25,480 applications). "
        "Model and feature schema loaded exclusively from MLflow Model Registry (S3 backend). "
        "V9+: Every prediction input logged to CSV for live Evidently drift detection. "
        "V10: Dynamic run_id resolution — no hardcoded IDs anywhere."
    ),
    version="10.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# -----------------------------------------------------------------------------
# INPUT SCHEMA — Pydantic v2 BaseModel
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
# V9+: INFERENCE LOGGING HELPER
# -----------------------------------------------------------------------------
def log_inference_input(application: VisaApplication) -> None:
    """
    Append the raw prediction input to monitoring/inference_log.csv.

    WHY THIS EXISTS (V9+):
      The Evidently drift service reads this CSV to compare live inference
      inputs against the training baseline and detect data drift.

    DESIGN:
      - Non-fatal: wrapped in try/except so logging failure never breaks predictions
      - Appends: never overwrites -- preserves full history for drift analysis
      - Header: written only on first row (when file does not exist yet)

    Docker Compose:
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
    """Root endpoint -- confirms the API is alive."""
    return {
        "message": "Visa Approval Prediction API",
        "status":  "running",
        "version": "10.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """
    Detailed health check -- used by Docker HEALTHCHECK and Kubernetes probes.
    Returns model version, run_id, and artifact source.
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
        "api_version":   "10.0.0",
    }


@app.post("/predict", tags=["Prediction"])
def predict_visa(application: VisaApplication):
    """
    Main prediction endpoint.

    Preprocessing pipeline (must match train.py exactly):
      Step 0: Log raw input to CSV for Evidently drift detection (V9+ -- non-fatal)
      Step 1: Convert Pydantic model -> single-row DataFrame
      Step 2: pd.get_dummies(drop_first=True) -- same as train.py
      Step 3: Add missing columns (categories absent in this row -> fill with 0)
      Step 4: Reorder to exactly match training column order (feature_names)
      Step 5: model.predict() + model.predict_proba()

    Returns:
      prediction            : "Certified" or "Denied"
      probability_certified : float -- P(Certified), e.g. 0.9234
      probability_denied    : float -- P(Denied), e.g. 0.0766
      confidence            : float -- max(proba), e.g. 0.9234
      model_version         : str   -- the registry version that made this prediction
    """
    start_time = time.time()

    # Step 0 (V9+): Log raw input for Evidently drift detection
    # Non-fatal -- a logging failure never blocks the prediction response
    log_inference_input(application)

    try:
        # Step 1: Convert request to DataFrame
        input_df = pd.DataFrame([application.model_dump()])

        # Step 2: One-hot encode -- MUST use same settings as train.py
        # drop_first=True: N categories -> N-1 dummies (avoids multicollinearity)
        encoded = pd.get_dummies(input_df, drop_first=True)

        # Step 3: Add columns that exist in training but not in this single row
        # Example: training saw "continent_Oceania" but this request is from "Asia"
        # get_dummies won't create the Oceania column -- fill it with 0
        for col in feature_names:
            if col not in encoded.columns:
                encoded[col] = 0

        # Step 4: Reorder to exactly match training column order
        # sklearn is order-sensitive -- wrong order = wrong predictions (no error thrown)
        encoded = encoded[feature_names]

        # Step 5: Run inference
        pred  = model.predict(encoded)[0]
        proba = model.predict_proba(encoded)[0]
        # proba[0] = P(Denied), proba[1] = P(Certified)

        result = {
            "prediction":            "Certified" if pred == 1 else "Denied",
            "probability_certified": round(float(proba[1]), 4),
            "probability_denied":    round(float(proba[0]), 4),
            "confidence":            round(float(max(proba)), 4),
            "model_version":         MODEL_VERSION,
        }

        # Track Prometheus metrics
        prediction_counter.labels(prediction=result["prediction"]).inc()
        api_requests.labels(endpoint="/predict", status="success").inc()

        return result

    except KeyError as e:
        api_requests.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(
            status_code=422,
            detail=(
                f"Feature mismatch -- column '{e}' not found. "
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
        # Always record latency -- even on errors
        prediction_latency.observe(time.time() - start_time)


@app.get("/model-info", tags=["Model"])
def model_info():
    """Full model metadata -- for auditing, CI/CD checks, and debugging."""
    return {
        "model_type":    type(model).__name__,
        "model_source":  MODEL_SOURCE,
        "model_version": MODEL_VERSION,
        "run_id":        RUN_ID,
        "tracking_uri":  TRACKING_URI,
        "model_uri":     MODEL_URI,
        "n_features":    len(feature_names),
        "features":      feature_names,
        "api_version":   "10.0.0",
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Prometheus metrics scrape endpoint.

    Exposes:
      predictions_total{prediction="Certified"|"Denied"} -- count by label
      prediction_latency_seconds (histogram)              -- response time distribution
      api_requests_total{endpoint, status}               -- request counts by endpoint + status
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)