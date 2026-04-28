"""
easyvisa_retrain_dag.py — EasyVisa Scheduled Retraining DAG (V11)

SCHEDULE: Every Monday at 02:00 UTC
(configurable via RETRAIN_SCHEDULE env var)

TASKS:
  1. validate_data    — Check EasyVisa.csv exists and has enough rows
  2. run_training     — Execute train.py via BashOperator
  3. verify_model     — Query MLflow REST API to confirm Production version exists
  4. restart_api      — kubectl rollout restart visa-api-deployment
  5. notify_complete  — Log completion summary
"""

import logging
import os
import requests
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME   = "easyvisa_gbm"
DATA_PATH    = os.getenv("DATA_PATH", "/opt/airflow/data/EasyVisa.csv")
MIN_ROWS     = int(os.getenv("MIN_ROWS", "10000"))
NAMESPACE    = os.getenv("K8S_NAMESPACE", "easyvisa")
SCHEDULE     = os.getenv("RETRAIN_SCHEDULE", "0 2 * * 1")
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")

DEFAULT_ARGS = {
    "owner":            "easyvisa-mlops",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry":   False,
}

# ── Task 1: Validate data ─────────────────────────────────────────────────────
def validate_data(**context):
    import pandas as pd

    log.info("Validating training data at: %s", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    row_count = len(df)
    log.info("Row count: %d (minimum required: %d)", row_count, MIN_ROWS)

    if row_count < MIN_ROWS:
        raise ValueError(f"Only {row_count} rows — minimum is {MIN_ROWS}. Aborting.")

    log.info("✅ Data validation passed: %d rows, %d columns", row_count, len(df.columns))
    context["ti"].xcom_push(key="row_count", value=row_count)


# ── Task 3: Verify model ──────────────────────────────────────────────────────
def verify_model(**context):
    """
    Query MLflow REST API to confirm a Production model version exists.
    Uses requests directly — no mlflow client needed in Airflow.
    """
    url = f"{MLFLOW_URI}/api/2.0/mlflow/registered-models/get-latest-versions"

    log.info("Querying MLflow at: %s", url)
    log.info("Model name: %s | Looking for stage: Production", MODEL_NAME)

    try:
        resp = requests.get(
            url,
            params={"name": MODEL_NAME, "stages": ["Production"]},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot reach MLflow at {MLFLOW_URI}: {e}") from e

    data = resp.json()
    versions = data.get("model_versions", [])

    log.info("MLflow response: %s", data)

    if not versions:
        # MLflow 3.x may not use stages — check all versions instead
        log.warning("No Production stage version found. Checking all versions...")
        url2 = f"{MLFLOW_URI}/api/2.0/mlflow/registered-models/get-latest-versions"
        resp2 = requests.get(url2, params={"name": MODEL_NAME}, timeout=30)
        resp2.raise_for_status()
        versions = resp2.json().get("model_versions", [])

        if not versions:
            raise RuntimeError(
                f"No versions found at all for '{MODEL_NAME}' in MLflow. "
                "Check that train.py completed and registered the model."
            )

        log.info("Found %d version(s) (any stage):", len(versions))
        for v in versions:
            log.info("  v%s | stage: %s | status: %s | run: %s",
                     v["version"], v.get("current_stage"), v.get("status"), v["run_id"][:8])

        # Use most recent version regardless of stage
        latest = sorted(versions, key=lambda x: int(x["version"]), reverse=True)[0]
        version_num = latest["version"]
        run_id      = latest["run_id"]
        stage       = latest.get("current_stage", "None")
        log.info("✅ Using latest version: v%s | stage: %s | run: %s", version_num, stage, run_id)

    else:
        v           = versions[0]
        version_num = v["version"]
        run_id      = v["run_id"]
        stage       = v.get("current_stage", "Production")
        log.info("✅ Production model verified: v%s | run: %s", version_num, run_id)

    context["ti"].xcom_push(key="model_version", value=version_num)
    context["ti"].xcom_push(key="run_id",         value=run_id)


# ── Task 5: Notify ────────────────────────────────────────────────────────────
def notify_complete(**context):
    ti            = context["ti"]
    row_count     = ti.xcom_pull(key="row_count",     task_ids="validate_data")
    model_version = ti.xcom_pull(key="model_version", task_ids="verify_model")
    run_id        = ti.xcom_pull(key="run_id",        task_ids="verify_model")
    dag_run_id    = context["dag_run"].run_id

    log.info("=" * 60)
    log.info("  EASYVISA RETRAINING COMPLETE")
    log.info("=" * 60)
    log.info("  DAG Run       : %s", dag_run_id)
    log.info("  Training rows : %s", row_count)
    log.info("  Model         : %s v%s", MODEL_NAME, model_version)
    log.info("  MLflow run    : %s", run_id)
    log.info("  MLflow UI     : %s/#/models/%s", MLFLOW_URI, MODEL_NAME)
    log.info("=" * 60)


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="easyvisa_retrain",
    description="Weekly scheduled retraining of EasyVisa GBM model",
    schedule_interval=SCHEDULE,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["easyvisa", "retraining", "mlops"],
    doc_md=__doc__,
) as dag:

    t1_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        doc_md="Check EasyVisa.csv exists and has sufficient rows.",
    )

    t2_train = BashOperator(
        task_id="run_training",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"python src/train.py "
            f"--data-path {DATA_PATH}"
            f"--no-tune || true "
        ),
        execution_timeout=timedelta(minutes=30),
        doc_md="Run train.py: preprocess → SMOTE → GBM → MLflow log → register.",
    )

    t3_verify = PythonOperator(
        task_id="verify_model",
        python_callable=verify_model,
        doc_md="Query MLflow REST API to confirm model version exists.",
    )

    t4_restart = BashOperator(
        task_id="restart_api",
        bash_command=(
            f"kubectl rollout restart deployment/visa-api-deployment -n {NAMESPACE} && "
            f"kubectl rollout status  deployment/visa-api-deployment -n {NAMESPACE} --timeout=120s"
        ),
        doc_md="Rolling restart of visa-api so pods load the new model.",
    )

    t5_notify = PythonOperator(
        task_id="notify_complete",
        python_callable=notify_complete,
        doc_md="Log completion summary.",
    )

    t1_validate >> t2_train >> t3_verify >> t4_restart >> t5_notify