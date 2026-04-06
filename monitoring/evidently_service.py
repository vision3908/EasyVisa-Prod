
"""
evidently_service.py — Live Evidently drift metrics for Prometheus (V9)

What this does:
  - Runs as a persistent HTTP service on port 8085
  - Every 60 seconds: loads reference data + inference log, computes drift
  - Exposes drift metrics in Prometheus exposition format at /metrics
  - Prometheus scrapes this endpoint → Grafana visualizes drift over time

Architecture:
  api.py writes inputs → monitoring/inference_log.csv
  evidently_service.py reads log → computes drift → exposes /metrics
  Prometheus scrapes :8085/metrics → stores time-series
  Grafana queries Prometheus → shows drift dashboard

Run locally (outside Docker):
  python monitoring/evidently_service.py

Inside Docker Compose:
  Runs automatically as the 'evidently-service' container.
"""

import logging
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REFERENCE_DATA_PATH = "data/EasyVisa.csv"
INFERENCE_LOG_PATH  = "monitoring/inference_log.csv"
DRIFT_REPORT_PATH   = "monitoring/drift_report.html"
SERVICE_PORT        = 8085
REFRESH_INTERVAL    = 60   # seconds between drift recomputes

# ── Prometheus metrics for drift ──────────────────────────────────────────────
# Use a separate registry to avoid conflicts with the FastAPI app
drift_registry = CollectorRegistry()

dataset_drift_detected = Gauge(
    "evidently_dataset_drift_detected",
    "1 if dataset drift detected, 0 if not",
    registry=drift_registry,
)
dataset_drift_score = Gauge(
    "evidently_dataset_drift_score",
    "Share of drifted features (0.0 to 1.0)",
    registry=drift_registry,
)
drifted_features_count = Gauge(
    "evidently_drifted_features_count",
    "Number of features where drift was detected",
    registry=drift_registry,
)
inference_log_rows = Gauge(
    "evidently_inference_log_rows",
    "Number of rows in the inference log",
    registry=drift_registry,
)

# Per-feature drift gauges (created dynamically on first run)
feature_drift_gauges: dict = {}


def compute_drift():
    """
    Load reference + current data, run Evidently, update Prometheus gauges.
    Called every REFRESH_INTERVAL seconds by the background thread.
    """
    try:
        # Load reference (training baseline)
        reference = pd.read_csv(REFERENCE_DATA_PATH)
        # Drop non-feature columns
        for col in ["case_id", "case_status"]:
            if col in reference.columns:
                reference.drop(col, axis=1, inplace=True)

        # Load inference log (current data)
        try:
            current = pd.read_csv(INFERENCE_LOG_PATH)
        except FileNotFoundError:
            log.warning("Inference log not found yet — waiting for predictions")
            inference_log_rows.set(0)
            return

        if len(current) < 10:
            log.info("Only %d inference rows — need at least 10 for drift", len(current))
            inference_log_rows.set(len(current))
            return

        inference_log_rows.set(len(current))
        log.info("Computing drift: %d reference rows vs %d inference rows",
                 len(reference), len(current))

        # Align columns — use only columns present in both
        common_cols = list(set(reference.columns) & set(current.columns))
        reference = reference[common_cols]
        current   = current[common_cols]

        # Run Evidently drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            *[ColumnDriftMetric(column_name=col) for col in common_cols]
        ])
        report.run(reference_data=reference, current_data=current)

        # Save HTML report for manual inspection
        report.save_html(DRIFT_REPORT_PATH)

        # Extract results
        result = report.as_dict()
        metrics_list = result.get("metrics", [])

        for metric in metrics_list:
            metric_id = metric.get("metric", "")
            result_data = metric.get("result", {})

            if metric_id == "DatasetDriftMetric":
                drift_detected = int(result_data.get("dataset_drift", False))
                drift_share    = float(result_data.get("share_of_drifted_columns", 0.0))
                n_drifted      = int(result_data.get("number_of_drifted_columns", 0))

                dataset_drift_detected.set(drift_detected)
                dataset_drift_score.set(drift_share)
                drifted_features_count.set(n_drifted)

                log.info("Dataset drift: detected=%s | score=%.3f | drifted=%d/%d features",
                         bool(drift_detected), drift_share, n_drifted, len(common_cols))

            elif metric_id == "ColumnDriftMetric":
                col_name = result_data.get("column_name", "unknown")
                col_drift = int(result_data.get("drift_detected", False))
                col_score = float(result_data.get("drift_score", 0.0))

                # Create gauge dynamically if first time seeing this feature
                safe_name = col_name.replace("-", "_").replace(" ", "_").lower()
                gauge_key = f"feature_{safe_name}"
                if gauge_key not in feature_drift_gauges:
                    feature_drift_gauges[gauge_key] = Gauge(
                        f"evidently_feature_drift_{safe_name}",
                        f"Drift score for feature: {col_name}",
                        registry=drift_registry,
                    )
                feature_drift_gauges[gauge_key].set(col_score)

    except Exception as e:
        log.error("Drift computation failed: %s", e)


def drift_refresh_loop():
    """Background thread — recomputes drift every REFRESH_INTERVAL seconds."""
    while True:
        compute_drift()
        time.sleep(REFRESH_INTERVAL)


# ── HTTP server — serves /metrics to Prometheus ───────────────────────────────
class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            output = generate_latest(drift_registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default HTTP access logs (noisy)


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Evidently Drift Service starting on port %d", SERVICE_PORT)
    log.info("Reference data : %s", REFERENCE_DATA_PATH)
    log.info("Inference log  : %s", INFERENCE_LOG_PATH)
    log.info("Refresh interval: %ds", REFRESH_INTERVAL)
    log.info("Metrics endpoint: http://0.0.0.0:%d/metrics", SERVICE_PORT)
    log.info("=" * 60)

    # Run first drift computation immediately
    compute_drift()

    # Start background refresh thread
    thread = threading.Thread(target=drift_refresh_loop, daemon=True)
    thread.start()

    # Start HTTP server (blocking)
    server = HTTPServer(("0.0.0.0", SERVICE_PORT), MetricsHandler)
    log.info("Evidently service ready — Prometheus can now scrape :8085/metrics")
    server.serve_forever()
