"""
train.py — EasyVisa Visa Approval Prediction (V6 — S3 Artifact Store)
"""
import argparse, logging, os, sys, warnings
import joblib, mlflow, mlflow.sklearn
import numpy as np, pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def require_mlflow_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        log.error("=" * 65)
        log.error("MLFLOW_TRACKING_URI is not set.")
        log.error("Fix:")
        log.error("  python -m mlflow server \\")
        log.error("      --backend-store-uri sqlite:///mlflow.db \\")
        log.error("      --default-artifact-root s3://easyvisa-mlflow-vision-2025/mlflow-artifacts \\")
        log.error("      --host 0.0.0.0 --port 5000")
        log.error("  $env:MLFLOW_TRACKING_URI = 'http://localhost:5000'")
        log.error("=" * 65)
        sys.exit(1)
    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if not aws_key or not aws_secret:
        log.warning("AWS credentials not set — S3 artifact writes will fail.")
        log.warning("Fix: set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION")
    log.info("MLflow tracking URI: %s", uri)
    return uri


def load_and_preprocess(data_path: str) -> tuple:
    log.info("Loading data from: %s", data_path)
    data = pd.read_csv(data_path).copy()
    data["no_of_employees"] = abs(data["no_of_employees"])
    data.drop("case_id", axis=1, inplace=True)
    data["case_status"] = data["case_status"].apply(lambda x: 1 if x == "Certified" else 0)
    X = data.drop("case_status", axis=1)
    y = data["case_status"]
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()
    log.info("Shape: %s | Features: %d", X.shape, len(feature_names))
    return X, y, feature_names


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=1, stratify=y_temp)
    log.info("Split — train: %d | val: %d | test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_sampling(X_train, y_train, strategy: str = "over"):
    if strategy == "over":
        log.info("SMOTE oversampling...")
        sampler = SMOTE(random_state=1)
    elif strategy == "under":
        log.info("RandomUnderSampler...")
        sampler = RandomUnderSampler(random_state=1)
    else:
        log.info("No resampling.")
        return X_train, y_train
    return sampler.fit_resample(X_train, y_train)


def build_model(model_name: str, tune: bool, X_train, y_train):
    scorer = metrics.make_scorer(metrics.f1_score)
    if model_name == "gbm":
        base = GradientBoostingClassifier(random_state=1)
        param_grid = {
            "n_estimators": [100, 125, 150, 175, 200],
            "learning_rate": [0.1, 0.05, 0.01, 0.005],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
        }
    elif model_name == "rf":
        base = RandomForestClassifier(random_state=1)
        param_grid = {
            "n_estimators": [50, 75, 100, 125, 150],
            "min_samples_leaf": [1, 2, 4, 5, 10],
            "max_features": ["sqrt", "log2", 0.3, 0.5, None],
            "max_samples": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    elif model_name == "ada":
        base = AdaBoostClassifier(random_state=1)
        param_grid = {
            "n_estimators": [50, 75, 100, 125, 150],
            "learning_rate": [1.0, 0.5, 0.1, 0.01],
            "estimator": [
                DecisionTreeClassifier(max_depth=1, random_state=1),
                DecisionTreeClassifier(max_depth=2, random_state=1),
                DecisionTreeClassifier(max_depth=3, random_state=1),
            ],
        }
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose: gbm | rf | ada")

    if tune:
        log.info("RandomizedSearchCV — 50 iterations × 5 folds...")
        search = RandomizedSearchCV(
            estimator=base, param_distributions=param_grid,
            n_iter=50, n_jobs=-1, scoring=scorer, cv=5, random_state=1
        )
        search.fit(X_train, y_train)
        log.info("Best params: %s | CV F1: %.4f", search.best_params_, search.best_score_)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_
    else:
        log.info("Skipping tuning. Using default hyperparameters.")
        model = base
        best_params = {}
        cv_score = None

    model.fit(X_train, y_train)
    return model, best_params, cv_score


def compute_metrics(model, X, y, prefix: str = "") -> dict:
    pred = model.predict(X)
    return {
        f"{prefix}accuracy": accuracy_score(y, pred),
        f"{prefix}recall": recall_score(y, pred),
        f"{prefix}precision": precision_score(y, pred),
        f"{prefix}f1": f1_score(y, pred),
    }


def log_feature_names_artifact(feature_names: list, artifact_subdir: str = "model"):
    local_path = "feature_names.pkl"
    joblib.dump(feature_names, local_path)
    mlflow.log_artifact(local_path, artifact_path=artifact_subdir)
    log.info("feature_names.pkl logged to MLflow artifact store (S3) under '%s/'", artifact_subdir)


def promote_model(model_name: str, version: int, stage: str = "Production"):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage, archive_existing_versions=True
    )
    log.info("✅ Model '%s' version %d → promoted to '%s'", model_name, version, stage)


def train(
    data_path: str = "EasyVisa.csv",
    model_name: str = "gbm",
    sampling: str = "over",
    tune: bool = True,
    experiment_name: str = "EasyVisa_Visa_Prediction",
    stage: str = "Production",
):
    tracking_uri = require_mlflow_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    registered_model_name = f"easyvisa_{model_name}"

    with mlflow.start_run(run_name=f"{model_name}_{sampling}_v6") as run:
        run_id = run.info.run_id
        log.info("MLflow run started | run_id: %s", run_id)

        mlflow.log_params({
            "model_name": model_name, "sampling_strategy": sampling,
            "tuning_enabled": tune, "data_path": data_path,
            "random_state": 1, "registry_stage": stage,
        })

        X, y, feature_names = load_and_preprocess(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        mlflow.log_params({
            "train_size": len(X_train), "val_size": len(X_val),
            "test_size": len(X_test), "n_features": len(feature_names),
        })

        X_res, y_res = apply_sampling(X_train, y_train, strategy=sampling)
        mlflow.log_param("resampled_train_size", len(X_res))

        model, best_params, cv_score = build_model(model_name, tune=tune, X_train=X_res, y_train=y_res)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", str(v))
        if cv_score is not None:
            mlflow.log_metric("cv_best_f1", cv_score)

        train_m = compute_metrics(model, X_res, y_res, prefix="train_")
        val_m = compute_metrics(model, X_val, y_val, prefix="val_")
        test_m = compute_metrics(model, X_test, y_test, prefix="test_")
        mlflow.log_metrics({**train_m, **val_m, **test_m})

        log.info("─── Validation Metrics ───")
        for k, v in val_m.items():
            log.info("  %-25s %.4f", k, v)
        log.info("─── Test Metrics ─────────")
        for k, v in test_m.items():
            log.info("  %-25s %.4f", k, v)

        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp_path = "feature_importances.csv"
            imp_df.to_csv(imp_path, index=False)
            mlflow.log_artifact(imp_path, artifact_path="reports")
            log.info("Top 5 features:\n%s", imp_df.head().to_string(index=False))

        log_feature_names_artifact(feature_names, artifact_subdir="model")

        input_example = X_val.head(5)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(registered_model_name, stages=["None"])
        new_version = versions[0].version if versions else 1
        log.info("Model registered: '%s' version: %s", registered_model_name, new_version)

        promote_model(registered_model_name, int(new_version), stage=stage)

        log.info("=" * 65)
        log.info("✅ V6 TRAINING COMPLETE")
        log.info("  Registry name : %s", registered_model_name)
        log.info("  Version       : %s", new_version)
        log.info("  Stage         : %s", stage)
        log.info("  Artifact store: S3 (via MLflow)")
        log.info("  api.py loads  : models:/%s/%s", registered_model_name, stage)
        log.info("  MLflow UI     : %s", tracking_uri)
        log.info("=" * 65)

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="EasyVisa V6 — MLflow Registry + S3 Artifact Store Training Pipeline"
    )
    parser.add_argument("--data-path", default="EasyVisa.csv")
    parser.add_argument("--model", default="gbm", choices=["gbm", "rf", "ada"])
    parser.add_argument("--sampling", default="over", choices=["over", "under", "original"])
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--experiment", default="EasyVisa_Visa_Prediction")
    parser.add_argument("--stage", default="Production", choices=["Production", "Staging", "None"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data_path,
        model_name=args.model,
        sampling=args.sampling,
        tune=not args.no_tune,
        experiment_name=args.experiment,
        stage=args.stage,
    )