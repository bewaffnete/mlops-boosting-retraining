from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
import pandas as pd

from data_processing.feature_engineering import transform
from retraining.drift import data_drift
from database.main import select
from retraining.train import training
from config import EXPERIMENT_NAME, MODEL_NAME


@task(name="Load train & new data", retries=2, retry_delay_seconds=10)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()
    train = select('train')
    train = train.drop('id', axis=1, errors='ignore')

    new = select('new_data')
    new = transform(new)
    new = new.astype(train.dtypes)

    logger.info(f"Loaded train: {len(train)} rows, new: {len(new)} rows")
    return train, new


@task(name="Detect data drift")
def detect_drift(train: pd.DataFrame, new: pd.DataFrame):
    logger = get_run_logger()
    report_path = "evidently_report.html"

    data_drift(train, new)

    logger.info("Drift report saved → evidently_report.html")

    with open(report_path, "rb") as f:
        create_markdown_artifact(
            key="drift-report",
            markdown=f"### Evidently Data Drift Report\n\n![report](artifact://{report_path})",
            description="Data Drift Report (Evidently)"
        )


@task(name="Prepare features for training")
def prepare_dataset(train: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.concat([train, new], axis=0, ignore_index=True)
    y = df['trip_duration']
    X = df.drop('trip_duration', axis=1)
    return X, y


@task(name="Train XGBoost model")
def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    logger = get_run_logger()
    model, params, mse = training(X, y)
    logger.info(f"Model trained → MSE = {mse:.4f}")
    return model, params, mse


@task(name="Log to MLflow + promote champion if better")
def log_and_promote(model, params: dict, mse: float):
    logger = get_run_logger()
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-retrain") as run:
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            model_format="ubj",
            registered_model_name=MODEL_NAME
        )

        # Log Evidently report as MLflow artifact
        mlflow.log_artifact("evidently_report.html", artifact_path="evidently_reports")

        challenger_version = model_info.registered_model_version

        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
            # champion_version = mv.version
            champion_mse = client.get_run(mv.run_id).data.metrics["mse"]
        except Exception:
            champion_version, champion_mse = None, float("inf")

        logger.info(f"Challenger MSE: {mse:.4f} | Champion MSE: {champion_mse:.4f}")

        if mse < champion_mse:
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="champion",
                version=challenger_version
            )
            logger.info(f"→ New champion selected! version = {challenger_version}")
            create_markdown_artifact(
                key="model-promotion",
                markdown=f"**New champion selected!**\n\nMSE: {mse:.4f} → version {challenger_version}",
            )
        else:
            logger.info("Champion remains the same")


@flow(name="Monthly Retraining Pipeline", log_prints=True)
def retraining_pipeline():
    train, new = load_data()
    detect_drift(train, new)
    X, y = prepare_dataset(train, new)
    model, params, mse = train_model(X, y)
    log_and_promote(model, params, mse)


if __name__ == "__main__":
    retraining_pipeline()