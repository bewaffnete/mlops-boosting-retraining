import mlflow.xgboost
from mlflow.tracking import MlflowClient
from models.main import mdl
from config import EXPERIMENT_NAME, MODEL_NAME


def log_first_model():
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run(run_name="initial_model"):
        mlflow.log_params({'n_estimators': 430,
                           'max_depth': 12,
                           'learning_rate': 0.05402312783375129,
                           'subsample': 0.9017728392842779,
                           'colsample_bytree': 0.7728227222041226,
                           'reg_lambda': 0.0038363685541247514,
                           'reg_alpha': 3.963210668802833,
                           'min_child_weight': 0.5224491544322849
                           })
        mlflow.log_metric("mse", 126.17)


        model_info = mlflow.xgboost.log_model(
            xgb_model=mdl(),
            artifact_path="model",
            model_format="ubj",
            registered_model_name=MODEL_NAME
        )
    version = model_info.registered_model_version

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version
    )


if __name__ == "__main__":
    log_first_model()
