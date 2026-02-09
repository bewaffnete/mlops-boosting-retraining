import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score, KFold
import optuna
from optuna.samplers import TPESampler

from config import OPTUNA
from database.main import select


def metric(y_true, y_pred):
    return mse(np.expm1(y_true), np.expm1(y_pred)) ** 0.5


def test_data():
    df = select('test')
    df = df.drop('id', axis=1)
    y = df['trip_duration']
    X_test = df.drop('trip_duration', axis=1)
    return X_test, y


def training(X_train, y_train):
    if OPTUNA != '0':
        return training_with_optuna(X_train, y_train)

    params = {
        "max_depth": np.random.randint(4, 15),
        "learning_rate": np.random.uniform(0.01, 0.2),
        "subsample": np.random.uniform(0.6, 1.0),
        'reg_lambda': np.random.uniform(0.001, 0.1),
        'reg_alpha': np.random.uniform(0.1, 5),
        'min_child_weight': np.random.uniform(0.05, 1),
        "colsample_bytree": np.random.uniform(0.6, 1.0),
        "n_estimators": np.random.randint(200, 600),
    }
    X_test, y_test = test_data()

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = metric(y_test, preds)
    return model, params, mse


def training_with_optuna(X_train, y_train):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=25),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),

        }

        model = xgb.XGBRegressor(**params)

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        mse = -np.mean(scores)
        return mse

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
    )

    study.optimize(
        objective,
        n_trials=8,
        timeout=3600,
        n_jobs=1,
        show_progress_bar=True
    )

    best_params = study.best_params

    model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=False
    )

    model.fit(X_train, y_train)
    X_test, y_test = test_data()

    preds = model.predict(X_test)
    mse = metric(y_test, preds)
    return model, best_params, mse
