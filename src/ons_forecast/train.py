"""Etapa 5: Treinamento do LightGBM e registro no MLflow."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from ons_forecast.data_loader import load_gold_se
from ons_forecast.features import FEATURE_COLUMNS, build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample_for_bin": 50000,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "random_state": 42,
    "verbosity": -1,
}


def fit_lgbm(train_df: pd.DataFrame, params: dict | None = None) -> LGBMRegressor:
    """Treina um LGBMRegressor com os atributos definidos em FEATURE_COLUMNS."""
    model_params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    clean = train_df.dropna(subset=["val_cargaenergiamwmed", *FEATURE_COLUMNS])
    model = LGBMRegressor(**model_params)
    model.fit(clean[FEATURE_COLUMNS], clean["val_cargaenergiamwmed"])
    return model


def train_and_log(
    experiment_name: str = "ons-carga-se",
    holdout_days: int = 84,
) -> None:
    """Treina o modelo no passado, avalia no holdout futuro e registra no MLflow."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    gold_df = load_gold_se()
    feat_df = build_features(gold_df).dropna(
        subset=["val_cargaenergiamwmed", *FEATURE_COLUMNS]
    )

    max_date = feat_df["data_referencia"].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_days)

    train_df = feat_df[feat_df["data_referencia"] <= cutoff_date]
    test_df = feat_df[feat_df["data_referencia"] > cutoff_date]

    with mlflow.start_run(run_name="lightgbm-se-d7"):
        mlflow.log_params(DEFAULT_LGBM_PARAMS)
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("train_start", str(train_df["data_referencia"].min().date()))
        mlflow.log_param("train_end", str(train_df["data_referencia"].max().date()))
        mlflow.log_param("test_start", str(test_df["data_referencia"].min().date()))
        mlflow.log_param("test_end", str(test_df["data_referencia"].max().date()))

        model = fit_lgbm(train_df)
        preds = model.predict(test_df[FEATURE_COLUMNS])

        mae = mean_absolute_error(test_df["val_cargaenergiamwmed"], preds)
        mape = mean_absolute_percentage_error(test_df["val_cargaenergiamwmed"], preds) * 100

        mlflow.log_metric("holdout_mae", mae)
        mlflow.log_metric("holdout_mape", mape)
        mlflow.lightgbm.log_model(model, name="model")

        logger.info(
            "Modelo registrado no MLflow | Holdout (%s a %s): MAE=%.2f MWmed | MAPE=%.2f%%",
            test_df["data_referencia"].min().date(),
            test_df["data_referencia"].max().date(),
            mae,
            mape,
        )


if __name__ == "__main__":
    train_and_log()
