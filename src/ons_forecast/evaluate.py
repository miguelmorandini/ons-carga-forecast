"""Etapa 6: Backtesting com janela temporal comparando LightGBM vs. Baseline em MAE e MAPE."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from ons_forecast.baseline import compute_naive_baseline
from ons_forecast.data_loader import load_gold_se
from ons_forecast.features import FEATURE_COLUMNS, build_features
from ons_forecast.train import fit_lgbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _predict_autoets(train_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Fit AutoETS(season_length=7) on historical series and forecast horizon_days ahead."""
    sf_input = pd.DataFrame(
        {
            "unique_id": "SE",
            "ds": train_df["data_referencia"],
            "y": train_df["val_cargaenergiamwmed"],
        }
    )
    sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
    forecast_df = sf.forecast(df=sf_input, h=horizon_days)
    return forecast_df.rename(
        columns={"ds": "data_referencia", "AutoETS": "y_pred_autoets"}
    )[["data_referencia", "y_pred_autoets"]]


def _evaluate_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    """Return (MAE, MAPE %) for a prediction series."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return mae, mape


def run_backtesting(
    n_windows: int = 12,
    horizon_days: int = 7,
    reports_dir: Path = Path("reports"),
    experiment_name: str = "ons-carga-se",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa backtesting com janela expansiva semanal (treina no passado, testa D+1..D+7)."""
    gold_df = load_gold_se()
    baseline_df = compute_naive_baseline(gold_df)
    feat_df = build_features(gold_df)

    merged = feat_df.merge(
        baseline_df[["data_referencia", "y_pred_baseline"]],
        on="data_referencia",
        how="left",
    ).dropna(subset=["val_cargaenergiamwmed", "y_pred_baseline", *FEATURE_COLUMNS])

    max_date = merged["data_referencia"].max()
    total_eval_days = n_windows * horizon_days
    eval_start = max_date - pd.Timedelta(days=total_eval_days - 1)

    window_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for w in range(n_windows):
        win_start = eval_start + pd.Timedelta(days=w * horizon_days)
        win_end = win_start + pd.Timedelta(days=horizon_days - 1)

        train_df = merged[merged["data_referencia"] < win_start]
        test_df = merged[
            (merged["data_referencia"] >= win_start)
            & (merged["data_referencia"] <= win_end)
        ].copy()

        if test_df.empty:
            continue

        model = fit_lgbm(train_df)
        test_df["y_pred_lgbm"] = model.predict(test_df[FEATURE_COLUMNS])

        ets_preds = _predict_autoets(
            gold_df[gold_df["data_referencia"] < win_start], horizon_days
        )
        test_df = test_df.merge(ets_preds, on="data_referencia", how="left")
        pred_rows.append(test_df)

        y_true = test_df["val_cargaenergiamwmed"]
        mae_base, mape_base = _evaluate_metrics(y_true, test_df["y_pred_baseline"])
        mae_ets, mape_ets = _evaluate_metrics(y_true, test_df["y_pred_autoets"])
        mae_lgbm, mape_lgbm = _evaluate_metrics(y_true, test_df["y_pred_lgbm"])

        window_rows.append(
            {
                "janela": w + 1,
                "inicio": win_start.date(),
                "fim": win_end.date(),
                "n_dias": len(test_df),
                "mae_baseline": mae_base,
                "mape_baseline_pct": mape_base,
                "mae_autoets": mae_ets,
                "mape_autoets_pct": mape_ets,
                "mae_lgbm": mae_lgbm,
                "mape_lgbm_pct": mape_lgbm,
            }
        )

    windows_df = pd.DataFrame(window_rows)
    all_preds = pd.concat(pred_rows, ignore_index=True)

    y_all = all_preds["val_cargaenergiamwmed"]
    summary_rows = []
    for label, col in [
        ("Baseline (D-7)", "y_pred_baseline"),
        ("AutoETS (statsforecast)", "y_pred_autoets"),
        ("LightGBM", "y_pred_lgbm"),
    ]:
        mae_val, mape_val = _evaluate_metrics(y_all, all_preds[col])
        summary_rows.append({"modelo": label, "mae_mwmed": mae_val, "mape_pct": mape_val})
    summary_df = pd.DataFrame(summary_rows)

    reports_dir.mkdir(parents=True, exist_ok=True)
    gold_df.to_csv(reports_dir / "gold_carga_diaria_se.csv", index=False)
    windows_df.to_csv(reports_dir / "backtesting_windows.csv", index=False)
    summary_df.to_csv(reports_dir / "backtesting_results.csv", index=False)
    pred_cols = [
        "data_referencia",
        "val_cargaenergiamwmed",
        "y_pred_baseline",
        "y_pred_autoets",
        "y_pred_lgbm",
    ]
    all_preds[pred_cols].to_csv(reports_dir / "backtesting_predictions.csv", index=False)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="backtesting-comparison"):
        mlflow.log_param("n_windows", n_windows)
        mlflow.log_param("horizon_days", horizon_days)
        mlflow.log_metric("baseline_mae", float(summary_df.loc[0, "mae_mwmed"]))
        mlflow.log_metric("baseline_mape", float(summary_df.loc[0, "mape_pct"]))
        mlflow.log_metric("autoets_mae", float(summary_df.loc[1, "mae_mwmed"]))
        mlflow.log_metric("autoets_mape", float(summary_df.loc[1, "mape_pct"]))
        mlflow.log_metric("lgbm_mae", float(summary_df.loc[2, "mae_mwmed"]))
        mlflow.log_metric("lgbm_mape", float(summary_df.loc[2, "mape_pct"]))
        mlflow.log_artifact(str(reports_dir / "backtesting_results.csv"))
        mlflow.log_artifact(str(reports_dir / "backtesting_windows.csv"))
        mlflow.log_artifact(str(reports_dir / "backtesting_predictions.csv"))

    print("\n=== Resumo Backtesting (Janelas Temporais de 7 dias) ===")
    print(summary_df.to_string(index=False))
    return windows_df, summary_df


if __name__ == "__main__":
    run_backtesting()
