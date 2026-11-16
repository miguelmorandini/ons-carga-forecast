"""Bonus: Global LightGBM model trained across all ONS subsystems (SE, S, NE, N)."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from ons_forecast.baseline import compute_naive_baseline
from ons_forecast.data_loader import load_silver_all
from ons_forecast.features import GLOBAL_FEATURE_COLUMNS, build_global_features
from ons_forecast.train import DEFAULT_LGBM_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_global_backtesting(
    n_windows: int = 12,
    horizon_days: int = 7,
    reports_dir: Path = Path("reports"),
    experiment_name: str = "ons-carga-global",
) -> pd.DataFrame:
    """Evaluate a single global LightGBM across SE, S, NE, and N using expanding windows."""
    silver_df = load_silver_all()
    feat_df = build_global_features(silver_df)

    base_frames: list[pd.DataFrame] = []
    for sub_id, sub_df in silver_df.groupby("id_subsistema", sort=False):
        b = compute_naive_baseline(sub_df)
        b["id_subsistema"] = sub_id
        base_frames.append(b[["data_referencia", "id_subsistema", "y_pred_baseline"]])
    base_all = pd.concat(base_frames, ignore_index=True)

    merged = feat_df.merge(
        base_all,
        on=["data_referencia", "id_subsistema"],
        how="left",
    ).dropna(subset=["val_cargaenergiamwmed", "y_pred_baseline", *GLOBAL_FEATURE_COLUMNS])
    merged["id_subsistema"] = merged["id_subsistema"].astype("category")

    max_date = merged["data_referencia"].max()
    eval_start = max_date - pd.Timedelta(days=n_windows * horizon_days - 1)

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

        model = LGBMRegressor(**DEFAULT_LGBM_PARAMS)
        model.fit(train_df[GLOBAL_FEATURE_COLUMNS], train_df["val_cargaenergiamwmed"])
        test_df["y_pred_global_lgbm"] = model.predict(test_df[GLOBAL_FEATURE_COLUMNS])
        pred_rows.append(test_df)

    all_preds = pd.concat(pred_rows, ignore_index=True)

    rows: list[dict] = []
    for sub_id, sub_preds in all_preds.groupby("id_subsistema", observed=True):
        y_true = sub_preds["val_cargaenergiamwmed"]
        rows.append(
            {
                "id_subsistema": str(sub_id),
                "mae_baseline": float(mean_absolute_error(y_true, sub_preds["y_pred_baseline"])),
                "mape_baseline_pct": float(
                    mean_absolute_percentage_error(y_true, sub_preds["y_pred_baseline"]) * 100
                ),
                "mae_global_lgbm": float(
                    mean_absolute_error(y_true, sub_preds["y_pred_global_lgbm"])
                ),
                "mape_global_lgbm_pct": float(
                    mean_absolute_percentage_error(y_true, sub_preds["y_pred_global_lgbm"]) * 100
                ),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("id_subsistema").reset_index(drop=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(reports_dir / "global_backtesting_results.csv", index=False)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="global-lightgbm-all-subsystems"):
        mlflow.log_param("n_windows", n_windows)
        mlflow.log_param("horizon_days", horizon_days)
        for row in results_df.itertuples():
            sub = row.id_subsistema.lower()
            mlflow.log_metric(f"{sub}_baseline_mae", row.mae_baseline)
            mlflow.log_metric(f"{sub}_global_lgbm_mae", row.mae_global_lgbm)
            mlflow.log_metric(f"{sub}_global_lgbm_mape", row.mape_global_lgbm_pct)
        mlflow.log_artifact(str(reports_dir / "global_backtesting_results.csv"))

    print("\n=== Global LightGBM Backtesting (SE, S, NE, N) ===")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    run_global_backtesting()
