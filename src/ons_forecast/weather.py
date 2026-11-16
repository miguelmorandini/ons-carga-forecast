"""Bonus: INMET historical weather ingestion and exogenous temperature evaluation."""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from ons_forecast.baseline import compute_naive_baseline
from ons_forecast.data_loader import load_gold_se
from ons_forecast.features import FEATURE_COLUMNS, build_features
from ons_forecast.train import fit_lgbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INMET_BASE_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos"
SE_STATIONS = ("A701", "A652", "A521", "A001")
DEFAULT_WEATHER_PATH = Path("data/bronze/inmet_se_daily_temp.parquet")
WEATHER_FEATURES = ["temp_mean_lag_7", "temp_max_lag_7", "temp_rolling_mean_7"]


def parse_inmet_station_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse a single INMET automatic station CSV into daily temperature aggregates."""
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=";",
        skiprows=8,
        decimal=",",
        encoding="latin1",
        usecols=lambda col: col.upper().startswith("DATA") or "BULBO SECO" in col.upper(),
    )
    date_col = df.columns[0]
    temp_col = df.columns[1]

    out = pd.DataFrame(
        {
            "data_referencia": pd.to_datetime(df[date_col], format="mixed", errors="coerce"),
            "temp_c": pd.to_numeric(df[temp_col], errors="coerce"),
        }
    ).dropna()
    # Filter INMET sentinel values (-9999) and physically impossible readings
    out = out[(out["temp_c"] > -50.0) & (out["temp_c"] < 60.0)]
    if out.empty:
        return pd.DataFrame(columns=["data_referencia", "temp_mean", "temp_max", "temp_min"])

    daily = (
        out.groupby("data_referencia", as_index=False)["temp_c"]
        .agg(temp_mean="mean", temp_max="max", temp_min="min")
    )
    return daily


def fetch_inmet_se_temperature(
    output_path: Path = DEFAULT_WEATHER_PATH,
    start_year: int = 2018,
    end_year: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download INMET annual ZIPs and aggregate daily SE temperature to Parquet."""
    if output_path.exists() and not force:
        logger.info("Reusing cached INMET temperature dataset at %s", output_path)
        df = pd.read_parquet(output_path)
        df["data_referencia"] = pd.to_datetime(df["data_referencia"])
        return df

    if end_year is None:
        end_year = date.today().year

    yearly_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        url = f"{INMET_BASE_URL}/{year}.zip"
        logger.info("Downloading INMET %d archive from %s ...", year, url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            zip_bytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            matched = [
                name
                for name in zf.namelist()
                if name.upper().endswith(".CSV")
                and any(code in name.upper() for code in SE_STATIONS)
            ]
            station_frames = [parse_inmet_station_csv(zf.read(name)) for name in matched]

        if station_frames:
            combined_year = pd.concat(station_frames, ignore_index=True)
            daily_se = combined_year.groupby("data_referencia", as_index=False).agg(
                temp_mean_se=("temp_mean", "mean"),
                temp_max_se=("temp_max", "max"),
                temp_min_se=("temp_min", "min"),
            )
            yearly_frames.append(daily_se)
            logger.info("Parsed %d daily SE temperature rows for %d.", len(daily_se), year)

    if not yearly_frames:
        raise RuntimeError("No INMET temperature data could be parsed.")

    result = (
        pd.concat(yearly_frames, ignore_index=True)
        .drop_duplicates(subset=["data_referencia"])
        .sort_values("data_referencia")
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info("Saved INMET daily SE temperature to %s (%d rows)", output_path, len(result))
    return result


def add_lagged_weather_features(
    feat_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge INMET daily temperature and compute strictly lagged (>= 7 days) features."""
    w = (
        weather_df[["data_referencia", "temp_mean_se", "temp_max_se"]]
        .drop_duplicates(subset=["data_referencia"])
        .set_index("data_referencia")
        .sort_index()
        .asfreq("D")
    )
    w["temp_mean_lag_7"] = w["temp_mean_se"].shift(7)
    w["temp_max_lag_7"] = w["temp_max_se"].shift(7)
    w["temp_rolling_mean_7"] = w["temp_mean_lag_7"].rolling(window=7, min_periods=4).mean()
    w = w[WEATHER_FEATURES].reset_index()

    return feat_df.merge(w, on="data_referencia", how="left")


def run_weather_backtesting(
    n_windows: int = 10,
    horizon_days: int = 7,
    reports_dir: Path = Path("reports"),
    force_download: bool = False,
) -> pd.DataFrame:
    """Compare LightGBM with and without lagged INMET temperature over common dates."""
    gold_df = load_gold_se()
    weather_df = fetch_inmet_se_temperature(force=force_download)

    baseline_df = compute_naive_baseline(gold_df)
    feat_df = add_lagged_weather_features(build_features(gold_df), weather_df)

    all_features = [*FEATURE_COLUMNS, *WEATHER_FEATURES]
    merged = feat_df.merge(
        baseline_df[["data_referencia", "y_pred_baseline"]],
        on="data_referencia",
        how="left",
    ).dropna(subset=["val_cargaenergiamwmed", "y_pred_baseline", *all_features])

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

        model_base_feats = fit_lgbm(train_df)
        test_df["y_pred_lgbm"] = model_base_feats.predict(test_df[FEATURE_COLUMNS])

        # Train identical LightGBM hyperparameters with added lagged INMET temperature
        from lightgbm import LGBMRegressor

        from ons_forecast.train import DEFAULT_LGBM_PARAMS

        model_weather = LGBMRegressor(**DEFAULT_LGBM_PARAMS)
        model_weather.fit(train_df[all_features], train_df["val_cargaenergiamwmed"])
        test_df["y_pred_lgbm_weather"] = model_weather.predict(test_df[all_features])
        pred_rows.append(test_df)

    all_preds = pd.concat(pred_rows, ignore_index=True)
    y_all = all_preds["val_cargaenergiamwmed"]

    summary_rows = []
    for label, col in [
        ("Baseline (D-7)", "y_pred_baseline"),
        ("LightGBM (Load + Calendar)", "y_pred_lgbm"),
        ("LightGBM + INMET Temp (lag>=7)", "y_pred_lgbm_weather"),
    ]:
        summary_rows.append(
            {
                "modelo": label,
                "mae_mwmed": float(mean_absolute_error(y_all, all_preds[col])),
                "mape_pct": float(mean_absolute_percentage_error(y_all, all_preds[col]) * 100),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(reports_dir / "weather_backtesting_results.csv", index=False)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ons-carga-se")
    with mlflow.start_run(run_name="lightgbm-with-inmet-weather"):
        mlflow.log_param("n_windows", n_windows)
        mlflow.log_param("stations", ",".join(SE_STATIONS))
        mlflow.log_metric("lgbm_mae", float(summary_df.loc[1, "mae_mwmed"]))
        mlflow.log_metric("lgbm_weather_mae", float(summary_df.loc[2, "mae_mwmed"]))
        mlflow.log_metric("lgbm_weather_mape", float(summary_df.loc[2, "mape_pct"]))
        mlflow.log_artifact(str(reports_dir / "weather_backtesting_results.csv"))

    print(
        f"\n=== Weather Backtesting ({all_preds['data_referencia'].min().date()} "
        f"to {all_preds['data_referencia'].max().date()}) ==="
    )
    print(summary_df.to_string(index=False))
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run INMET temperature exogenous benchmark.")
    parser.add_argument("--force-download", action="store_true", help="Re-download INMET ZIPs.")
    args = parser.parse_args()
    run_weather_backtesting(force_download=args.force_download)


if __name__ == "__main__":
    main()
