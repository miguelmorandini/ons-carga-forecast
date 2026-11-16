"""Single-command runner for the complete ONS load forecasting pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from ons_forecast.baseline import compute_naive_baseline
from ons_forecast.data_loader import load_gold_se
from ons_forecast.evaluate import run_backtesting
from ons_forecast.ingest import ingest_bronze
from ons_forecast.train import train_and_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DBT_DIR = Path("dbt_project")


def _run_dbt(command: str) -> None:
    """Execute a dbt CLI subcommand using the active Python environment."""
    cmd = [
        sys.executable,
        "-m",
        "dbt.cli.main",
        command,
        "--project-dir",
        str(DBT_DIR),
        "--profiles-dir",
        str(DBT_DIR),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_pipeline(skip_ingest: bool = False, force_ingest: bool = False) -> None:
    """Execute Bronze -> Silver/Gold (dbt) -> Baseline -> LightGBM -> Backtesting."""
    if not skip_ingest:
        logger.info("Step 1/5: Bronze ingestion...")
        ingest_bronze(force=force_ingest)
    else:
        logger.info("Step 1/5: Skipping Bronze ingestion (--skip-ingest).")

    logger.info("Step 2/5: dbt run & dbt test (Silver + Gold)...")
    _run_dbt("run")
    _run_dbt("test")

    logger.info("Step 3/5: Naive baseline (D-7)...")
    gold_df = load_gold_se()
    baseline_df = compute_naive_baseline(gold_df).dropna(subset=["y_pred_baseline"])
    mae = (baseline_df["val_cargaenergiamwmed"] - baseline_df["y_pred_baseline"]).abs().mean()
    logger.info("Baseline full-series MAE: %.2f MWmed", mae)

    logger.info("Step 4/5: LightGBM training + MLflow logging...")
    train_and_log()

    logger.info("Step 5/5: Temporal backtesting...")
    run_backtesting()
    logger.info("Pipeline completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ONS load forecasting pipeline.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip Bronze download step.")
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Re-download all historical Bronze Parquet files.",
    )
    args = parser.parse_args()
    run_pipeline(skip_ingest=args.skip_ingest, force_ingest=args.force_ingest)


if __name__ == "__main__":
    main()
