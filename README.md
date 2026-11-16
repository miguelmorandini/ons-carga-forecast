# ONS Energy Load Forecast — Subsystem SE

End-to-end pipeline to forecast daily energy load (MWmed) for the Southeast/Central-West (`SE`) subsystem 7 days ahead using ONS open data (`2018`–`2026`).

## Answer to the Challenge Question

**Yes.** Across temporal backtesting with 12 expanding 7-day windows (84 days from `2026-08-23` to `2026-11-14`), **LightGBM** outperformed the naive `D-7` baseline (same day of the previous week).  
LightGBM reduced **MAE from 2,787.07 MWmed to 2,179.31 MWmed** (**607.76 MWmed** improvement, **21.8%** reduction) and **MAPE from 6.19% to 4.76%** (**1.43 p.p.** reduction, **23.1%** relative gain).

| Model | MAE (MWmed) | MAPE (%) |
| --- | ---: | ---: |
| Baseline (D-7) | 2,787.07 | 6.19% |
| AutoETS (`statsforecast`) | 2,365.24 | 5.22% |
| **LightGBM** | **2,179.31** | **4.76%** |

## How to Run

Prerequisites: Python `>=3.12` and [`uv`](https://docs.astral.sh/uv/).

### Single-Command Execution

```bash
uv sync
uv run ons-pipeline
```

*(A [Makefile](Makefile) and [Dockerfile](Dockerfile) are also included for `make pipeline` or containerized runs.)*

### Step-by-Step Execution

```bash
# 1. Install dependencies

# 2. Bronze ingestion (raw Parquet files from 2018 to current year)
uv run python -m ons_forecast.ingest

# 3. Silver and Gold transformations + data quality tests (dbt + DuckDB)
uv run python -m dbt.cli.main run --project-dir dbt_project --profiles-dir dbt_project
uv run python -m dbt.cli.main test --project-dir dbt_project --profiles-dir dbt_project

# 4. Naive baseline (D-7)
uv run python -m ons_forecast.baseline

# 5. Train LightGBM + log experiment to MLflow
uv run python -m ons_forecast.train

# 6. Temporal backtesting (outputs reports/*.csv)
uv run python -m ons_forecast.evaluate
```

### Interactive Dashboard & Experiment Tracking

```bash
# Launch the Streamlit dashboard (Portuguese UI)
uv run streamlit run src/ons_forecast/dashboard.py

# Launch the MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

#### Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io/) and click **Create app**.
2. Select repository `miguelmorandini/ons-carga-forecast`, branch `main`, and main file path `src/ons_forecast/dashboard.py`.
3. Click **Deploy** (the dashboard automatically reads the exported Gold & backtesting CSVs in `reports/` when `data/warehouse.duckdb` is not present).

### Lint & Tests

```bash
uv run ruff check .
uv run pytest -v
```

### Global Multi-Subsystem Model (`SE`, `S`, `NE`, `N`)

```bash
uv run python -m ons_forecast.global_model
```

| Subsystem | Baseline D-7 MAE | Global LightGBM MAE | Global LightGBM MAPE |
| --- | ---: | ---: | ---: |
| `N` | 211.27 | **194.47** | **1.99%** |
| `NE` | **456.29** | 467.87 | 3.12% |
| `S` | 732.35 | **521.52** | **3.90%** |
| `SE` | 2,787.07 | **2,155.03** | **4.69%** |

## Technical Decisions

- **Zero temporal leakage**: to forecast `D+7`, all autoregressive features strictly use lags `>= 7 days` (`lag_7`, `lag_14`, `lag_21`, `lag_28`, and rolling mean/std computed over `t-7`) aligned on a continuous daily grid.
- **Calendar features**: day of week, month, day of year, weekend indicator, and Brazilian national holidays (`holidays`).
- **Validation**: expanding-window weekly backtesting (always training on the past and evaluating on the next 7 days).

## Next Steps (With More Time)

- Add historical and forecasted temperature (INMET data) as an exogenous feature, since heatwaves and cold snaps strongly drive `SE` load.
- Benchmark statistical models (`AutoARIMA` / `ETS` via `statsforecast`) and evaluate a global model across all subsystems (`SE`, `S`, `NE`, `N`).
- Package the full pipeline in Docker/Makefile for single-command execution.
