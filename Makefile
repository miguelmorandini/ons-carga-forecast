.PHONY: sync pipeline global-model weather-model lint test dashboard mlflow

sync:
	uv sync --extra pipeline

pipeline:
	uv run ons-pipeline

global-model:
	uv run python -m ons_forecast.global_model

weather-model:
	uv run python -m ons_forecast.weather

lint:
	uv run ruff check .

test:
	uv run pytest -v

dashboard:
	uv run streamlit run src/ons_forecast/dashboard.py

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
