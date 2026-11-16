FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY dbt_project ./dbt_project

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "ons-pipeline"]
