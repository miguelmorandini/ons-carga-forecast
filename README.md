# Previsão de Carga de Energia — Subsistema SE (ONS)

Pipeline ponta a ponta para prever a carga diária de energia (MWmed) do subsistema Sudeste/Centro-Oeste (`SE`) para os próximos 7 dias, utilizando dados abertos do ONS (`2018`–`2026`).

## Resposta à pergunta do desafio

**Sim.** No backtesting temporal com 12 janelas expansivas de 7 dias (84 dias entre 23/08/2026 e 14/11/2026), o modelo **LightGBM** superou o baseline ingênuo (`D-7`, mesmo dia da semana anterior).  
O LightGBM reduziu o **MAE de 2.787,07 MWmed para 2.179,31 MWmed** (ganho de **607,76 MWmed**, ou **21,8%**) e o **MAPE de 6,19% para 4,76%** (redução de **1,43 p.p.**, ou **23,1% relativos**).

| Modelo | MAE (MWmed) | MAPE (%) |
| --- | ---: | ---: |
| Baseline (D-7) | 2.787,07 | 6,19% |
| **LightGBM** | **2.179,31** | **4,76%** |

## Como rodar o projeto

Pré-requisitos: Python `>=3.12` e [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Instalar dependências
uv sync

# 2. Ingestão Bronze (Parquets brutos de 2018 até o ano atual)
uv run python -m ons_forecast.ingest

# 3. Transformação Silver e Gold + testes de qualidade (dbt + DuckDB)
uv run dbt run --project-dir dbt_project --profiles-dir dbt_project
uv run dbt test --project-dir dbt_project --profiles-dir dbt_project

# 4. Baseline (D-7)
uv run python -m ons_forecast.baseline

# 5. Treinamento LightGBM + registro no MLflow
uv run python -m ons_forecast.train

# 6. Backtesting temporal (gera reports/backtesting_results.csv)
uv run python -m ons_forecast.evaluate
```

Para visualizar os experimentos no MLflow:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Decisões técnicas

- **Sem vazamento temporal**: para prever `D+7`, todas as features autoregressivas usam defasagem `>= 7 dias` (`lag_7`, `lag_14`, `lag_21`, `lag_28` e médias/desvios móveis sobre `t-7`) alinhadas por grade diária contínua.
- **Variáveis de calendário**: dia da semana, mês, dia do ano, indicador de fim de semana e feriados nacionais brasileiros (`holidays`).
- **Validação**: backtesting com janela expansiva semanal (sempre treinando no passado e testando nos próximos 7 dias).

## O que eu faria com mais tempo

- Incluir temperatura histórica e prevista (dados do INMET) como variável exógena, já que ondas de calor e frio impactam fortemente a carga do subsistema `SE`.
- Comparar com modelos estatísticos (`AutoARIMA` / `ETS` via `statsforecast`) e testar um modelo global para todos os subsistemas (`SE`, `S`, `NE`, `N`).
- Empacotar o pipeline em Docker/Makefile para execução em comando único, adicionar lint/testes (`ruff` + `pytest`) e um dashboard em Streamlit para o time de operação.
