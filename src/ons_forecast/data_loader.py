"""Utilitário para carregar a série Gold do DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path("data/warehouse.duckdb")


def load_gold_se(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Carrega a série diária do subsistema SE da camada Gold no DuckDB."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Banco DuckDB não encontrado em {db_path}. Execute o dbt antes."
        )

    with duckdb.connect(str(db_path), read_only=True) as conn:
        df = conn.execute(
            """
            SELECT
                data_referencia,
                id_subsistema,
                nom_subsistema,
                val_cargaenergiamwmed
            FROM gold.gold_carga_diaria_se
            ORDER BY data_referencia
            """
        ).fetchdf()

    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    return df
