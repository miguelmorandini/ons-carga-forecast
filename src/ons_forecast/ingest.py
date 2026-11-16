"""Etapa 2: Ingestão da camada Bronze (Parquets brutos do ONS de 2018 até o ano atual)."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ONS_BASE_URL = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di"
DEFAULT_START_YEAR = 2018


def ingest_bronze(
    output_dir: Path = Path("data/bronze"),
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
) -> list[Path]:
    """Baixa os arquivos Parquet anuais do ONS sem nenhuma alteração para a camada Bronze."""
    if end_year is None:
        end_year = date.today().year

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for year in range(start_year, end_year + 1):
        filename = f"CARGA_ENERGIA_{year}.parquet"
        url = f"{ONS_BASE_URL}/{filename}"
        dest = output_dir / filename

        logger.info("Baixando %s ...", url)
        try:
            tmp_dest = dest.with_suffix(".parquet.tmp")
            urllib.request.urlretrieve(url, tmp_dest)
            tmp_dest.replace(dest)
            size_kb = dest.stat().st_size / 1024
            logger.info("Salvo em %s (%.1f KB)", dest, size_kb)
            downloaded.append(dest)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and year == end_year:
                logger.warning("Arquivo para %d ainda não disponível (404); ignorando.", year)
            else:
                raise

    if not downloaded:
        raise RuntimeError("Nenhum arquivo Parquet foi baixado para a camada Bronze.")

    logger.info("Ingestão Bronze concluída: %d arquivos em %s", len(downloaded), output_dir)
    return downloaded


if __name__ == "__main__":
    ingest_bronze()
