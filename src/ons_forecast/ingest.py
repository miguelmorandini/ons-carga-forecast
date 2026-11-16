"""Etapa 2: Ingestão da camada Bronze (Parquets brutos do ONS de 2018 até o ano atual)."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ONS_BASE_URL = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di"
DEFAULT_START_YEAR = 2018
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


def _is_valid_parquet(path: Path) -> int | None:
    """Valida apenas a integridade estrutural do Parquet (rodapé legível e > 0 linhas)."""
    try:
        metadata = pq.read_metadata(path)
    except Exception:
        return None
    return metadata.num_rows if metadata.num_rows > 0 else None


def _download_with_retries(
    url: str,
    dest: Path,
    timeout: float,
    max_retries: int,
) -> None:
    """Baixa uma URL via streaming com timeout e retentativas em falhas transitórias."""
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, dest.open("wb") as out:
                shutil.copyfileobj(response, out)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_STATUS or attempt == max_retries:
                raise
            wait_s = 2 ** (attempt - 1)
            logger.warning(
                "Erro HTTP %d ao baixar %s (tentativa %d/%d); aguardando %ds...",
                exc.code,
                url,
                attempt,
                max_retries,
                wait_s,
            )
            time.sleep(wait_s)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == max_retries:
                raise
            wait_s = 2 ** (attempt - 1)
            logger.warning(
                "Falha de rede ao baixar %s (%s, tentativa %d/%d); aguardando %ds...",
                url,
                exc,
                attempt,
                max_retries,
                wait_s,
            )
            time.sleep(wait_s)


def ingest_bronze(
    output_dir: Path = Path("data/bronze"),
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[Path]:
    """Baixa os arquivos Parquet anuais do ONS sem nenhuma alteração para a camada Bronze."""
    current_year = date.today().year
    if end_year is None:
        end_year = current_year

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    for year in range(start_year, end_year + 1):
        filename = f"CARGA_ENERGIA_{year}.parquet"
        url = f"{ONS_BASE_URL}/{filename}"
        dest = output_dir / filename

        if not force and year < current_year and dest.exists():
            existing_rows = _is_valid_parquet(dest)
            if existing_rows is not None:
                logger.info("Reutilizando %s (%d linhas, ano fechado).", dest, existing_rows)
                files.append(dest)
                continue
            logger.warning("Arquivo local %s inválido ou vazio; baixando novamente.", dest)

        logger.info("Baixando %s ...", url)
        tmp_dest = dest.with_suffix(".parquet.tmp")
        try:
            _download_with_retries(url, tmp_dest, timeout=timeout, max_retries=max_retries)
            num_rows = _is_valid_parquet(tmp_dest)
            if num_rows is None:
                raise RuntimeError(
                    f"Arquivo baixado de {url} não é um Parquet válido ou está vazio."
                )
            tmp_dest.replace(dest)
            size_kb = dest.stat().st_size / 1024
            logger.info("Salvo em %s (%.1f KB, %d linhas)", dest, size_kb, num_rows)
            files.append(dest)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and year == end_year:
                logger.warning("Arquivo para %d ainda não disponível (404); ignorando.", year)
            else:
                raise
        finally:
            tmp_dest.unlink(missing_ok=True)

    if not files:
        raise RuntimeError("Nenhum arquivo Parquet disponível na camada Bronze.")

    logger.info("Ingestão Bronze concluída: %d arquivos em %s", len(files), output_dir)
    return files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestão da camada Bronze (Parquets ONS).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/bronze"))
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebaixa inclusive anos anteriores já presentes localmente.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ingest_bronze(
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
