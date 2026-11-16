"""Testes automatizados para o pipeline de previsão de carga ONS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ons_forecast.baseline import compute_naive_baseline
from ons_forecast.features import FEATURE_COLUMNS, build_features
from ons_forecast.ingest import _is_valid_parquet
from ons_forecast.train import fit_lgbm


def _make_sample_series(n_days: int = 60, start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    values = 40000.0 + np.arange(n_days) * 10.0
    return pd.DataFrame({"data_referencia": dates, "val_cargaenergiamwmed": values})


def test_baseline_aligns_strictly_by_d_minus_7() -> None:
    df = _make_sample_series(n_days=20)
    # Remove um dia no meio para garantir que não usa deslocamento posicional (shift)
    df_with_gap = df[df["data_referencia"] != pd.Timestamp("2025-01-10")].reset_index(drop=True)

    result = compute_naive_baseline(df_with_gap)
    row_17 = result.loc[result["data_referencia"] == pd.Timestamp("2025-01-17"), "y_pred_baseline"]
    assert pd.isna(row_17.iloc[0])

    row_18 = result.loc[result["data_referencia"] == pd.Timestamp("2025-01-18"), "y_pred_baseline"]
    expected_11 = df.loc[
        df["data_referencia"] == pd.Timestamp("2025-01-11"), "val_cargaenergiamwmed"
    ]
    assert row_18.iloc[0] == expected_11.iloc[0]


def test_features_have_no_temporal_leakage_within_7_days() -> None:
    df_orig = _make_sample_series(n_days=60)
    target_date = pd.Timestamp("2025-02-15")

    df_perturbed = df_orig.copy()
    mask_recent = (df_perturbed["data_referencia"] > target_date - pd.Timedelta(days=7)) & (
        df_perturbed["data_referencia"] <= target_date
    )
    df_perturbed.loc[mask_recent, "val_cargaenergiamwmed"] = 999999.0

    feat_orig = build_features(df_orig)
    feat_pert = build_features(df_perturbed)

    row_orig = feat_orig.loc[feat_orig["data_referencia"] == target_date, FEATURE_COLUMNS]
    row_pert = feat_pert.loc[feat_pert["data_referencia"] == target_date, FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(row_orig, row_pert)


def test_features_include_brazilian_holidays() -> None:
    df = _make_sample_series(n_days=40, start="2025-04-01")
    feats = build_features(df)
    tiradentes = feats.loc[feats["data_referencia"] == pd.Timestamp("2025-04-21"), "is_holiday_br"]
    assert tiradentes.iloc[0] == 1


def test_fit_lgbm_trains_and_predicts() -> None:
    df = _make_sample_series(n_days=90)
    feats = build_features(df).dropna(subset=["val_cargaenergiamwmed", *FEATURE_COLUMNS])
    model = fit_lgbm(feats.iloc[:-7], params={"n_estimators": 20})
    preds = model.predict(feats.iloc[-7:][FEATURE_COLUMNS])
    assert len(preds) == 7
    assert np.all(np.isfinite(preds))


def test_is_valid_parquet_rejects_corrupt_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "corrupt.parquet"
    bad_file.write_bytes(b"not a parquet file")
    assert _is_valid_parquet(bad_file) is None
