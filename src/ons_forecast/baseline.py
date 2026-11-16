"""Etapa 4: Baseline ingênuo (carga de D+7 igual à do mesmo dia da semana anterior, D-7)."""

from __future__ import annotations

import pandas as pd

from ons_forecast.data_loader import load_gold_se


def compute_naive_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a previsão ingênua y_hat(t) = y(t - 7 dias) alinhada por data."""
    out = df[["data_referencia", "val_cargaenergiamwmed"]].copy()
    lookup = out.assign(
        data_referencia=out["data_referencia"] + pd.Timedelta(days=7)
    ).rename(columns={"val_cargaenergiamwmed": "y_pred_baseline"})

    out = out.merge(lookup, on="data_referencia", how="left")
    return out


if __name__ == "__main__":
    gold_df = load_gold_se()
    baseline_df = compute_naive_baseline(gold_df)
    valid = baseline_df.dropna(subset=["y_pred_baseline"])
    mae = (valid["val_cargaenergiamwmed"] - valid["y_pred_baseline"]).abs().mean()
    mape = (
        (valid["val_cargaenergiamwmed"] - valid["y_pred_baseline"]).abs()
        / valid["val_cargaenergiamwmed"]
    ).mean() * 100
    print(f"Baseline (série completa válida: {len(valid)} dias) -> MAE: {mae:.2f} MWmed | MAPE: {mape:.2f}%")
