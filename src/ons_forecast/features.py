"""Engenharia de atributos sem vazamento temporal para previsão D+7."""

from __future__ import annotations

import holidays
import pandas as pd

FEATURE_COLUMNS = [
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday_br",
]
SUBSYSTEM_CATEGORIES = ["SE", "S", "NE", "N"]
GLOBAL_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "id_subsistema"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera features de lags (>= 7 dias) e calendário sobre grade diária contínua."""
    daily = (
        df[["data_referencia", "val_cargaenergiamwmed"]]
        .drop_duplicates(subset=["data_referencia"])
        .set_index("data_referencia")
        .sort_index()
        .asfreq("D")
    )

    # Lags estritamente >= 7 dias para evitar vazamento ao prever D+7
    for lag in (7, 14, 21, 28):
        daily[f"lag_{lag}"] = daily["val_cargaenergiamwmed"].shift(lag)

    # Janelas móveis calculadas exclusivamente sobre dados disponíveis até t - 7
    shifted_7 = daily["val_cargaenergiamwmed"].shift(7)
    daily["rolling_mean_7"] = shifted_7.rolling(window=7, min_periods=4).mean()
    daily["rolling_std_7"] = shifted_7.rolling(window=7, min_periods=4).std()
    daily["rolling_mean_28"] = shifted_7.rolling(window=28, min_periods=14).mean()
    daily["rolling_std_28"] = shifted_7.rolling(window=28, min_periods=14).std()

    # Variáveis de calendário conhecidas antecipadamente para a data alvo t
    idx = daily.index
    daily["day_of_week"] = idx.day_of_week
    daily["month"] = idx.month
    daily["day_of_year"] = idx.day_of_year
    daily["is_weekend"] = (idx.day_of_week >= 5).astype(int)

    years = sorted(idx.year.unique().tolist())
    br_holidays = holidays.country_holidays("BR", years=years)
    daily["is_holiday_br"] = idx.to_series().dt.date.isin(br_holidays).astype(int).values

    return daily.reset_index()


def build_global_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leak-free D+7 features per subsystem for a global multi-series model."""
    frames: list[pd.DataFrame] = []
    for sub_id, sub_df in df.groupby("id_subsistema", sort=False):
        feat = build_features(sub_df)
        feat["id_subsistema"] = sub_id
        frames.append(feat)

    out = pd.concat(frames, ignore_index=True)
    out["id_subsistema"] = pd.Categorical(out["id_subsistema"], categories=SUBSYSTEM_CATEGORIES)
    return out
