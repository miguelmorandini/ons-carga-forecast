"""Interactive Streamlit dashboard for ONS Southeast (SE) load forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from ons_forecast.data_loader import DEFAULT_DB_PATH, load_gold_se
except ModuleNotFoundError:  # Allow running directly on Streamlit Cloud without editable install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ons_forecast.data_loader import DEFAULT_DB_PATH, load_gold_se

REPORTS_DIR = Path("reports")


@st.cache_data(show_spinner=False)
def load_historical_series(reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    """Load the full SE daily load series from DuckDB or the exported Gold CSV."""
    if DEFAULT_DB_PATH.exists():
        df = load_gold_se()
    else:
        csv_path = reports_dir / "gold_carga_diaria_se.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Neither {DEFAULT_DB_PATH} nor {csv_path} was found."
            )
        df = pd.read_csv(csv_path)
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    return df.sort_values("data_referencia").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_backtesting_artifacts(
    reports_dir: Path = REPORTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load backtesting summary, window metrics, and daily predictions."""
    results_path = reports_dir / "backtesting_results.csv"
    windows_path = reports_dir / "backtesting_windows.csv"
    preds_path = reports_dir / "backtesting_predictions.csv"

    for path in (results_path, windows_path, preds_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing backtesting artifact: {path}")

    results_df = pd.read_csv(results_path)
    windows_df = pd.read_csv(windows_path)
    windows_df["inicio"] = pd.to_datetime(windows_df["inicio"])
    windows_df["fim"] = pd.to_datetime(windows_df["fim"])

    preds_df = pd.read_csv(preds_path)
    preds_df["data_referencia"] = pd.to_datetime(preds_df["data_referencia"])
    preds_df["erro_abs_baseline"] = (
        preds_df["val_cargaenergiamwmed"] - preds_df["y_pred_baseline"]
    ).abs()
    preds_df["erro_abs_lgbm"] = (
        preds_df["val_cargaenergiamwmed"] - preds_df["y_pred_lgbm"]
    ).abs()
    preds_df["ape_baseline_pct"] = (
        preds_df["erro_abs_baseline"] / preds_df["val_cargaenergiamwmed"]
    ) * 100
    preds_df["ape_lgbm_pct"] = (
        preds_df["erro_abs_lgbm"] / preds_df["val_cargaenergiamwmed"]
    ) * 100
    return results_df, windows_df, preds_df


def build_series_figure(
    hist_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    show_baseline: bool,
) -> go.Figure:
    """Build the interactive historical load + D+7 forecast overlay chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_df["data_referencia"],
            y=hist_df["val_cargaenergiamwmed"],
            mode="lines",
            name="Carga Real (SE)",
            line={"color": "#1f77b4", "width": 2},
        )
    )
    if show_baseline and not preds_df.empty:
        fig.add_trace(
            go.Scatter(
                x=preds_df["data_referencia"],
                y=preds_df["y_pred_baseline"],
                mode="lines",
                name="Baseline (D-7)",
                line={"color": "#7f7f7f", "width": 1.8, "dash": "dot"},
            )
        )
    if not preds_df.empty:
        fig.add_trace(
            go.Scatter(
                x=preds_df["data_referencia"],
                y=preds_df["y_pred_lgbm"],
                mode="lines",
                name="LightGBM (D+7)",
                line={"color": "#d62728", "width": 2.2},
            )
        )

    fig.update_layout(
        title="Série Histórica e Previsões (MWmed)",
        xaxis_title="Data de Referência",
        yaxis_title="Carga de Energia (MWmed)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def build_error_figure(
    preds_df: pd.DataFrame,
    show_baseline: bool,
    metric: str,
) -> go.Figure:
    """Build the daily forecast error comparison chart (MAE or APE %)."""
    fig = go.Figure()
    if metric == "mwmed":
        lgbm_col, base_col = "erro_abs_lgbm", "erro_abs_baseline"
        title, y_title = "Erro Absoluto Diário (MWmed)", "Erro Absoluto (MWmed)"
    else:
        lgbm_col, base_col = "ape_lgbm_pct", "ape_baseline_pct"
        title, y_title = "Erro Percentual Absoluto Diário (%)", "APE (%)"

    if show_baseline:
        fig.add_trace(
            go.Scatter(
                x=preds_df["data_referencia"],
                y=preds_df[base_col],
                mode="lines+markers",
                name="Baseline (D-7)",
                line={"color": "#7f7f7f", "dash": "dot"},
                marker={"size": 5},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=preds_df["data_referencia"],
            y=preds_df[lgbm_col],
            mode="lines+markers",
            name="LightGBM (D+7)",
            line={"color": "#d62728"},
            marker={"size": 5},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Data de Referência",
        yaxis_title=y_title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def build_windows_figure(windows_df: pd.DataFrame, show_baseline: bool) -> go.Figure:
    """Build the per-window (7-day) MAE comparison bar chart."""
    labels = [f"J{int(row.janela)} ({row.inicio:%d/%m})" for row in windows_df.itertuples()]
    fig = go.Figure()

    if show_baseline:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=windows_df["mae_baseline"],
                name="Baseline (D-7)",
                marker_color="#7f7f7f",
            )
        )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=windows_df["mae_lgbm"],
            name="LightGBM (D+7)",
            marker_color="#d62728",
        )
    )

    fig.update_layout(
        title="MAE por Janela de Backtesting (7 dias)",
        xaxis_title="Janela Temporal",
        yaxis_title="MAE (MWmed)",
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Previsão de Carga ONS — Subsistema SE",
        page_icon="⚡",
        layout="wide",
    )
    st.title("⚡ Previsão de Carga de Energia — Subsistema SE (ONS)")
    st.caption(
        "Acompanhamento da série histórica diária (MWmed), previsões D+7 (LightGBM vs. "
        "Baseline D-7) e análise de erro no backtesting temporal."
    )

    try:
        hist_df = load_historical_series()
        results_df, windows_df, preds_df = load_backtesting_artifacts()
    except FileNotFoundError as exc:
        st.error(
            f"Dados não encontrados ({exc}). Execute antes o pipeline (`dbt run` e "
            "`python -m ons_forecast.evaluate`)."
        )
        return

    base_row = results_df.loc[results_df["modelo"] == "Baseline (D-7)"].iloc[0]
    lgbm_row = results_df.loc[results_df["modelo"] == "LightGBM"].iloc[0]

    mae_diff = lgbm_row["mae_mwmed"] - base_row["mae_mwmed"]
    mae_pct_gain = (mae_diff / base_row["mae_mwmed"]) * 100
    mape_diff = lgbm_row["mape_pct"] - base_row["mape_pct"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "MAE — LightGBM",
        f"{lgbm_row['mae_mwmed']:,.2f} MWmed",
        delta=f"{mae_diff:,.2f} MWmed ({mae_pct_gain:.1f}%)",
        delta_color="inverse",
    )
    col2.metric(
        "MAPE — LightGBM",
        f"{lgbm_row['mape_pct']:.2f}%",
        delta=f"{mape_diff:.2f} p.p.",
        delta_color="inverse",
    )
    col3.metric("MAE — Baseline (D-7)", f"{base_row['mae_mwmed']:,.2f} MWmed")
    col4.metric("Janelas Avaliadas", f"{len(windows_df)} × 7 dias ({len(preds_df)} dias)")

    with st.sidebar:
        st.header("Filtros")
        preset = st.radio(
            "Período da série histórica",
            ["Últimos 6 meses", "Último ano", "Série completa"],
            index=0,
        )
        max_dt = hist_df["data_referencia"].max()
        if preset == "Últimos 6 meses":
            start_dt = max_dt - pd.Timedelta(days=180)
        elif preset == "Último ano":
            start_dt = max_dt - pd.Timedelta(days=365)
        else:
            start_dt = hist_df["data_referencia"].min()

        show_baseline = st.checkbox("Exibir Baseline (D-7)", value=True)
        window_options = ["Todas as janelas"] + [
            f"Janela {int(r.janela)} ({r.inicio:%d/%m} a {r.fim:%d/%m})"
            for r in windows_df.itertuples()
        ]
        selected_window = st.selectbox("Focar janela de backtesting", window_options)

    hist_filtered = hist_df[hist_df["data_referencia"] >= start_dt]
    preds_filtered = preds_df.copy()
    if selected_window != "Todas as janelas":
        win_idx = int(selected_window.split()[1]) - 1
        win_row = windows_df.iloc[win_idx]
        preds_filtered = preds_df[
            (preds_df["data_referencia"] >= win_row["inicio"])
            & (preds_df["data_referencia"] <= win_row["fim"])
        ]
        hist_filtered = hist_df[
            (hist_df["data_referencia"] >= win_row["inicio"] - pd.Timedelta(days=14))
            & (hist_df["data_referencia"] <= win_row["fim"])
        ]

    st.plotly_chart(
        build_series_figure(hist_filtered, preds_filtered, show_baseline),
        use_container_width=True,
    )

    tab_err_mw, tab_err_pct, tab_windows, tab_table = st.tabs(
        [
            "Erro Diário (MWmed)",
            "Erro Diário (%)",
            "Desempenho por Janela",
            "Tabela Detalhada",
        ]
    )

    with tab_err_mw:
        st.plotly_chart(
            build_error_figure(preds_filtered, show_baseline, metric="mwmed"),
            use_container_width=True,
        )
    with tab_err_pct:
        st.plotly_chart(
            build_error_figure(preds_filtered, show_baseline, metric="pct"),
            use_container_width=True,
        )
    with tab_windows:
        st.plotly_chart(
            build_windows_figure(windows_df, show_baseline),
            use_container_width=True,
        )
    with tab_table:
        display_df = preds_filtered.rename(
            columns={
                "data_referencia": "Data",
                "val_cargaenergiamwmed": "Carga Real (MWmed)",
                "y_pred_baseline": "Baseline D-7 (MWmed)",
                "y_pred_lgbm": "LightGBM D+7 (MWmed)",
                "erro_abs_baseline": "Erro Abs. Baseline (MWmed)",
                "erro_abs_lgbm": "Erro Abs. LightGBM (MWmed)",
                "ape_baseline_pct": "APE Baseline (%)",
                "ape_lgbm_pct": "APE LightGBM (%)",
            }
        )
        display_df["Data"] = display_df["Data"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
