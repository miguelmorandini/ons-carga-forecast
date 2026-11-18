"""Interactive Streamlit dashboard for ONS Southeast (SE) load forecasting."""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
FONT_FAMILY = "IBM Plex Sans, sans-serif"


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
    if "y_pred_autoets" in preds_df.columns:
        preds_df["erro_abs_autoets"] = (
            preds_df["val_cargaenergiamwmed"] - preds_df["y_pred_autoets"]
        ).abs()
        preds_df["ape_autoets_pct"] = (
            preds_df["erro_abs_autoets"] / preds_df["val_cargaenergiamwmed"]
        ) * 100
    return results_df, windows_df, preds_df


@st.cache_data(show_spinner=False)
def load_optional_benchmarks(
    reports_dir: Path = REPORTS_DIR,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load optional global and exogenous-weather benchmark CSVs when available."""
    global_path = reports_dir / "global_backtesting_results.csv"
    weather_path = reports_dir / "weather_backtesting_results.csv"
    global_df = pd.read_csv(global_path) if global_path.exists() else None
    weather_df = pd.read_csv(weather_path) if weather_path.exists() else None
    return global_df, weather_df


@dataclass(frozen=True)
class ChallengeSummary:
    """Typed summary of backtesting results answering the challenge question."""

    beat_baseline: bool
    mae_baseline: float
    mape_baseline: float
    mae_lgbm: float
    mape_lgbm: float
    mae_autoets: float | None
    mape_autoets: float | None
    mae_reduction_mw: float
    mae_reduction_pct: float
    mape_reduction_pp: float
    mape_rel_gain_pct: float
    lgbm_wins_windows: int
    total_windows: int
    n_days: int
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


def build_challenge_summary(
    results_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    preds_df: pd.DataFrame,
) -> ChallengeSummary:
    """Compute summary metrics answering the challenge question from backtesting outputs."""
    base_row = results_df.loc[results_df["modelo"] == "Baseline (D-7)"].iloc[0]
    lgbm_row = results_df.loc[results_df["modelo"] == "LightGBM"].iloc[0]
    ets_rows = results_df.loc[results_df["modelo"] == "AutoETS (statsforecast)"]

    mae_base = float(base_row["mae_mwmed"])
    mape_base = float(base_row["mape_pct"])
    mae_lgbm = float(lgbm_row["mae_mwmed"])
    mape_lgbm = float(lgbm_row["mape_pct"])

    mae_reduction_mw = mae_base - mae_lgbm
    mae_reduction_pct = (mae_reduction_mw / mae_base) * 100
    mape_reduction_pp = mape_base - mape_lgbm
    mape_rel_gain_pct = (mape_reduction_pp / mape_base) * 100

    lgbm_wins = int((windows_df["mae_lgbm"] < windows_df["mae_baseline"]).sum())
    total_windows = len(windows_df)

    return ChallengeSummary(
        beat_baseline=mae_lgbm < mae_base,
        mae_baseline=mae_base,
        mape_baseline=mape_base,
        mae_lgbm=mae_lgbm,
        mape_lgbm=mape_lgbm,
        mae_autoets=float(ets_rows.iloc[0]["mae_mwmed"]) if not ets_rows.empty else None,
        mape_autoets=float(ets_rows.iloc[0]["mape_pct"]) if not ets_rows.empty else None,
        mae_reduction_mw=mae_reduction_mw,
        mae_reduction_pct=mae_reduction_pct,
        mape_reduction_pp=mape_reduction_pp,
        mape_rel_gain_pct=mape_rel_gain_pct,
        lgbm_wins_windows=lgbm_wins,
        total_windows=total_windows,
        n_days=len(preds_df),
        eval_start=pd.Timestamp(preds_df["data_referencia"].min()),
        eval_end=pd.Timestamp(preds_df["data_referencia"].max()),
    )


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
    if not preds_df.empty and "y_pred_autoets" in preds_df.columns:
        fig.add_trace(
            go.Scatter(
                x=preds_df["data_referencia"],
                y=preds_df["y_pred_autoets"],
                mode="lines",
                name="AutoETS (statsforecast)",
                line={"color": "#2ca02c", "width": 1.8, "dash": "dash"},
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
        font={"family": FONT_FAMILY},
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
        lgbm_col, base_col, ets_col = "erro_abs_lgbm", "erro_abs_baseline", "erro_abs_autoets"
        title, y_title = "Erro Absoluto Diário (MWmed)", "Erro Absoluto (MWmed)"
    else:
        lgbm_col, base_col, ets_col = "ape_lgbm_pct", "ape_baseline_pct", "ape_autoets_pct"
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

    if ets_col in preds_df.columns:
        fig.add_trace(
            go.Scatter(
                x=preds_df["data_referencia"],
                y=preds_df[ets_col],
                mode="lines+markers",
                name="AutoETS (statsforecast)",
                line={"color": "#2ca02c", "dash": "dash"},
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
        font={"family": FONT_FAMILY},
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

    if "mae_autoets" in windows_df.columns:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=windows_df["mae_autoets"],
                name="AutoETS (statsforecast)",
                marker_color="#2ca02c",
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
        font={"family": FONT_FAMILY},
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def _format_model_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format a (modelo, mae_mwmed, mape_pct) benchmark table for display."""
    formatted = df.rename(
        columns={
            "modelo": "Modelo",
            "mae_mwmed": "MAE (MWmed)",
            "mape_pct": "MAPE (%)",
        }
    ).copy()
    formatted["MAE (MWmed)"] = formatted["MAE (MWmed)"].map(lambda v: f"{v:,.2f}")
    formatted["MAPE (%)"] = formatted["MAPE (%)"].map(lambda v: f"{v:.2f}%")
    return formatted


def _render_challenge_answer(summary: ChallengeSummary) -> None:
    """Render the direct executive answer to the challenge question."""
    verdict = "Sim" if summary.beat_baseline else "Não"
    eval_start = summary.eval_start.strftime("%d/%m/%Y")
    eval_end = summary.eval_end.strftime("%d/%m/%Y")
    ets_note = ""
    if summary.mae_autoets is not None and summary.mape_autoets is not None:
        ets_note = (
            f" O modelo estatístico **AutoETS (`statsforecast`)** também superou o baseline "
            f"(MAE de **{summary.mae_autoets:,.2f} MWmed** | "
            f"MAPE de **{summary.mape_autoets:.2f}%**), mas o LightGBM manteve o menor erro."
        )

    message = (
        f"**Resposta ao Desafio — É possível superar o baseline ingênuo (D-7) em D+7?**\n\n"
        f"**{verdict}.** No backtesting temporal com **{summary.total_windows} janelas "
        f"expansivas de 7 dias** ({summary.n_days} dias entre `{eval_start}` e `{eval_end}`), "
        f"o **LightGBM** reduziu o MAE de **{summary.mae_baseline:,.2f} MWmed** para "
        f"**{summary.mae_lgbm:,.2f} MWmed** (ganho de **{summary.mae_reduction_mw:,.2f} MWmed**, "
        f"**-{summary.mae_reduction_pct:.1f}%**) e o MAPE de **{summary.mape_baseline:.2f}%** "
        f"para **{summary.mape_lgbm:.2f}%** (**-{summary.mape_reduction_pp:.2f} p.p.**, ganho "
        f"relativo de **{summary.mape_rel_gain_pct:.1f}%**), vencendo o baseline em "
        f"**{summary.lgbm_wins_windows} de {summary.total_windows}** janelas semanais."
        f"{ets_note}"
    )
    if summary.beat_baseline:
        st.success(message)
    else:
        st.warning(message)


def _render_roadmap_tab(
    hist_df: pd.DataFrame,
    results_df: pd.DataFrame,
    summary: ChallengeSummary,
    global_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> None:
    """Render structured answers covering every stage of the project roadmap."""
    hist_start = hist_df["data_referencia"].min().strftime("%d/%m/%Y")
    hist_end = hist_df["data_referencia"].max().strftime("%d/%m/%Y")
    eval_start = summary.eval_start.strftime("%d/%m/%Y")
    eval_end = summary.eval_end.strftime("%d/%m/%Y")

    st.subheader("Roteiro do Projeto e Respostas Técnicas")
    st.markdown(
        f"""
1. **Objetivo e Baseline Ingênuo (`D-7`)**  
   Prever a carga diária de energia (**MWmed**) do subsistema **Sudeste/Centro-Oeste (`SE`)**
   com horizonte de **7 dias à frente (`D+7`)**, comparando contra o baseline ingênuo
   `y_hat(t) = y(t - 7 dias)` alinhado estritamente por data (evitando deslocamento posicional em
   eventuais falhas de calendário).

2. **Pipeline de Dados (Arquitetura Medalhão: Bronze → Silver → Gold)**  
   - **Bronze**: ingestão anual dos dados abertos do ONS
     (`CARGA_ENERGIA_*.parquet`, `2018`–`2026`).  
   - **Silver (`dbt` + DuckDB)**: padronização de tipos, remoção de nulos e deduplicação por
     `(id_subsistema, data_referencia)` com testes automatizados de qualidade (unicidade, valores
     positivos e continuidade de datas).  
   - **Gold (`gold_carga_diaria_se`)**: recorte diário do subsistema `SE` totalizando
     **{len(hist_df):,} dias** contínuos (`{hist_start}` a `{hist_end}`).

3. **Engenharia de Atributos sem Vazamento Temporal (Zero Leakage)**  
   Para garantir previsibilidade real em `D+7`, todas as variáveis históricas utilizam apenas dados
   disponíveis até `t - 7` sobre uma grade diária contínua:
   - **Lags autorregressivos**: `lag_7`, `lag_14`, `lag_21` e `lag_28`.  
   - **Estatísticas móveis sobre `t - 7`**: média e desvio padrão de 7 e 28 dias
     (`rolling_mean_7`, `rolling_std_7`, `rolling_mean_28`, `rolling_std_28`).  
   - **Calendário conhecido antecipadamente**: `day_of_week`, `month`, `day_of_year`,
     `is_weekend` e feriados nacionais brasileiros (`is_holiday_br` via `holidays`).

4. **Validação Temporal e Benchmark de Modelos**  
   Avaliação com **{summary.total_windows} janelas expansivas de 7 dias**
   ({summary.n_days} dias entre `{eval_start}` e `{eval_end}`): em cada janela, o treino ocorre
   estritamente no passado (`data < inicio_janela`) e a previsão é avaliada nos 7 dias seguintes,
   com rastreamento de parâmetros, métricas e artefatos no **MLflow**.
"""
    )

    st.dataframe(_format_model_metrics_table(results_df), use_container_width=True, hide_index=True)

    st.subheader("Extensões Implementadas (Próximos Passos)")
    col_global, col_weather = st.columns(2)

    with col_global:
        st.markdown("**Modelo Global Multi-Subsistema (`SE`, `S`, `NE`, `N`)**")
        st.caption(
            "Treinamento único compartilhado entre todos os subsistemas brasileiros usando "
            "features isoladas por subsistema e `id_subsistema` como variável categórica."
        )
        if global_df is not None:
            g_disp = global_df.rename(
                columns={
                    "id_subsistema": "Subsistema",
                    "mae_baseline": "MAE Baseline (MWmed)",
                    "mape_baseline_pct": "MAPE Baseline (%)",
                    "mae_global_lgbm": "MAE Global LGBM (MWmed)",
                    "mape_global_lgbm_pct": "MAPE Global LGBM (%)",
                }
            ).copy()
            for col in ("MAE Baseline (MWmed)", "MAE Global LGBM (MWmed)"):
                g_disp[col] = g_disp[col].map(lambda v: f"{v:,.2f}")
            for col in ("MAPE Baseline (%)", "MAPE Global LGBM (%)"):
                g_disp[col] = g_disp[col].map(lambda v: f"{v:.2f}%")
            st.dataframe(g_disp, use_container_width=True, hide_index=True)
        else:
            st.info("Artefato `reports/global_backtesting_results.csv` não encontrado.")

    with col_weather:
        st.markdown("**Variável Exógena de Temperatura (INMET, `lag >= 7`)**")
        st.caption(
            "Temperatura média e máxima diária das estações INMET do Sudeste (`A701`, `A652`, "
            "`A521`, `A001`) com defasagem mínima de 7 dias na janela comum disponível."
        )
        if weather_df is not None:
            st.dataframe(
                _format_model_metrics_table(weather_df),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Artefato `reports/weather_backtesting_results.csv` não encontrado.")


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Previsão de Carga ONS — Subsistema SE",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"], .stApp {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Previsão de Carga de Energia — Subsistema SE (ONS)")
    st.caption(
        "Acompanhamento da série histórica diária (MWmed), previsões D+7 (LightGBM, AutoETS e "
        "Baseline D-7), resposta ao desafio e análise de erro no backtesting temporal."
    )

    try:
        hist_df = load_historical_series()
        results_df, windows_df, preds_df = load_backtesting_artifacts()
        global_df, weather_df = load_optional_benchmarks()
    except FileNotFoundError as exc:
        st.error(
            f"Dados não encontrados ({exc}). Execute antes o pipeline (`dbt run` e "
            "`python -m ons_forecast.evaluate`)."
        )
        return

    summary = build_challenge_summary(results_df, windows_df, preds_df)

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

    tab_dashboard, tab_roadmap = st.tabs(["Dashboard", "Roteiro"])

    with tab_dashboard:
        mae_diff = summary.mae_lgbm - summary.mae_baseline
        mae_pct_gain = (mae_diff / summary.mae_baseline) * 100
        mape_diff = summary.mape_lgbm - summary.mape_baseline

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "MAE — LightGBM",
            f"{summary.mae_lgbm:,.2f} MWmed",
            delta=f"{mae_diff:,.2f} MWmed ({mae_pct_gain:.1f}%)",
            delta_color="inverse",
        )
        col2.metric(
            "MAPE — LightGBM",
            f"{summary.mape_lgbm:.2f}%",
            delta=f"{mape_diff:.2f} p.p.",
            delta_color="inverse",
        )
        col3.metric("MAE — Baseline (D-7)", f"{summary.mae_baseline:,.2f} MWmed")
        col4.metric(
            "Janelas Vencidas (LGBM vs. D-7)",
            f"{summary.lgbm_wins_windows}/{summary.total_windows} ({summary.n_days} dias)",
        )

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
                    "y_pred_autoets": "AutoETS (MWmed)",
                    "y_pred_lgbm": "LightGBM D+7 (MWmed)",
                    "erro_abs_baseline": "Erro Abs. Baseline (MWmed)",
                    "erro_abs_autoets": "Erro Abs. AutoETS (MWmed)",
                    "erro_abs_lgbm": "Erro Abs. LightGBM (MWmed)",
                    "ape_baseline_pct": "APE Baseline (%)",
                    "ape_autoets_pct": "APE AutoETS (%)",
                    "ape_lgbm_pct": "APE LightGBM (%)",
                }
            )
            display_df["Data"] = display_df["Data"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab_roadmap:
        _render_challenge_answer(summary)
        _render_roadmap_tab(hist_df, results_df, summary, global_df, weather_df)


if __name__ == "__main__":
    main()
