"""
AlphaForge — Equity Curve Dashboard
====================================
Interactive Dash/Plotly dashboard showing:
  • Equity curve vs benchmark (top panel)
  • Drawdown % (bottom panel, shared x-axis)
  • 5-metric strip (Net P&L, CAGR, Sharpe, Max DD, Win Rate)
  • Stats sidebar (Sharpe, Sortino, Calmar, VaR, CVaR, Skew, Kurt)
  • Time range selector (3M / 6M / 1Y / 3Y / All)
  • Strategy dropdown (multi-strategy support)

Usage:
    from alphaforge.IndividualAnalysis.EquityCurve import run_dashboard
    run_dashboard(strategies_dict)          # dict name → df
    run_dashboard(single_df)                # single DataFrame
"""

import threading
import webbrowser

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import dash
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from alphaforge.metrics import compute_metrics, sharpe_ratio

# ── Palette (matches matplotlib dark theme) ────────────────────────────────────
_BG      = "#0f1117"
_CARD    = "#1a1a2e"
_PANEL   = "#16213e"
_BOX     = "#0f3460"
_ACCENT  = "#00d4ff"
_GREEN   = "#2ecc71"
_RED     = "#e74c3c"
_BENCH   = "#aaaaaa"
_TEXT    = "#e0e0e0"
_DIM     = "#888888"
_WARN    = "#f0a500"

# ── Internal helpers ───────────────────────────────────────────────────────────

def _equity_series(df: pd.DataFrame, initial_capital: float):
    s = df.sort_values("Close time").reset_index(drop=True)
    equity = initial_capital + s["Profit/Loss"].cumsum()
    return s["Close time"], equity.values


def _drawdown_pct(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak * 100


def _benchmark(dates: pd.Series, initial_capital: float, annual_return: float = 0.025) -> np.ndarray:
    t0    = dates.iloc[0]
    years = (dates - t0).dt.total_seconds() / (365.25 * 86_400)
    return initial_capital * (1 + annual_return) ** years.values


def _filter_range(df: pd.DataFrame, rng: str) -> pd.DataFrame:
    if rng == "All":
        return df
    end   = df["Close time"].max()
    delta = {"3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
             "1Y": pd.DateOffset(years=1),  "3Y": pd.DateOffset(years=3)}[rng]
    return df[df["Close time"] >= end - delta].reset_index(drop=True)


def _symbol(df: pd.DataFrame) -> str | None:
    if "Symbol" not in df.columns:
        return None
    raw = str(df["Symbol"].iloc[0])
    return raw.split("_")[0] if "_" in raw else raw


def _sortino(df: pd.DataFrame, risk_free_rate: float = 0.01) -> float:
    """Sortino using same daily business-day approach as Sharpe."""
    start = df["Close time"].min().date()
    end   = df["Close time"].max().date()
    daily = (
        df.groupby(df["Close time"].dt.date)["Profit/Loss"]
        .sum()
        .reindex(pd.bdate_range(start, end).date, fill_value=0.0)
    )
    rf_d   = 10_000 * risk_free_rate / 252
    excess = daily - rf_d
    neg    = excess[excess < 0]
    if len(neg) < 2:
        return float("nan")
    dstd = float(np.sqrt((neg ** 2).mean()))
    return float((daily.mean() - rf_d) / dstd * np.sqrt(252)) if dstd > 0 else float("nan")


def _calmar(df: pd.DataFrame, initial_capital: float) -> float:
    from alphaforge.metrics import cagr as _cagr, pct_drawdown as _dd
    c = _cagr(df, initial_capital=initial_capital)
    d = _dd(df,  initial_capital=initial_capital)
    return float(c / d) if d and d > 0 else float("nan")


def _cvar(df: pd.DataFrame, confidence: float = 0.99) -> float:
    pnl = df["Profit/Loss"].dropna().values
    var = np.percentile(pnl, (1 - confidence) * 100)
    tail = pnl[pnl <= var]
    return float(tail.mean()) if len(tail) > 0 else float("nan")


# ── UI component builders ──────────────────────────────────────────────────────

def _btn(label: str, val: str) -> html.Button:
    return html.Button(
        label,
        id=f"btn-{val}",
        n_clicks=0,
        style={
            "backgroundColor": _BOX,
            "color": _TEXT,
            "border": f"1px solid {_BOX}",
            "borderRadius": "4px",
            "padding": "5px 14px",
            "cursor": "pointer",
            "fontSize": "12px",
            "fontFamily": "monospace",
            "transition": "background 0.15s",
        },
    )


def _metric_card(label: str, value: str, positive: bool | None = None) -> html.Div:
    if positive is True:
        color = _GREEN
    elif positive is False:
        color = _RED
    else:
        color = _TEXT
    return html.Div(
        style={
            "flex": "1",
            "backgroundColor": _CARD,
            "padding": "14px 16px",
            "textAlign": "center",
            "borderRight": f"1px solid {_BG}",
        },
        children=[
            html.Div(label, style={"fontSize": "10px", "color": _DIM, "marginBottom": "5px",
                                   "letterSpacing": "0.06em", "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "bold", "color": color}),
        ],
    )


def _sidebar_header(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "color": _ACCENT,
            "fontSize": "10px",
            "fontWeight": "bold",
            "letterSpacing": "0.1em",
            "textTransform": "uppercase",
            "marginTop": "18px",
            "marginBottom": "10px",
            "borderBottom": f"1px solid {_BOX}",
            "paddingBottom": "5px",
        },
    )


def _sidebar_row(label: str, value: str, warn: bool = False) -> html.Div:
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between",
               "marginBottom": "9px", "fontSize": "12px"},
        children=[
            html.Span(label, style={"color": _DIM}),
            html.Span(value,  style={"color": _WARN if warn else _TEXT, "fontWeight": "bold"}),
        ],
    )


def _fmt(v, fmt=".2f", prefix="", suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{prefix}{v:{fmt}}{suffix}"


# ── Main entry point ───────────────────────────────────────────────────────────

def run_dashboard(
    strategies,
    *,
    initial_capital: float = 10_000,
    benchmark_return: float = 0.025,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """
    Launch the equity curve dashboard in a local browser.

    Args:
        strategies:        dict {name: df} OR a single trades DataFrame.
        initial_capital:   Starting equity in USD (default 10 000).
        benchmark_return:  Annual return for the benchmark line (default 2.5 %).
        port:              Local port (default 8050).
        debug:             Enable Dash debug/hot-reload mode.
    """
    if isinstance(strategies, pd.DataFrame):
        df   = strategies
        name = df["strategy"].iloc[0] if "strategy" in df.columns else "Strategy"
        strategies = {name: df}

    names   = list(strategies.keys())
    default = names[0]

    # ── App shell ─────────────────────────────────────────────────────────────
    app = dash.Dash(__name__, title="AlphaForge — Equity Curve")
    app.layout = html.Div(
        style={"backgroundColor": _BG, "minHeight": "100vh",
               "fontFamily": "'Courier New', monospace", "color": _TEXT},
        children=[

            # ── Top bar ───────────────────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": _CARD,
                    "padding": "10px 24px",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "14px",
                    "borderBottom": f"2px solid {_BOX}",
                },
                children=[
                    html.Span("AlphaForge", style={"color": _ACCENT, "fontSize": "17px",
                                                    "fontWeight": "bold"}),
                    html.Span("/ equity curve", style={"color": _DIM, "fontSize": "13px",
                                                        "marginRight": "16px"}),
                    dcc.Dropdown(
                        id="strat-dd",
                        options=[{"label": n, "value": n} for n in names],
                        value=default,
                        clearable=False,
                        style={"width": "300px", "fontSize": "12px"},
                    ),
                    html.Div(id="symbol-badge"),
                    html.Div(style={"flex": "1"}),
                    html.Div(
                        style={"display": "flex", "gap": "5px"},
                        children=[_btn(lbl, val) for lbl, val in
                                  [("3M","3M"),("6M","6M"),("1Y","1Y"),("3Y","3Y"),("All","All")]],
                    ),
                    dcc.Store(id="range-store", data="All"),
                ],
            ),

            # ── Metrics strip ─────────────────────────────────────────────────
            html.Div(
                id="metrics-strip",
                style={"display": "flex", "gap": "1px", "backgroundColor": _BG,
                       "padding": "1px 0"},
            ),

            # ── Main body ─────────────────────────────────────────────────────
            html.Div(
                style={"display": "flex", "height": "calc(100vh - 162px)"},
                children=[
                    # Chart
                    html.Div(
                        style={"flex": "1", "padding": "8px 8px 8px 8px",
                               "minWidth": "0"},
                        children=[
                            dcc.Graph(
                                id="chart",
                                style={"height": "100%"},
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                    # Sidebar
                    html.Div(
                        id="sidebar",
                        style={
                            "width": "230px",
                            "flexShrink": "0",
                            "backgroundColor": _CARD,
                            "padding": "12px 16px",
                            "borderLeft": f"2px solid {_BOX}",
                            "overflowY": "auto",
                        },
                    ),
                ],
            ),
        ],
    )

    # ── Range store callback ───────────────────────────────────────────────────
    @app.callback(
        Output("range-store", "data"),
        [Input(f"btn-{v}", "n_clicks") for v in ["3M", "6M", "1Y", "3Y", "All"]],
        prevent_initial_call=True,
    )
    def _pick_range(*_):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "All"
        return ctx.triggered[0]["prop_id"].split(".")[0].replace("btn-", "")

    # ── Main dashboard callback ────────────────────────────────────────────────
    @app.callback(
        [
            Output("chart",         "figure"),
            Output("metrics-strip", "children"),
            Output("sidebar",       "children"),
            Output("symbol-badge",  "children"),
        ],
        [Input("strat-dd", "value"), Input("range-store", "data")],
    )
    def _update(strat_name: str, rng: str):
        df_full = strategies[strat_name].sort_values("Close time").reset_index(drop=True)
        df      = _filter_range(df_full, rng)

        if df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                paper_bgcolor=_BG, plot_bgcolor=_PANEL,
                font=dict(color=_TEXT),
                annotations=[dict(text="No data in selected range",
                                  xref="paper", yref="paper", x=0.5, y=0.5,
                                  showarrow=False, font=dict(color=_DIM, size=16))],
            )
            return empty_fig, [], [], html.Span()

        # ── Symbol badge ──────────────────────────────────────────────────────
        sym = _symbol(df)
        badge = html.Span(
            sym or "",
            style={"backgroundColor": _BOX, "color": _ACCENT, "padding": "3px 10px",
                   "borderRadius": "4px", "fontSize": "11px", "fontWeight": "bold",
                   "letterSpacing": "0.08em"},
        ) if sym else html.Span()

        # ── Chart data ────────────────────────────────────────────────────────
        dates, equity = _equity_series(df, initial_capital)
        dd            = _drawdown_pct(equity)
        bench         = _benchmark(dates, initial_capital, benchmark_return)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.03,
        )

        # Equity fill + line
        fig.add_trace(go.Scatter(
            x=dates, y=equity,
            mode="lines", name="Equity",
            line=dict(color=_ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.07)",
            hovertemplate="<b>Equity</b>: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        # Benchmark dashed
        fig.add_trace(go.Scatter(
            x=dates, y=bench,
            mode="lines", name=f"Benchmark ({benchmark_return*100:.0f}%/yr)",
            line=dict(color=_BENCH, width=1.5, dash="dash"),
            hovertemplate="<b>Benchmark</b>: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        # Drawdown area
        fig.add_trace(go.Scatter(
            x=dates, y=dd,
            mode="lines", name="Drawdown",
            line=dict(color=_RED, width=1.2),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.22)",
            hovertemplate="<b>Drawdown</b>: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)

        # Styling
        fig.update_layout(
            paper_bgcolor=_BG,
            plot_bgcolor=_PANEL,
            font=dict(color=_TEXT, family="'Courier New', monospace", size=11),
            legend=dict(
                bgcolor=_CARD, bordercolor=_BOX, borderwidth=1,
                font=dict(color=_TEXT, size=11),
                orientation="h", x=0, y=1.02, xanchor="left",
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
            hoverlabel=dict(bgcolor=_CARD, bordercolor=_BOX, font=dict(color=_TEXT)),
        )
        fig.update_xaxes(gridcolor=_BOX, showgrid=True, zeroline=False,
                         tickfont=dict(color=_DIM))
        fig.update_yaxes(gridcolor=_BOX, showgrid=True, zeroline=False,
                         tickfont=dict(color=_DIM))
        fig.update_yaxes(tickprefix="$", row=1, col=1)
        fig.update_yaxes(ticksuffix="%", row=2, col=1)

        # ── Metrics strip (always on full dataset for consistency) ────────────
        m     = compute_metrics(df_full, initial_capital=initial_capital)
        strip = [
            _metric_card("Net P&L",
                         f"${m.get('total_profit', 0):,.0f}",
                         m.get('total_profit', 0) >= 0),
            _metric_card("CAGR",
                         f"{m.get('cagr', 0):.2f}%",
                         m.get('cagr', 0) >= 0),
            _metric_card("Sharpe",
                         f"{m.get('sharpe_ratio', 0):.2f}",
                         m.get('sharpe_ratio', 0) >= 1.0),
            _metric_card("Max Drawdown",
                         f"-{m.get('pct_drawdown', 0):.2f}%",
                         False),
            _metric_card("Win Rate",
                         f"{m.get('winning_percentage', 0):.1f}%",
                         m.get('winning_percentage', 0) >= 50),
        ]

        # ── Stats sidebar ─────────────────────────────────────────────────────
        pnl  = df_full["Profit/Loss"].dropna()
        skew = float(scipy_stats.skew(pnl))
        kurt = float(scipy_stats.kurtosis(pnl))
        var  = float(np.percentile(pnl, 1))
        cvar = _cvar(df_full)
        sor  = _sortino(df_full)
        cal  = _calmar(df_full, initial_capital)
        shr  = sharpe_ratio(df_full)

        n_trades = len(df_full)
        start_dt = df_full["Close time"].min().strftime("%Y-%m-%d")
        end_dt   = df_full["Close time"].max().strftime("%Y-%m-%d")

        sidebar = [
            _sidebar_header("Overview"),
            _sidebar_row("Trades",    f"{n_trades:,}"),
            _sidebar_row("Period",    start_dt),
            _sidebar_row("",          end_dt),
            _sidebar_row("Profit Factor", _fmt(m.get("profit_factor"), ".2f")),
            _sidebar_row("R Expectancy",  _fmt(m.get("r_expectancy"), ".3f")),

            _sidebar_header("Risk-Adjusted"),
            _sidebar_row("Sharpe",  _fmt(shr,  ".3f")),
            _sidebar_row("Sortino", _fmt(sor,  ".3f")),
            _sidebar_row("Calmar",  _fmt(cal,  ".3f")),
            _sidebar_row("Ret/DD",  _fmt(m.get("return_dd_ratio"), ".2f")),

            _sidebar_header("Distribution"),
            _sidebar_row("VaR 99%",  f"${var:,.2f}"),
            _sidebar_row("CVaR 99%", f"${cvar:,.2f}" if not np.isnan(cvar) else "N/A"),
            _sidebar_row("Skewness", f"{skew:.3f}", warn=abs(skew) > 0.5),
            _sidebar_row("Kurtosis", f"{kurt:.3f}", warn=abs(kurt) > 1.0),

            _sidebar_header("Trades"),
            _sidebar_row("Avg Trade",  f"${m.get('average_trade', 0):,.2f}"),
            _sidebar_row("Avg Win",    f"${m.get('average_win', 0):,.2f}"),
            _sidebar_row("Avg Loss",   f"${m.get('average_loss', 0):,.2f}"),
            _sidebar_row("Max Con. W", str(m.get("max_consecutive_wins",  0))),
            _sidebar_row("Max Con. L", str(m.get("max_consecutive_losses", 0))),
        ]

        return fig, strip, sidebar, badge

    # ── Launch ────────────────────────────────────────────────────────────────
    url = f"http://127.0.0.1:{port}"
    if not debug:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\nAlphaForge Equity Curve Dashboard\n  {url}\n  Ctrl+C to stop.\n")
    app.run(port=port, debug=debug)
