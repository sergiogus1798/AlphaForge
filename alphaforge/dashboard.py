"""
AlphaForge — Unified Dashboard
================================
Single Dash app with three tabs:

  1. General Backtest   — 2×2 distribution analysis + stats sidebar
  2. Rolling Breakdown  — N-period distribution grid + stats sidebar
  3. Equity Analysis    — Equity curve / drawdown / benchmark + stats sidebar

Usage:
    from alphaforge.dashboard import run_dashboard
    run_dashboard(strategies_dict)
    run_dashboard(single_df)
"""

import math
import threading
import webbrowser

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import dash
from dash import Input, Output, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alphaforge.metrics import compute_metrics, sharpe_ratio

# ── Palette ────────────────────────────────────────────────────────────────────
_BG     = "#0f1117"
_CARD   = "#1a1a2e"
_PANEL  = "#16213e"
_BOX    = "#0f3460"
_ACCENT = "#00d4ff"
_GREEN  = "#2ecc71"
_RED    = "#e74c3c"
_TEXT   = "#e0e0e0"
_DIM    = "#888888"
_WARN   = "#f0a500"
_KDE    = "#f0a500"
_NORMAL = "#e056fd"
_CI     = "#00d4ff"
_MEAN   = "#ffd32a"
_BENCH  = "#aaaaaa"

PERIOD_COLORS = ["#00d4ff", "#f0a500", "#2ecc71", "#e056fd", "#ff6b6b", "#ffd32a"]

_AXIS = dict(gridcolor=_BOX, showgrid=True, zeroline=False,
             tickfont=dict(color=_DIM, size=10))
_LAYOUT = dict(
    paper_bgcolor=_BG, plot_bgcolor=_PANEL,
    font=dict(color=_TEXT, family="'Courier New', monospace", size=11),
    legend=dict(bgcolor=_CARD, bordercolor=_BOX, borderwidth=1,
                font=dict(color=_TEXT, size=10)),
    margin=dict(l=8, r=8, t=32, b=8),
    hovermode="x",
    hoverlabel=dict(bgcolor=_CARD, bordercolor=_BOX, font=dict(color=_TEXT)),
)

# Height constants (px) — figures use explicit pixel heights for reliability
_CHART_H_PX  = 400    # each of the 4 overview charts
_EQUITY_H    = "calc(100vh - 175px)"   # equity chart (CSS, single full-height panel)

_STATS_ANN = dict(
    x=0.99, y=0.99, xref="paper", yref="paper",
    xanchor="right", yanchor="top", showarrow=False,
    font=dict(color=_TEXT, size=10, family="'Courier New', monospace"),
    bgcolor=_BOX, bordercolor=_ACCENT, borderpad=6, align="left",
)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure builders
# ══════════════════════════════════════════════════════════════════════════════

def _hist_fig(pnl: np.ndarray, title: str = "", bins: int = 60) -> go.Figure:
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 2:
        return go.Figure()

    counts, edges = np.histogram(pnl, bins=bins)
    centers   = (edges[:-1] + edges[1:]) / 2
    bin_width = float(edges[1] - edges[0])
    scale     = len(pnl) * bin_width

    mu       = float(pnl.mean())
    sigma    = float(pnl.std())
    skew     = float(scipy_stats.skew(pnl))
    kurt     = float(scipy_stats.kurtosis(pnl))
    var_99   = float(np.percentile(pnl, 1))
    ci_low   = float(np.percentile(pnl, 2.5))
    ci_high  = float(np.percentile(pnl, 97.5))

    xr = np.linspace(float(pnl.min()), float(pnl.max()), 500)
    kde_y    = scipy_stats.gaussian_kde(pnl)(xr) * scale
    normal_y = scipy_stats.norm.pdf(xr, mu, sigma) * scale

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker_color=[_GREEN if c >= 0 else _RED for c in centers],
        showlegend=False,
        hovertemplate="$%{x:,.0f}: %{y} trades<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=xr, y=kde_y,    mode="lines", name="KDE",
                             line=dict(color=_KDE,    width=2.2)))
    fig.add_trace(go.Scatter(x=xr, y=normal_y, mode="lines", name="Normal",
                             line=dict(color=_NORMAL, width=1.6, dash="dash"),
                             fill="tozeroy", fillcolor="rgba(224,86,253,0.08)"))

    skew_f = "  ⚠" if abs(skew) > 0.5 else ""
    kurt_f = "  ⚠" if abs(kurt) > 1.0 else ""

    fig.update_layout(**{
        **_LAYOUT,
        "height": _CHART_H_PX,
        "title": dict(text=title, font=dict(color=_TEXT, size=11), x=0.5),
        "bargap": 0.05,
        "legend": dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                       bgcolor="rgba(15,52,96,0.85)", bordercolor=_ACCENT, borderwidth=1,
                       font=dict(color=_TEXT, size=10)),
        "xaxis": dict(**_AXIS, title="Trade PnL ($)", tickprefix="$"),
        "yaxis": dict(**_AXIS, title="Count"),
        "shapes": [
            dict(type="line", x0=0,      x1=0,      y0=0, y1=1, yref="paper",
                 line=dict(color="black",   width=2.0)),
            dict(type="line", x0=var_99,  x1=var_99, y0=0, y1=1, yref="paper",
                 line=dict(color="#ff9f43", width=1.4, dash="dot")),
            dict(type="line", x0=mu,      x1=mu,     y0=0, y1=1, yref="paper",
                 line=dict(color=_MEAN,    width=1.3, dash="dot")),
            dict(type="line", x0=ci_low,  x1=ci_low, y0=0, y1=1, yref="paper",
                 line=dict(color=_CI,      width=1.2, dash="dash")),
            dict(type="line", x0=ci_high, x1=ci_high,y0=0, y1=1, yref="paper",
                 line=dict(color=_CI,      width=1.2, dash="dash")),
        ],
        "annotations": [
            dict(text=f"VaR 99%<br>${var_99:,.0f}", x=var_99, y=0.88, yref="paper",
                 xanchor="right", showarrow=False, font=dict(color="#ff9f43", size=9)),
            dict(text=f"Mean<br>${mu:,.2f}",         x=mu,     y=0.96, yref="paper",
                 xanchor="left",  showarrow=False, font=dict(color=_MEAN, size=9)),
            dict(text=f"2.5%<br>${ci_low:,.0f}",     x=ci_low, y=0.74, yref="paper",
                 xanchor="right", showarrow=False, font=dict(color=_CI, size=9)),
            dict(text=f"97.5%<br>${ci_high:,.0f}",   x=ci_high,y=0.74, yref="paper",
                 xanchor="left",  showarrow=False, font=dict(color=_CI, size=9)),
            dict(**_STATS_ANN, text=(
                f"μ = ${mu:,.2f}<br>"
                f"σ = ${sigma:,.2f}<br>"
                f"VaR 99% = ${var_99:,.2f}<br>"
                f"Skew = {skew:.3f}{skew_f}<br>"
                f"Kurt = {kurt:.3f}{kurt_f}"
            )),
        ],
    })
    return fig


def _win_loss_fig(pnl: np.ndarray, title: str = "") -> go.Figure:
    pnl    = pnl[~np.isnan(pnl)]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(wins) < 2 or len(losses) < 2:
        return go.Figure()

    xr     = np.linspace(float(pnl.min()), float(pnl.max()), 600)
    ratio  = len(wins) / len(losses)
    payoff = wins.mean() / abs(losses.mean())

    fig = go.Figure()
    for subset, color, label in [(wins, _GREEN, "Wins"), (losses, _RED, "Losses")]:
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        kde_y = scipy_stats.gaussian_kde(subset)(xr)
        fig.add_trace(go.Scatter(
            x=xr, y=kde_y, mode="lines", name=label,
            line=dict(color=color, width=2.2),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.18)",
        ))
        fig.add_vline(x=float(subset.mean()), line_color=color,
                      line_width=1.2, line_dash="dash", opacity=0.8)

    fig.add_vline(x=0, line_color="black", line_width=2.0)
    fig.add_annotation(**_STATS_ANN, text=(
        f"Wins   n={len(wins)}   μ=${wins.mean():,.2f}   σ=${wins.std():,.2f}<br>"
        f"Losses n={len(losses)}   μ=${losses.mean():,.2f}   σ=${losses.std():,.2f}<br>"
        f"W/L ratio: {ratio:.2f}   Payoff: {payoff:.2f}"
    ))

    fig.update_layout(**{
        **_LAYOUT,
        "height": _CHART_H_PX,
        "title": dict(text=title, font=dict(color=_TEXT, size=11), x=0.5),
        "legend": dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                       bgcolor="rgba(15,52,96,0.85)", bordercolor=_ACCENT, borderwidth=1,
                       font=dict(color=_TEXT, size=10)),
        "xaxis": dict(**_AXIS, title="Trade PnL ($)", tickprefix="$"),
        "yaxis": dict(**_AXIS, title="Density"),
    })
    return fig


def _rolling_dist_fig(df: pd.DataFrame, n_periods: int, title: str = "",
                      initial_capital: float = 10_000) -> go.Figure:
    sorted_df = df.sort_values("Close time").reset_index(drop=True)
    chunks    = np.array_split(sorted_df, n_periods)
    all_pnl   = sorted_df["Profit/Loss"].dropna().values
    xr        = np.linspace(float(all_pnl.min()), float(all_pnl.max()), 600)

    fig = go.Figure()
    for i, chunk in enumerate(chunks):
        pnl = chunk["Profit/Loss"].dropna().values
        if len(pnl) < 2:
            continue
        color = PERIOD_COLORS[i % len(PERIOD_COLORS)]
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        t0    = chunk["Close time"].min()
        t1    = chunk["Close time"].max()
        start = t0.strftime("%Y-%m")
        end   = t1.strftime("%Y-%m")
        years = max((t1 - t0).total_seconds() / (365.25 * 86_400), 1 / 365)
        ann_pct = pnl.sum() / initial_capital / years * 100
        label = (f"P{i+1}  {start}→{end}   {len(pnl)} trades"
                 f"   Net: ${pnl.sum():,.0f}   Ann: {ann_pct:+.1f}%   Avg: ${pnl.mean():,.2f}")
        kde_y = scipy_stats.gaussian_kde(pnl)(xr)
        fig.add_trace(go.Scatter(
            x=xr, y=kde_y, mode="lines", name=label,
            line=dict(color=color, width=2.0),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.12)",
        ))
        fig.add_vline(x=float(pnl.mean()), line_color=color,
                      line_width=1.0, line_dash="dot", opacity=0.7)

    fig.add_vline(x=0, line_color="black", line_width=2.0)
    layout = {**_LAYOUT,
              "height": _CHART_H_PX,
              "title": dict(text=title, font=dict(color=_TEXT, size=11), x=0.5),
              "legend": dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                             bgcolor="rgba(15,52,96,0.85)", bordercolor=_ACCENT,
                             borderwidth=1, font=dict(color=_TEXT, size=9)),
              "margin": dict(l=8, r=8, t=32, b=8),
              "xaxis": dict(**_AXIS, title="Trade PnL ($)", tickprefix="$"),
              "yaxis": dict(**_AXIS, title="Density")}
    fig.update_layout(**layout)
    return fig


def _mae_mfe_fig(df: pd.DataFrame, title: str = "") -> go.Figure:
    mae_col = next((c for c in df.columns if "mae" in c.lower()), None)
    mfe_col = next((c for c in df.columns if "mfe" in c.lower()), None)
    if mae_col is None or mfe_col is None:
        return go.Figure()

    mae = df[mae_col].dropna().abs().values
    mfe = df[mfe_col].dropna().abs().values
    pnl = df["Profit/Loss"].dropna().values
    if len(mae) < 2 or len(mfe) < 2:
        return go.Figure()

    avg_mae = float(mae.mean())
    avg_mfe = float(mfe.mean())
    avg_pnl = float(pnl.mean())
    eff = f"{avg_pnl / avg_mfe * 100:.1f}%" if avg_mfe > 0 else "N/A"
    cap = f"{avg_pnl / avg_mae * 100:.1f}%" if avg_mae > 0 else "N/A"

    x_max = float(np.percentile(mfe, 95))
    xr = np.linspace(0, x_max, 600)
    fig = go.Figure()
    for values, color, label in [(mae, _RED, "MAE"), (mfe, _GREEN, "MFE")]:
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        kde_y = scipy_stats.gaussian_kde(values)(xr)
        fig.add_trace(go.Scatter(
            x=xr, y=kde_y, mode="lines", name=label,
            line=dict(color=color, width=2.2),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.18)",
        ))
        fig.add_vline(x=float(values.mean()), line_color=color,
                      line_width=1.2, line_dash="dash", opacity=0.8)

    fig.add_vline(x=avg_pnl, line_color=_MEAN, line_width=1.3, line_dash="dot")
    fig.add_annotation(**_STATS_ANN, text=(
        f"Avg MAE = ${avg_mae:,.2f}<br>"
        f"Avg MFE = ${avg_mfe:,.2f}<br>"
        f"Avg PnL = ${avg_pnl:,.2f}<br>"
        f"Efficiency (PnL/MFE) = {eff}<br>"
        f"PnL/MAE ratio = {cap}"
    ))
    fig.update_layout(**{
        **_LAYOUT,
        "height": _CHART_H_PX,
        "title": dict(text=title, font=dict(color=_TEXT, size=11), x=0.5),
        "legend": dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                       bgcolor="rgba(15,52,96,0.85)", bordercolor=_ACCENT, borderwidth=1,
                       font=dict(color=_TEXT, size=10)),
        "xaxis": dict(**_AXIS, title="$ amount", tickprefix="$", range=[0, x_max]),
        "yaxis": dict(**_AXIS, title="Density"),
    })
    return fig


def _rolling_breakdown_fig(df: pd.DataFrame, n_periods: int,
                           initial_capital: float = 10_000) -> go.Figure:
    sorted_df = df.sort_values("Close time").reset_index(drop=True)
    chunks    = np.array_split(sorted_df, n_periods)
    ncols     = 2 if n_periods <= 4 else 3
    nrows     = math.ceil(n_periods / ncols)

    titles = []
    for i, chunk in enumerate(chunks):
        pnl   = chunk["Profit/Loss"].dropna()
        t0    = chunk["Close time"].min()
        t1    = chunk["Close time"].max()
        start = t0.strftime("%Y-%m")
        end   = t1.strftime("%Y-%m")
        years = max((t1 - t0).total_seconds() / (365.25 * 86_400), 1/365)
        ann_pct = pnl.sum() / initial_capital / years * 100
        eq    = pnl.cumsum()
        mdd   = float((eq.cummax() - eq).max())
        titles.append(
            f"Period {i+1}  |  {start} → {end}  |  {len(pnl)} trades<br>"
            f"Net: ${pnl.sum():,.0f}   Ann.Ret: {ann_pct:+.1f}%   Win: {(pnl>0).mean()*100:.1f}%   Max DD: ${mdd:,.0f}"
        )

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        vertical_spacing=0.14, horizontal_spacing=0.08)

    for i, chunk in enumerate(chunks):
        row = i // ncols + 1
        col = i %  ncols + 1
        pnl   = chunk["Profit/Loss"].dropna().values
        color = PERIOD_COLORS[i % len(PERIOD_COLORS)]
        if len(pnl) < 2:
            continue

        counts, edges = np.histogram(pnl, bins=40)
        centers   = (edges[:-1] + edges[1:]) / 2
        bin_width = float(edges[1] - edges[0])
        scale     = len(pnl) * bin_width
        xr        = np.linspace(float(pnl.min()), float(pnl.max()), 400)

        fig.add_trace(go.Bar(
            x=centers, y=counts,
            marker_color=[_GREEN if c >= 0 else _RED for c in centers],
            showlegend=False,
            hovertemplate="$%{x:,.0f}: %{y}<extra></extra>",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=xr, y=scipy_stats.gaussian_kde(pnl)(xr) * scale,
            mode="lines", line=dict(color=color, width=2.0), showlegend=False,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=xr,
            y=scipy_stats.norm.pdf(xr, pnl.mean(), pnl.std()) * scale,
            mode="lines", line=dict(color=_NORMAL, width=1.4, dash="dash"),
            fill="tozeroy", fillcolor="rgba(224,86,253,0.07)", showlegend=False,
        ), row=row, col=col)
        fig.add_vline(x=0,           line_color="black", line_width=1.6, row=row, col=col)
        fig.add_vline(x=float(pnl.mean()), line_color=_MEAN, line_width=1.2,
                      line_dash="dot", row=row, col=col)

    for i, ann in enumerate(fig.layout.annotations[:n_periods]):
        ann.font.color = PERIOD_COLORS[i % len(PERIOD_COLORS)]
        ann.font.size  = 10

    row_h_px = 380
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        font=dict(color=_TEXT, family="'Courier New', monospace", size=10),
        margin=dict(l=8, r=8, t=40, b=8),
        height=nrows * row_h_px,
        showlegend=False, bargap=0.05, hovermode="x",
        hoverlabel=dict(bgcolor=_CARD, bordercolor=_BOX, font=dict(color=_TEXT)),
    )
    fig.update_xaxes(**_AXIS, tickprefix="$")
    fig.update_yaxes(**_AXIS)
    return fig


def _equity_fig(df: pd.DataFrame, initial_capital: float,
                benchmark_return: float, rng: str) -> go.Figure:
    df = _filter_range(df, rng)
    if df.empty:
        return go.Figure()

    dates, equity = _equity_series(df, initial_capital)
    equity = equity - initial_capital
    dd    = _drawdown_pct(equity + initial_capital)
    bench = _benchmark(dates, initial_capital, benchmark_return) - initial_capital

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.03)

    fig.add_trace(go.Scatter(
        x=dates, y=equity, mode="lines", name="Equity",
        line=dict(color=_ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        hovertemplate="<b>Equity</b>: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=bench, mode="lines",
        name=f"Benchmark ({benchmark_return*100:.0f}%/yr)",
        line=dict(color=_BENCH, width=1.5, dash="dash"),
        hovertemplate="<b>Benchmark</b>: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=dd, mode="lines", name="Drawdown",
        line=dict(color=_RED, width=1.2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.22)",
        hovertemplate="<b>Drawdown</b>: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(**{
        **_LAYOUT,
        "legend": dict(bgcolor=_CARD, bordercolor=_BOX, borderwidth=1,
                       font=dict(color=_TEXT, size=10),
                       orientation="h", x=0, y=1.02, xanchor="left"),
        "margin": dict(l=10, r=10, t=10, b=10),
        "hovermode": "x unified",
    })
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    fig.update_yaxes(tickprefix="$",  row=1, col=1)
    fig.update_yaxes(ticksuffix="%",  row=2, col=1)
    return fig


# ── Data helpers ───────────────────────────────────────────────────────────────

def _equity_series(df, ic):
    s = df.sort_values("Close time").reset_index(drop=True)
    return s["Close time"], (ic + s["Profit/Loss"].cumsum()).values


def _drawdown_pct(equity):
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak * 100


def _benchmark(dates, ic, r):
    t0    = dates.iloc[0]
    years = (dates - t0).dt.total_seconds() / (365.25 * 86_400)
    return ic * (1 + r) ** years.values


def _filter_range(df, rng):
    if rng == "All":
        return df
    end   = df["Close time"].max()
    delta = {"3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
             "1Y": pd.DateOffset(years=1),  "3Y": pd.DateOffset(years=3)}[rng]
    return df[df["Close time"] >= end - delta].reset_index(drop=True)


def _symbol(df):
    if "Symbol" not in df.columns:
        return None
    raw = str(df["Symbol"].iloc[0])
    return raw.split("_")[0] if "_" in raw else raw


def _sortino(df, rfr=0.01):
    start = df["Close time"].min().date()
    end   = df["Close time"].max().date()
    daily = (df.groupby(df["Close time"].dt.date)["Profit/Loss"].sum()
               .reindex(pd.bdate_range(start, end).date, fill_value=0.0))
    rf_d  = 10_000 * rfr / 252
    exc   = daily - rf_d
    neg   = exc[exc < 0]
    if len(neg) < 2:
        return float("nan")
    dstd = float(np.sqrt((neg ** 2).mean()))
    return float((daily.mean() - rf_d) / dstd * np.sqrt(252)) if dstd > 0 else float("nan")


def _calmar(df, ic):
    from alphaforge.metrics import cagr as _cagr, pct_drawdown as _dd
    c = _cagr(df, initial_capital=ic)
    d = _dd(df,  initial_capital=ic)
    return float(c / d) if d and d > 0 else float("nan")


def _cvar(df):
    pnl = df["Profit/Loss"].dropna().values
    var = np.percentile(pnl, 1)
    tail = pnl[pnl <= var]
    return float(tail.mean()) if len(tail) > 0 else float("nan")


def _fmt(v, spec=".2f", prefix="", suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{prefix}{v:{spec}}{suffix}"


# ── UI components ──────────────────────────────────────────────────────────────

def _metric_card(label, value, positive=None):
    color = _GREEN if positive is True else (_RED if positive is False else _TEXT)
    return html.Div(
        style={"flex": "1", "backgroundColor": _CARD, "padding": "12px 16px",
               "textAlign": "center", "borderRight": f"1px solid {_BG}"},
        children=[
            html.Div(label, style={"fontSize": "10px", "color": _DIM,
                                   "marginBottom": "4px", "textTransform": "uppercase",
                                   "letterSpacing": "0.06em"}),
            html.Div(value, style={"fontSize": "18px", "fontWeight": "bold",
                                   "color": color}),
        ],
    )


def _sh(text):   # sidebar header
    return html.Div(text, style={
        "color": _ACCENT, "fontSize": "10px", "fontWeight": "bold",
        "letterSpacing": "0.1em", "textTransform": "uppercase",
        "marginTop": "14px", "marginBottom": "8px",
        "borderBottom": f"1px solid {_BOX}", "paddingBottom": "4px",
    })


def _sr(label, value, warn=False):  # sidebar row
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between",
               "marginBottom": "7px", "fontSize": "12px"},
        children=[
            html.Span(label, style={"color": _DIM}),
            html.Span(value, style={"color": _WARN if warn else _TEXT,
                                    "fontWeight": "bold"}),
        ],
    )


def _sidebar(content_rows, sidebar_id="stats-sidebar", is_open=True):
    label = "◀" if is_open else "▶"
    open_style   = {"position": "relative", "width": "245px", "flexShrink": "0",
                    "backgroundColor": _CARD, "padding": "12px 16px",
                    "borderLeft": f"2px solid {_BOX}",
                    "overflowY": "auto", "alignSelf": "stretch",
                    "transition": "width 0.2s ease"}
    closed_style = {**open_style, "width": "20px", "padding": "8px 0",
                    "overflow": "hidden"}
    style = open_style if is_open else closed_style
    toggle_btn = html.Button(
        label,
        id="sidebar-toggle",
        n_clicks=0,
        title="Collapse / expand",
        style={
            "position": "absolute", "top": "6px", "left": "-18px",
            "width": "18px", "height": "32px",
            "backgroundColor": _BOX, "color": _ACCENT,
            "border": f"1px solid {_BOX}", "borderRight": "none",
            "borderRadius": "4px 0 0 4px",
            "cursor": "pointer", "fontSize": "10px",
            "padding": "0", "lineHeight": "1",
        },
    )
    return html.Div(
        id=sidebar_id,
        style=style,
        children=[toggle_btn] + content_rows,
    )


def _graph(fig):
    return dcc.Graph(figure=fig, config={"displayModeBar": False},
                     style={"width": "100%", "minWidth": "0"})


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_dashboard(
    strategies,
    *,
    initial_capital: float = 10_000,
    benchmark_return: float = 0.05,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """
    Launch the unified AlphaForge dashboard.

    Args:
        strategies:       dict {name: df} OR a single trades DataFrame.
        initial_capital:  Starting equity in USD (default 10 000).
        benchmark_return: Annual return for the benchmark line (default 5%).
        port:             Local port (default 8050).
        debug:            Enable Dash debug/hot-reload mode.
    """
    if isinstance(strategies, pd.DataFrame):
        df   = strategies
        name = df["strategy"].iloc[0] if "strategy" in df.columns else "Strategy"
        strategies = {name: df}

    if len(strategies) > 1:
        combined = pd.concat(strategies.values(), ignore_index=True)
        strategies = {"★ Portfolio (All)": combined, **strategies}

    names   = list(strategies.keys())
    default = names[0]

    app = dash.Dash(__name__, title="AlphaForge")

    _TAB  = {"backgroundColor": _CARD, "color": _DIM, "border": "none",
             "padding": "8px 22px", "fontSize": "12px", "fontFamily": "monospace"}
    _TSEL = {**_TAB, "color": _ACCENT, "borderBottom": f"2px solid {_ACCENT}"}

    app.layout = html.Div(
        style={"backgroundColor": _BG, "height": "100vh", "overflow": "hidden",
               "fontFamily": "'Courier New', monospace", "color": _TEXT,
               "display": "flex", "flexDirection": "column"},
        children=[

            # ── Top bar ───────────────────────────────────────────────────────
            html.Div(
                style={"backgroundColor": _CARD, "padding": "10px 24px",
                       "display": "flex", "alignItems": "center", "gap": "14px",
                       "borderBottom": f"2px solid {_BOX}", "flexShrink": "0"},
                children=[
                    html.Span("AlphaForge", style={"color": _ACCENT, "fontSize": "17px",
                                                    "fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="strat-dd",
                        options=[{"label": n, "value": n} for n in names],
                        value=default,
                        clearable=False,
                        style={"width": "300px", "fontSize": "12px"},
                    ),
                    html.Div(id="sym-badge"),
                    html.Div(style={"flex": "1"}),

                    # Period buttons (shown for distribution tabs)
                    html.Div(id="period-controls", style={"display": "flex",
                                                           "alignItems": "center",
                                                           "gap": "5px"},
                             children=[
                                 html.Span("Rolling periods:", style={"color": _DIM,
                                                               "fontSize": "12px"}),
                                 *[html.Button(str(n), id=f"per-{n}", n_clicks=0,
                                   style={"backgroundColor": _BOX, "color": _TEXT,
                                          "border": f"1px solid {_BOX}",
                                          "borderRadius": "4px", "padding": "4px 11px",
                                          "cursor": "pointer", "fontSize": "12px",
                                          "fontFamily": "monospace"})
                                   for n in [2, 3, 4, 5, 6]],
                             ]),

                    # Time range buttons (shown for equity tab)
                    html.Div(id="range-controls", style={"display": "none",
                                                          "alignItems": "center",
                                                          "gap": "5px"},
                             children=[
                                 *[html.Button(lbl, id=f"rng-{v}", n_clicks=0,
                                   style={"backgroundColor": _BOX, "color": _TEXT,
                                          "border": f"1px solid {_BOX}",
                                          "borderRadius": "4px", "padding": "4px 11px",
                                          "cursor": "pointer", "fontSize": "12px",
                                          "fontFamily": "monospace"})
                                   for lbl, v in [("3M","3M"),("6M","6M"),
                                                  ("1Y","1Y"),("3Y","3Y"),("All","All")]],
                             ]),

                    dcc.Store(id="periods-store", data=4),
                    dcc.Store(id="range-store",   data="All"),
                    dcc.Store(id="sidebar-open",  data=True),
                ],
            ),

            # ── Metrics strip ─────────────────────────────────────────────────
            html.Div(id="metrics-strip",
                     style={"display": "flex", "gap": "1px",
                            "backgroundColor": _BG, "flexShrink": "0"}),

            # ── Tabs ──────────────────────────────────────────────────────────
            dcc.Tabs(
                id="tabs", value="general",
                style={"backgroundColor": _CARD, "flexShrink": "0"},
                children=[
                    dcc.Tab(label="General Backtest",    value="general",
                            style=_TAB, selected_style=_TSEL),
                    dcc.Tab(label="Rolling Distributions", value="rolling",
                            style=_TAB, selected_style=_TSEL),
                    dcc.Tab(label="Equity Analysis",     value="equity",
                            style=_TAB, selected_style=_TSEL),
                ],
            ),

            # ── Tab content ───────────────────────────────────────────────────
            html.Div(id="tab-content",
                     style={"flex": "1", "overflowY": "auto",
                            "display": "flex"}),
        ],
    )

    # ── Store callbacks ────────────────────────────────────────────────────────
    @app.callback(
        Output("periods-store", "data"),
        [Input(f"per-{n}", "n_clicks") for n in [2, 3, 4, 5, 6]],
        prevent_initial_call=True,
    )
    def _pick_periods(*_):
        ctx = dash.callback_context
        if not ctx.triggered:
            return 4
        return int(ctx.triggered[0]["prop_id"].split(".")[0].replace("per-", ""))

    @app.callback(
        Output("range-store", "data"),
        [Input(f"rng-{v}", "n_clicks") for v in ["3M","6M","1Y","3Y","All"]],
        prevent_initial_call=True,
    )
    def _pick_range(*_):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "All"
        return ctx.triggered[0]["prop_id"].split(".")[0].replace("rng-", "")

    # ── Sidebar collapse toggle ────────────────────────────────────────────────
    @app.callback(
        [Output("sidebar-open",  "data"),
         Output("stats-sidebar", "style"),
         Output("sidebar-toggle","children")],
        Input("sidebar-toggle", "n_clicks"),
        dash.dependencies.State("sidebar-open", "data"),
        prevent_initial_call=True,
    )
    def _toggle_sidebar(n, is_open):
        open_style  = {"position": "relative", "width": "245px", "flexShrink": "0",
                       "backgroundColor": _CARD, "padding": "12px 16px",
                       "borderLeft": f"2px solid {_BOX}",
                       "overflowY": "auto", "alignSelf": "stretch",
                       "transition": "width 0.2s ease"}
        closed_style = {**open_style, "width": "20px", "padding": "8px 0",
                        "overflow": "hidden"}
        if is_open:
            return False, closed_style, "▶"
        return True, open_style, "◀"

    # ── Show/hide topbar controls based on active tab ──────────────────────────
    @app.callback(
        [Output("period-controls", "style"),
         Output("range-controls",  "style")],
        Input("tabs", "value"),
    )
    def _toggle_controls(tab):
        flex = {"display": "flex", "alignItems": "center", "gap": "5px"}
        none = {"display": "none"}
        if tab == "equity":
            return none, flex
        return flex, none

    # ── Main render ────────────────────────────────────────────────────────────
    @app.callback(
        [Output("tab-content",   "children"),
         Output("metrics-strip", "children"),
         Output("sym-badge",     "children")],
        [Input("strat-dd",      "value"),
         Input("tabs",          "value"),
         Input("periods-store", "data"),
         Input("range-store",   "data")],
        dash.dependencies.State("sidebar-open", "data"),
    )
    def _render(strat_name, tab, n_periods, rng, sidebar_is_open):
        df  = strategies[strat_name].sort_values("Close time").reset_index(drop=True)
        pnl = df["Profit/Loss"].dropna().values

        # Symbol badge
        sym = _symbol(df)
        badge = html.Span(
            sym or "",
            style={"backgroundColor": _BOX, "color": _ACCENT,
                   "padding": "3px 10px", "borderRadius": "4px",
                   "fontSize": "11px", "fontWeight": "bold"},
        ) if sym else html.Span()

        # Metrics (always full dataset)
        m = compute_metrics(df, initial_capital=initial_capital)

        # ── General Backtest tab ───────────────────────────────────────────────
        if tab == "general":
            strip = [
                _metric_card("Net P&L",    f"${m.get('total_profit',0):,.0f}",
                             m.get('total_profit',0) >= 0),
                _metric_card("Avg Trade",  f"${m.get('average_trade',0):,.2f}",
                             m.get('average_trade',0) >= 0),
                _metric_card("Win Rate",   f"{m.get('winning_percentage',0):.1f}%",
                             m.get('winning_percentage',0) >= 50),
                _metric_card("Profit Factor", f"{m.get('profit_factor',0):.2f}",
                             m.get('profit_factor',0) >= 1.5),
                _metric_card("R Expect.",  f"{m.get('r_expectancy',0):.3f}",
                             m.get('r_expectancy',0) >= 0),
            ]

            skew   = float(scipy_stats.skew(pnl))
            kurt   = float(scipy_stats.kurtosis(pnl))
            var_99 = float(np.percentile(pnl, 1))
            cvar   = _cvar(df)
            wins   = pnl[pnl > 0];  losses = pnl[pnl < 0]
            ratio  = len(wins)/len(losses) if len(losses) > 0 else float("nan")
            payoff = (wins.mean()/abs(losses.mean())
                      if len(wins) > 0 and len(losses) > 0 else float("nan"))
            mae_col = next((c for c in df.columns if "mae" in c.lower()), None)
            mfe_col = next((c for c in df.columns if "mfe" in c.lower()), None)
            avg_mae_str = avg_mfe_str = eff_str = cap_str = "N/A"
            if mae_col and mfe_col:
                avg_mae = df[mae_col].dropna().abs().mean()
                avg_mfe = df[mfe_col].dropna().abs().mean()
                avg_mae_str = f"${avg_mae:,.2f}"
                avg_mfe_str = f"${avg_mfe:,.2f}"
                if avg_mfe > 0:
                    eff_str = f"{pnl.mean() / avg_mfe * 100:.1f}%"
                if avg_mae > 0:
                    cap_str = f"{pnl.mean() / avg_mae * 100:.1f}%"
            shr = sharpe_ratio(df)
            sor = _sortino(df)
            cal = _calmar(df, initial_capital)
            start_dt = df["Close time"].min().strftime("%Y-%m-%d")
            end_dt   = df["Close time"].max().strftime("%Y-%m-%d")

            sl = _sidebar([
                _sh("Performance"),
                _sr("Net P&L",          f"${m.get('total_profit',0):,.2f}"),
                _sr("CAGR",             f"{m.get('cagr',0):.2f}%"),
                _sr("Yearly Avg Ret.",  f"{m.get('yearly_avg_pct_return',0):.2f}%"),
                _sr("Annual % Max DD",  f"{m.get('annual_pct_max_dd',0):.2f}%"),
                _sr("Gross Profit",     f"${m.get('gross_profit',0):,.2f}"),
                _sr("Gross Loss",       f"${m.get('gross_loss',0):,.2f}"),
                _sr("Period",           start_dt),
                _sr("",                 end_dt),
                _sh("Risk-Adjusted"),
                _sr("Sharpe",           _fmt(shr, ".3f")),
                _sr("Sortino",          _fmt(sor, ".3f")),
                _sr("Calmar",           _fmt(cal, ".3f")),
                _sr("Profit Factor",    f"{m.get('profit_factor',0):.2f}"),
                _sr("Return/DD Ratio",  f"{m.get('return_dd_ratio',0):.2f}"),
                _sr("Max Drawdown",     f"${m.get('drawdown',0):,.2f}"),
                _sr("% Drawdown",       f"{m.get('pct_drawdown',0):.2f}%"),
                _sh("Trades"),
                _sr("Total Trades",     f"{m.get('num_trades',0):,}"),
                _sr("Win Rate",         f"{m.get('winning_percentage',0):.1f}%"),
                _sr("Avg Trade",        f"${m.get('average_trade',0):,.2f}"),
                _sr("Avg Win",          f"${m.get('average_win',0):,.2f}"),
                _sr("Avg Loss",         f"${m.get('average_loss',0):,.2f}"),
                _sr("R Expectancy",     f"{m.get('r_expectancy',0):.4f}"),
                _sr("Max Con. Wins",    str(m.get('max_consecutive_wins',0))),
                _sr("Max Con. Losses",  str(m.get('max_consecutive_losses',0))),
                _sh("Distribution"),
                _sr("μ (mean)",         f"${pnl.mean():,.2f}"),
                _sr("σ (std)",          f"${pnl.std():,.2f}"),
                _sr("VaR 99%",          f"${var_99:,.2f}"),
                _sr("CVaR 99%",         _fmt(cvar, ".2f", "$")),
                _sr("Skewness",         f"{skew:.3f}", warn=abs(skew) > 0.5),
                _sr("Kurtosis",         f"{kurt:.3f}", warn=abs(kurt) > 1.0),
                _sh("Win / Loss"),
                _sr("Win count",        f"{len(wins):,}"),
                _sr("Loss count",       f"{len(losses):,}"),
                _sr("W/L Ratio",        _fmt(ratio,  ".2f")),
                _sr("Payoff Ratio",     _fmt(payoff, ".2f")),
                _sh("MAE / MFE"),
                _sr("Avg MAE",          avg_mae_str),
                _sr("Avg MFE",          avg_mfe_str),
                _sr("Efficiency",       eff_str),
                _sr("PnL/MAE",          cap_str),
            ], is_open=sidebar_is_open)

            lbl = strat_name.split("/")[-1]
            sym_lbl = f"  |  {sym}" if sym else ""
            grid = html.Div(
                style={"flex": "1", "display": "grid", "minWidth": "0",
                       "gridTemplateColumns": "1fr 1fr",
                       "gap": "3px", "padding": "3px"},
                children=[
                    _graph(_hist_fig(pnl, f"Trade Distribution — {lbl}{sym_lbl}")),
                    _graph(_win_loss_fig(pnl, f"Win vs Loss — {lbl}{sym_lbl}")),
                    _graph(_rolling_dist_fig(df, n_periods,
                                            f"Rolling Distribution ({n_periods}p) — {lbl}{sym_lbl}",
                                            initial_capital)),
                    _graph(_mae_mfe_fig(df, f"MAE / MFE — {lbl}{sym_lbl}")),
                ],
            )
            content = html.Div(style={"display": "flex", "flex": "1", "minWidth": "0",
                                       "overflow": "hidden"},
                               children=[grid, sl])

        # ── Rolling Breakdown tab ──────────────────────────────────────────────
        elif tab == "rolling":
            strip = [
                _metric_card("Net P&L",   f"${m.get('total_profit',0):,.0f}",
                             m.get('total_profit',0) >= 0),
                _metric_card("Trades",    f"{m.get('num_trades',0):,}"),
                _metric_card("Win Rate",  f"{m.get('winning_percentage',0):.1f}%",
                             m.get('winning_percentage',0) >= 50),
                _metric_card("Avg Trade", f"${m.get('average_trade',0):,.2f}",
                             m.get('average_trade',0) >= 0),
                _metric_card("Max DD",    f"-{m.get('pct_drawdown',0):.2f}%", False),
            ]
            skew = float(scipy_stats.skew(pnl))
            kurt = float(scipy_stats.kurtosis(pnl))

            # Per-period annual returns for sidebar
            sorted_df  = df.sort_values("Close time").reset_index(drop=True)
            chunks     = np.array_split(sorted_df, n_periods)
            period_ann = []
            for i, chunk in enumerate(chunks):
                cpnl  = chunk["Profit/Loss"].dropna()
                t0    = chunk["Close time"].min()
                t1    = chunk["Close time"].max()
                years = max((t1 - t0).total_seconds() / (365.25 * 86_400), 1 / 365)
                ann   = cpnl.sum() / initial_capital / years * 100
                start = t0.strftime("%Y-%m")
                end   = t1.strftime("%Y-%m")
                period_ann.append(_sr(f"P{i+1} {start}→{end}", f"{ann:+.1f}%",
                                      warn=ann < 0))

            sl = _sidebar([
                _sh("Per Period Stats"),
                _sr("Periods",      str(n_periods)),
                _sr("Total trades", f"{len(pnl):,}"),
                _sr("Per period",   f"~{len(pnl)//n_periods:,}"),
                _sh("Annual Return by Period"),
                *period_ann,
                _sh("Overall"),
                _sr("μ (mean)",  f"${pnl.mean():,.2f}"),
                _sr("σ (std)",   f"${pnl.std():,.2f}"),
                _sr("Skewness",  f"{skew:.3f}", warn=abs(skew) > 0.5),
                _sr("Kurtosis",  f"{kurt:.3f}", warn=abs(kurt) > 1.0),
            ], is_open=sidebar_is_open)

            content = html.Div(
                style={"display": "flex", "flex": "1", "minWidth": "0",
                       "overflow": "hidden"},
                children=[
                    html.Div(
                        style={"flex": "1", "minWidth": "0", "overflowY": "auto"},
                        children=[_graph(_rolling_breakdown_fig(df, n_periods, initial_capital))],
                    ),
                    sl,
                ],
            )

        # ── Equity Analysis tab ────────────────────────────────────────────────
        else:
            strip = [
                _metric_card("Net P&L",    f"${m.get('total_profit',0):,.0f}",
                             m.get('total_profit',0) >= 0),
                _metric_card("CAGR",       f"{m.get('cagr',0):.2f}%",
                             m.get('cagr',0) >= 0),
                _metric_card("Sharpe",     f"{m.get('sharpe_ratio',0):.2f}",
                             m.get('sharpe_ratio',0) >= 1.0),
                _metric_card("Max DD",     f"-{m.get('pct_drawdown',0):.2f}%", False),
                _metric_card("Win Rate",   f"{m.get('winning_percentage',0):.1f}%",
                             m.get('winning_percentage',0) >= 50),
            ]
            shr = sharpe_ratio(df)
            sor = _sortino(df)
            cal = _calmar(df, initial_capital)
            cva = _cvar(df)
            pnl_arr = df["Profit/Loss"].dropna().values
            var_99  = float(np.percentile(pnl_arr, 1))
            skew    = float(scipy_stats.skew(pnl_arr))
            kurt    = float(scipy_stats.kurtosis(pnl_arr))
            start_dt = df["Close time"].min().strftime("%Y-%m-%d")
            end_dt   = df["Close time"].max().strftime("%Y-%m-%d")

            sl = _sidebar([
                _sh("Overview"),
                _sr("Trades",       f"{len(df):,}"),
                _sr("Period",       start_dt),
                _sr("",             end_dt),
                _sr("Profit Factor", _fmt(m.get("profit_factor"), ".2f")),
                _sr("R Expectancy",  _fmt(m.get("r_expectancy"), ".3f")),
                _sh("Risk-Adjusted"),
                _sr("Sharpe",   _fmt(shr, ".3f")),
                _sr("Sortino",  _fmt(sor, ".3f")),
                _sr("Calmar",   _fmt(cal, ".3f")),
                _sr("Ret/DD",   _fmt(m.get("return_dd_ratio"), ".2f")),
                _sh("Distribution"),
                _sr("VaR 99%",  f"${var_99:,.2f}"),
                _sr("CVaR 99%", _fmt(cva, ".2f", "$")),
                _sr("Skewness", f"{skew:.3f}", warn=abs(skew) > 0.5),
                _sr("Kurtosis", f"{kurt:.3f}", warn=abs(kurt) > 1.0),
                _sh("Trades"),
                _sr("Avg Trade", f"${m.get('average_trade',0):,.2f}"),
                _sr("Avg Win",   f"${m.get('average_win',0):,.2f}"),
                _sr("Avg Loss",  f"${m.get('average_loss',0):,.2f}"),
                _sr("Max Con. W", str(m.get("max_consecutive_wins",  0))),
                _sr("Max Con. L", str(m.get("max_consecutive_losses", 0))),
            ], is_open=sidebar_is_open)

            content = html.Div(
                style={"display": "flex", "flex": "1", "minWidth": "0",
                       "overflow": "hidden", "height": _EQUITY_H},
                children=[
                    dcc.Graph(
                        figure=_equity_fig(df, initial_capital, benchmark_return, rng),
                        style={"flex": "1", "minWidth": "0", "height": "100%"},
                        config={"displayModeBar": False},
                    ),
                    sl,
                ],
            )

        return content, strip, badge

    # ── Launch ────────────────────────────────────────────────────────────────
    url = f"http://127.0.0.1:{port}"
    if not debug:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\nAlphaForge Dashboard\n  {url}\n  Ctrl+C to stop.\n")
    app.run(port=port, debug=debug)
