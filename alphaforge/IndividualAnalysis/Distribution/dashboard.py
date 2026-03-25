"""
AlphaForge — Distribution Dashboard
=====================================
Interactive Dash/Plotly version of all distribution analysis plots.

Tabs:
  • Overview      — 2×2: Trade Distribution | Win vs Loss
                          Rolling Distribution | MAE / MFE
  • Rolling       — N-period breakdown (one full dist plot per period)

Usage:
    from alphaforge.IndividualAnalysis.Distribution import run_distribution_dashboard
    run_distribution_dashboard(strategies_dict)
    run_distribution_dashboard(single_df)
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

# ── Palette ────────────────────────────────────────────────────────────────────
_BG      = "#0f1117"
_CARD    = "#1a1a2e"
_PANEL   = "#16213e"
_BOX     = "#0f3460"
_ACCENT  = "#00d4ff"
_GREEN   = "#2ecc71"
_RED     = "#e74c3c"
_TEXT    = "#e0e0e0"
_DIM     = "#888888"
_WARN    = "#f0a500"
_KDE     = "#f0a500"
_NORMAL  = "#e056fd"
_CI      = "#00d4ff"
_MEAN    = "#ffd32a"
_MAE_C   = "#e74c3c"
_MFE_C   = "#2ecc71"

PERIOD_COLORS = ["#00d4ff", "#f0a500", "#2ecc71", "#e056fd", "#ff6b6b", "#ffd32a"]

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_PANEL,
    font=dict(color=_TEXT, family="'Courier New', monospace", size=11),
    legend=dict(bgcolor=_CARD, bordercolor=_BOX, borderwidth=1,
                font=dict(color=_TEXT, size=10)),
    margin=dict(l=8, r=8, t=32, b=8),
    hovermode="x",
    hoverlabel=dict(bgcolor=_CARD, bordercolor=_BOX, font=dict(color=_TEXT)),
)

_AXIS_STYLE = dict(gridcolor=_BOX, showgrid=True, zeroline=False,
                   tickfont=dict(color=_DIM))


# ── Plot builders ──────────────────────────────────────────────────────────────

def _hist_figure(pnl: np.ndarray, title: str = "", bins: int = 60) -> go.Figure:
    """Trade PnL histogram: green/red bars + KDE + normal fit + markers."""
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 2:
        return go.Figure()

    counts, edges = np.histogram(pnl, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = [_GREEN if c >= 0 else _RED for c in centers]
    bin_width   = edges[1] - edges[0]

    mu        = float(pnl.mean())
    sigma     = float(pnl.std())
    skewness  = float(scipy_stats.skew(pnl))
    kurt      = float(scipy_stats.kurtosis(pnl))
    var_99    = float(np.percentile(pnl, 1))
    ci_low    = float(np.percentile(pnl, 2.5))
    ci_high   = float(np.percentile(pnl, 97.5))

    x_range = np.linspace(pnl.min(), pnl.max(), 600)
    scale   = len(pnl) * bin_width          # convert density → count scale

    kde_y    = scipy_stats.gaussian_kde(pnl)(x_range) * scale
    normal_y = scipy_stats.norm.pdf(x_range, mu, sigma) * scale

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker_color=bar_colors,
        name="Trades",
        showlegend=False,
        hovertemplate="$%{x:,.0f}: %{y} trades<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=kde_y,
        mode="lines", name="KDE",
        line=dict(color=_KDE, width=2.2),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_y,
        mode="lines", name="Normal",
        line=dict(color=_NORMAL, width=1.6, dash="dash"),
        fill="tozeroy", fillcolor="rgba(224,86,253,0.08)",
    ))

    skew_flag = "  ⚠" if abs(skewness) > 0.5 else ""
    kurt_flag = "  ⚠" if abs(kurt) > 1.0    else ""
    stats_ann = (
        f"μ = ${mu:,.2f}<br>"
        f"σ = ${sigma:,.2f}<br>"
        f"VaR 99% = ${var_99:,.2f}<br>"
        f"Skew = {skewness:.3f}{skew_flag}<br>"
        f"Kurt = {kurt:.3f}{kurt_flag}"
    )

    shapes = [
        dict(type="line", x0=0,       x1=0,       y0=0, y1=1, yref="paper",
             line=dict(color="black",  width=2.0)),
        dict(type="line", x0=var_99,  x1=var_99,  y0=0, y1=1, yref="paper",
             line=dict(color="#ff9f43", width=1.5, dash="dot")),
        dict(type="line", x0=mu,      x1=mu,      y0=0, y1=1, yref="paper",
             line=dict(color=_MEAN,    width=1.3, dash="dot")),
        dict(type="line", x0=ci_low,  x1=ci_low,  y0=0, y1=1, yref="paper",
             line=dict(color=_CI,      width=1.3, dash="dash")),
        dict(type="line", x0=ci_high, x1=ci_high, y0=0, y1=1, yref="paper",
             line=dict(color=_CI,      width=1.3, dash="dash")),
    ]
    annotations = [
        dict(text=f"VaR 99%<br>${var_99:,.0f}",  x=var_99,  y=0.88, yref="paper",
             xanchor="right", showarrow=False, font=dict(color="#ff9f43", size=9)),
        dict(text=f"Mean<br>${mu:,.2f}",           x=mu,      y=0.96, yref="paper",
             xanchor="left",  showarrow=False, font=dict(color=_MEAN,    size=9)),
        dict(text=f"2.5%<br>${ci_low:,.0f}",       x=ci_low,  y=0.74, yref="paper",
             xanchor="right", showarrow=False, font=dict(color=_CI,      size=9)),
        dict(text=f"97.5%<br>${ci_high:,.0f}",     x=ci_high, y=0.74, yref="paper",
             xanchor="left",  showarrow=False, font=dict(color=_CI,      size=9)),
        dict(text=stats_ann, x=0.99, y=0.99, xref="paper", yref="paper",
             xanchor="right", yanchor="top", showarrow=False,
             font=dict(color=_TEXT, size=10, family="'Courier New', monospace"),
             bgcolor=_BOX, bordercolor=_ACCENT, borderpad=6, align="left"),
    ]

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT, size=12), x=0.5),
        shapes=shapes, annotations=annotations,
        bargap=0.05,
        xaxis=dict(**_AXIS_STYLE, title="Trade PnL ($)", tickprefix="$"),
        yaxis=dict(**_AXIS_STYLE, title="Count"),
    )
    return fig


def _win_loss_figure(pnl: np.ndarray, title: str = "") -> go.Figure:
    """Overlaid KDE curves for winning and losing trades."""
    pnl   = pnl[~np.isnan(pnl)]
    wins  = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(wins) < 2 or len(losses) < 2:
        return go.Figure()

    x_min = pnl.min()
    x_max = pnl.max()
    x_range = np.linspace(x_min, x_max, 600)

    ratio   = len(wins) / len(losses)
    payoff  = wins.mean() / abs(losses.mean())

    fig = go.Figure()

    for subset, color, label in [(wins, _GREEN, "Wins"), (losses, _RED, "Losses")]:
        kde_y = scipy_stats.gaussian_kde(subset)(x_range)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_y,
            mode="lines", name=label,
            line=dict(color=color, width=2.2),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
        ))
        fig.add_vline(x=float(subset.mean()), line_color=color,
                      line_width=1.2, line_dash="dash", opacity=0.8)

    stats_ann = (
        f"Wins   n={len(wins)}   μ=${wins.mean():,.2f}   σ=${wins.std():,.2f}<br>"
        f"Losses n={len(losses)}   μ=${losses.mean():,.2f}   σ=${losses.std():,.2f}<br>"
        f"W/L ratio: {ratio:.2f}   Payoff: {payoff:.2f}"
    )

    fig.add_vline(x=0, line_color="black", line_width=2.0)
    fig.add_annotation(
        text=stats_ann, x=0.99, y=0.99, xref="paper", yref="paper",
        xanchor="right", yanchor="top", showarrow=False,
        font=dict(color=_TEXT, size=10, family="'Courier New', monospace"),
        bgcolor=_BOX, bordercolor=_ACCENT, borderpad=6, align="left",
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT, size=12), x=0.5),
        xaxis=dict(**_AXIS_STYLE, title="Trade PnL ($)", tickprefix="$"),
        yaxis=dict(**_AXIS_STYLE, title="Density"),
    )
    return fig


def _rolling_dist_figure(df: pd.DataFrame, n_periods: int, title: str = "") -> go.Figure:
    """Overlaid KDE per time period."""
    sorted_df = df.sort_values("Close time").reset_index(drop=True)
    chunks    = np.array_split(sorted_df, n_periods)

    x_all = sorted_df["Profit/Loss"].dropna().values
    x_range = np.linspace(x_all.min(), x_all.max(), 600)

    fig = go.Figure()

    for i, chunk in enumerate(chunks):
        pnl = chunk["Profit/Loss"].dropna().values
        if len(pnl) < 2:
            continue
        color = PERIOD_COLORS[i % len(PERIOD_COLORS)]
        start = chunk["Close time"].min().strftime("%Y-%m")
        end   = chunk["Close time"].max().strftime("%Y-%m")
        label = (f"P{i+1}  {start} → {end}"
                 f"   {len(pnl)} trades"
                 f"   Net: ${pnl.sum():,.0f}"
                 f"   Avg: ${pnl.mean():,.2f}")
        kde_y = scipy_stats.gaussian_kde(pnl)(x_range)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_y,
            mode="lines", name=label,
            line=dict(color=color, width=2.0),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
        ))
        fig.add_vline(x=float(pnl.mean()), line_color=color,
                      line_width=1.0, line_dash="dot", opacity=0.7)

    fig.add_vline(x=0, line_color="black", line_width=2.0)
    _base_no_legend = {k: v for k, v in _LAYOUT_BASE.items() if k not in ("legend", "margin")}
    fig.update_layout(
        **_base_no_legend,
        title=dict(text=title, font=dict(color=_TEXT, size=12), x=0.5),
        legend=dict(**_LAYOUT_BASE["legend"], orientation="v",
                    x=1.01, y=1, xanchor="left"),
        xaxis=dict(**_AXIS_STYLE, title="Trade PnL ($)", tickprefix="$"),
        yaxis=dict(**_AXIS_STYLE, title="Density"),
        margin=dict(l=8, r=240, t=32, b=8),
    )
    return fig


def _mae_mfe_figure(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Overlaid KDE for MAE vs MFE, with avg PnL marker."""
    mae_col = next((c for c in df.columns if "mae" in c.lower()), None)
    mfe_col = next((c for c in df.columns if "mfe" in c.lower()), None)
    pnl_col = "Profit/Loss"

    if mae_col is None or mfe_col is None:
        return go.Figure()

    mae  = df[mae_col].dropna().abs().values
    mfe  = df[mfe_col].dropna().abs().values
    pnl  = df[pnl_col].dropna().values

    if len(mae) < 2 or len(mfe) < 2:
        return go.Figure()

    avg_mae = float(mae.mean())
    avg_mfe = float(mfe.mean())
    avg_pnl = float(pnl.mean())
    efficiency = (avg_pnl / avg_mfe * 100) if avg_mfe > 0 else float("nan")
    captured   = (avg_pnl / avg_mae * 100) if avg_mae > 0 else float("nan")

    x_max   = max(mae.max(), mfe.max())
    x_range = np.linspace(0, x_max, 600)

    fig = go.Figure()
    for values, color, label in [(mae, _MAE_C, "MAE"), (mfe, _MFE_C, "MFE")]:
        kde_y = scipy_stats.gaussian_kde(values)(x_range)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_y,
            mode="lines", name=label,
            line=dict(color=color, width=2.2),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
        ))
        fig.add_vline(x=float(values.mean()), line_color=color,
                      line_width=1.2, line_dash="dash", opacity=0.8)

    fig.add_vline(x=avg_pnl, line_color=_MEAN, line_width=1.3, line_dash="dot")

    eff_str = f"{efficiency:.1f}%" if not math.isnan(efficiency) else "N/A"
    cap_str = f"{captured:.1f}%"  if not math.isnan(captured)   else "N/A"
    stats_ann = (
        f"Avg MAE = ${avg_mae:,.2f}<br>"
        f"Avg MFE = ${avg_mfe:,.2f}<br>"
        f"Avg PnL = ${avg_pnl:,.2f}<br>"
        f"Efficiency (PnL/MFE) = {eff_str}<br>"
        f"PnL/MAE ratio = {cap_str}"
    )
    fig.add_annotation(
        text=stats_ann, x=0.99, y=0.99, xref="paper", yref="paper",
        xanchor="right", yanchor="top", showarrow=False,
        font=dict(color=_TEXT, size=10, family="'Courier New', monospace"),
        bgcolor=_BOX, bordercolor=_ACCENT, borderpad=6, align="left",
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT, size=12), x=0.5),
        xaxis=dict(**_AXIS_STYLE, title="$ amount", tickprefix="$"),
        yaxis=dict(**_AXIS_STYLE, title="Density"),
    )
    return fig


def _rolling_breakdown_figure(df: pd.DataFrame, n_periods: int) -> go.Figure:
    """N subplots, one full distribution per period."""
    sorted_df = df.sort_values("Close time").reset_index(drop=True)
    chunks    = np.array_split(sorted_df, n_periods)
    ncols     = 2 if n_periods <= 4 else 3
    nrows     = math.ceil(n_periods / ncols)

    subplot_titles = []
    for i, chunk in enumerate(chunks):
        pnl   = chunk["Profit/Loss"].dropna()
        start = chunk["Close time"].min().strftime("%Y-%m")
        end   = chunk["Close time"].max().strftime("%Y-%m")
        net   = pnl.sum()
        win   = (pnl > 0).mean() * 100
        eq    = pnl.cumsum()
        mdd   = float((eq.cummax() - eq).max())
        subplot_titles.append(
            f"Period {i+1}  |  {start} → {end}  |  {len(pnl)} trades<br>"
            f"Net: ${net:,.0f}   Win Rate: {win:.1f}%   Max DD: ${mdd:,.0f}"
        )

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    for i, chunk in enumerate(chunks):
        row = i // ncols + 1
        col = i %  ncols + 1
        pnl   = chunk["Profit/Loss"].dropna().values
        color = PERIOD_COLORS[i % len(PERIOD_COLORS)]

        if len(pnl) < 2:
            continue

        counts, edges = np.histogram(pnl, bins=40)
        centers   = (edges[:-1] + edges[1:]) / 2
        bin_width = edges[1] - edges[0]
        bar_colors = [_GREEN if c >= 0 else _RED for c in centers]

        x_range  = np.linspace(pnl.min(), pnl.max(), 400)
        scale    = len(pnl) * bin_width
        kde_y    = scipy_stats.gaussian_kde(pnl)(x_range) * scale
        normal_y = scipy_stats.norm.pdf(x_range, pnl.mean(), pnl.std()) * scale

        fig.add_trace(go.Bar(
            x=centers, y=counts,
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="$%{x:,.0f}: %{y}<extra></extra>",
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=x_range, y=kde_y,
            mode="lines", name=f"P{i+1} KDE",
            line=dict(color=color, width=2.0),
            showlegend=False,
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=x_range, y=normal_y,
            mode="lines", name=f"P{i+1} Normal",
            line=dict(color=_NORMAL, width=1.4, dash="dash"),
            fill="tozeroy", fillcolor="rgba(224,86,253,0.07)",
            showlegend=False,
        ), row=row, col=col)

        fig.add_vline(x=0, line_color="black", line_width=1.6,
                      row=row, col=col)
        fig.add_vline(x=float(pnl.mean()), line_color=_MEAN,
                      line_width=1.2, line_dash="dot",
                      row=row, col=col)

    # Colour the subplot titles to match period colours
    for i, ann in enumerate(fig.layout.annotations[:n_periods]):
        ann.font.color = PERIOD_COLORS[i % len(PERIOD_COLORS)]
        ann.font.size  = 10

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(color=_TEXT, family="'Courier New', monospace", size=10),
        margin=dict(l=8, r=8, t=40, b=8),
        showlegend=False,
        bargap=0.05,
        hovermode="x",
        hoverlabel=dict(bgcolor=_CARD, bordercolor=_BOX, font=dict(color=_TEXT)),
    )
    fig.update_xaxes(**_AXIS_STYLE, tickprefix="$")
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _metric_card(label: str, value: str, positive: bool | None = None) -> html.Div:
    color = (_GREEN if positive is True else
             _RED   if positive is False else _TEXT)
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


def _sidebar_header(text: str) -> html.Div:
    return html.Div(text, style={
        "color": _ACCENT, "fontSize": "10px", "fontWeight": "bold",
        "letterSpacing": "0.1em", "textTransform": "uppercase",
        "marginTop": "16px", "marginBottom": "8px",
        "borderBottom": f"1px solid {_BOX}", "paddingBottom": "4px",
    })


def _sidebar_row(label: str, value: str, warn: bool = False) -> html.Div:
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between",
               "marginBottom": "8px", "fontSize": "12px"},
        children=[
            html.Span(label, style={"color": _DIM}),
            html.Span(value, style={"color": _WARN if warn else _TEXT,
                                    "fontWeight": "bold"}),
        ],
    )


def _symbol(df: pd.DataFrame) -> str | None:
    if "Symbol" not in df.columns:
        return None
    raw = str(df["Symbol"].iloc[0])
    return raw.split("_")[0] if "_" in raw else raw


def _periods_btn(n: int) -> html.Button:
    return html.Button(
        str(n), id=f"per-{n}", n_clicks=0,
        style={"backgroundColor": _BOX, "color": _TEXT, "border": f"1px solid {_BOX}",
               "borderRadius": "4px", "padding": "4px 12px", "cursor": "pointer",
               "fontSize": "12px", "fontFamily": "monospace"},
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def run_distribution_dashboard(
    strategies,
    *,
    initial_capital: float = 10_000,
    port: int = 8051,
    debug: bool = False,
) -> None:
    """
    Launch the distribution analysis dashboard in a local browser.

    Args:
        strategies:      dict {name: df} OR a single trades DataFrame.
        initial_capital: Starting equity in USD (default 10 000).
        port:            Local port (default 8051).
        debug:           Enable Dash debug/hot-reload mode.
    """
    if isinstance(strategies, pd.DataFrame):
        df   = strategies
        name = df["strategy"].iloc[0] if "strategy" in df.columns else "Strategy"
        strategies = {name: df}

    names   = list(strategies.keys())
    default = names[0]

    app = dash.Dash(__name__, title="AlphaForge — Distribution")

    _TAB_STYLE = {
        "backgroundColor": _CARD, "color": _DIM, "border": "none",
        "padding": "8px 20px", "fontSize": "12px", "fontFamily": "monospace",
    }
    _TAB_SELECTED = {**_TAB_STYLE, "color": _ACCENT,
                     "borderBottom": f"2px solid {_ACCENT}"}

    app.layout = html.Div(
        style={"backgroundColor": _BG, "minHeight": "100vh",
               "fontFamily": "'Courier New', monospace", "color": _TEXT},
        children=[

            # ── Top bar ───────────────────────────────────────────────────────
            html.Div(
                style={"backgroundColor": _CARD, "padding": "10px 24px",
                       "display": "flex", "alignItems": "center", "gap": "14px",
                       "borderBottom": f"2px solid {_BOX}"},
                children=[
                    html.Span("AlphaForge", style={"color": _ACCENT, "fontSize": "17px",
                                                    "fontWeight": "bold"}),
                    html.Span("/ distribution", style={"color": _DIM, "fontSize": "13px",
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
                    html.Span("Periods:", style={"color": _DIM, "fontSize": "12px"}),
                    html.Div(
                        style={"display": "flex", "gap": "5px"},
                        children=[_periods_btn(n) for n in [2, 3, 4, 5, 6]],
                    ),
                    dcc.Store(id="periods-store", data=4),
                ],
            ),

            # ── Metrics strip ─────────────────────────────────────────────────
            html.Div(id="metrics-strip",
                     style={"display": "flex", "gap": "1px",
                            "backgroundColor": _BG, "padding": "1px 0"}),

            # ── Tabs ──────────────────────────────────────────────────────────
            dcc.Tabs(
                id="tabs",
                value="overview",
                style={"backgroundColor": _CARD},
                children=[
                    dcc.Tab(label="Overview",          value="overview",
                            style=_TAB_STYLE, selected_style=_TAB_SELECTED),
                    dcc.Tab(label="Rolling Breakdown",  value="rolling",
                            style=_TAB_STYLE, selected_style=_TAB_SELECTED),
                ],
            ),

            # ── Tab content ───────────────────────────────────────────────────
            html.Div(id="tab-content",
                     style={"height": "calc(100vh - 175px)", "overflow": "hidden"}),
        ],
    )

    # ── Periods store ──────────────────────────────────────────────────────────
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

    # ── Main callback ──────────────────────────────────────────────────────────
    @app.callback(
        [
            Output("tab-content",   "children"),
            Output("metrics-strip", "children"),
            Output("symbol-badge",  "children"),
        ],
        [
            Input("strat-dd",     "value"),
            Input("tabs",         "value"),
            Input("periods-store","data"),
        ],
    )
    def _update(strat_name: str, tab: str, n_periods: int):
        from alphaforge.metrics import compute_metrics

        df  = strategies[strat_name].sort_values("Close time").reset_index(drop=True)
        pnl = df["Profit/Loss"].dropna().values

        # Symbol badge
        sym = _symbol(df)
        badge = html.Span(
            sym or "",
            style={"backgroundColor": _BOX, "color": _ACCENT,
                   "padding": "3px 10px", "borderRadius": "4px",
                   "fontSize": "11px", "fontWeight": "bold",
                   "letterSpacing": "0.08em"},
        ) if sym else html.Span()

        # Metrics strip
        m = compute_metrics(df, initial_capital=initial_capital)
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

        # ── Sidebar stats ─────────────────────────────────────────────────────
        skew   = float(scipy_stats.skew(pnl))
        kurt   = float(scipy_stats.kurtosis(pnl))
        var_99 = float(np.percentile(pnl, 1))
        cvar   = float(pnl[pnl <= var_99].mean()) if (pnl <= var_99).any() else float("nan")
        wins   = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        ratio  = len(wins) / len(losses) if len(losses) > 0 else float("nan")
        payoff = (wins.mean() / abs(losses.mean())
                  if len(wins) > 0 and len(losses) > 0 else float("nan"))

        mae_col = next((c for c in df.columns if "mae" in c.lower()), None)
        mfe_col = next((c for c in df.columns if "mfe" in c.lower()), None)
        eff_str = "N/A"
        if mae_col and mfe_col:
            avg_pnl  = pnl.mean()
            avg_mfe  = df[mfe_col].dropna().abs().mean()
            if avg_mfe > 0:
                eff_str = f"{avg_pnl / avg_mfe * 100:.1f}%"

        sidebar = html.Div(
            style={"width": "220px", "flexShrink": "0", "backgroundColor": _CARD,
                   "padding": "12px 16px", "borderLeft": f"2px solid {_BOX}",
                   "overflowY": "auto", "height": "100%"},
            children=[
                _sidebar_header("Distribution"),
                _sidebar_row("Trades",   f"{len(pnl):,}"),
                _sidebar_row("μ (mean)", f"${pnl.mean():,.2f}",   warn=False),
                _sidebar_row("σ (std)",  f"${pnl.std():,.2f}"),
                _sidebar_row("VaR 99%",  f"${var_99:,.2f}"),
                _sidebar_row("CVaR 99%", f"${cvar:,.2f}" if not math.isnan(cvar) else "N/A"),
                _sidebar_row("Skewness", f"{skew:.3f}",  warn=abs(skew) > 0.5),
                _sidebar_row("Kurtosis", f"{kurt:.3f}",  warn=abs(kurt) > 1.0),
                _sidebar_header("Win / Loss"),
                _sidebar_row("Win count",  f"{len(wins):,}"),
                _sidebar_row("Loss count", f"{len(losses):,}"),
                _sidebar_row("W/L ratio",  f"{ratio:.2f}" if not math.isnan(ratio) else "N/A"),
                _sidebar_row("Payoff",     f"{payoff:.2f}" if not math.isnan(payoff) else "N/A"),
                _sidebar_row("Avg win",    f"${wins.mean():,.2f}" if len(wins) else "N/A"),
                _sidebar_row("Avg loss",   f"${losses.mean():,.2f}" if len(losses) else "N/A"),
                _sidebar_header("MAE / MFE"),
                _sidebar_row("Efficiency", eff_str),
            ],
        )

        # ── Overview tab ──────────────────────────────────────────────────────
        if tab == "overview":
            strat_label = strat_name.split("/")[-1]
            sym_label   = f"  |  {sym}" if sym else ""

            content = html.Div(
                style={"display": "flex", "height": "100%"},
                children=[
                    # 2×2 chart grid
                    html.Div(
                        style={"flex": "1", "display": "grid",
                               "gridTemplateColumns": "1fr 1fr",
                               "gridTemplateRows": "1fr 1fr",
                               "gap": "4px", "padding": "4px",
                               "minWidth": "0"},
                        children=[
                            dcc.Graph(
                                figure=_hist_figure(
                                    pnl,
                                    title=f"Trade Distribution — {strat_label}{sym_label}",
                                ),
                                style={"height": "100%"},
                                config={"displayModeBar": False},
                            ),
                            dcc.Graph(
                                figure=_win_loss_figure(
                                    pnl,
                                    title=f"Win vs Loss — {strat_label}{sym_label}",
                                ),
                                style={"height": "100%"},
                                config={"displayModeBar": False},
                            ),
                            dcc.Graph(
                                figure=_rolling_dist_figure(
                                    df, n_periods,
                                    title=f"Rolling Distribution ({n_periods} periods) — {strat_label}{sym_label}",
                                ),
                                style={"height": "100%"},
                                config={"displayModeBar": False},
                            ),
                            dcc.Graph(
                                figure=_mae_mfe_figure(
                                    df,
                                    title=f"MAE / MFE — {strat_label}{sym_label}",
                                ),
                                style={"height": "100%"},
                                config={"displayModeBar": False},
                            ),
                        ],
                    ),
                    sidebar,
                ],
            )

        # ── Rolling breakdown tab ─────────────────────────────────────────────
        else:
            content = html.Div(
                style={"display": "flex", "height": "100%"},
                children=[
                    dcc.Graph(
                        figure=_rolling_breakdown_figure(df, n_periods),
                        style={"flex": "1", "height": "100%"},
                        config={"displayModeBar": False},
                    ),
                    sidebar,
                ],
            )

        return content, strip, badge

    # ── Launch ────────────────────────────────────────────────────────────────
    url = f"http://127.0.0.1:{port}"
    if not debug:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\nAlphaForge Distribution Dashboard\n  {url}\n  Ctrl+C to stop.\n")
    app.run(port=port, debug=debug)
