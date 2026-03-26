"""
dashboard.py — Portfolio Explorer dashboard.

Two-tab Dash app for browsing and comparing generated portfolios.

Tab 1 — Portfolio View:
  Dropdown to select a portfolio (sorted by Sharpe).
  Left panel : strategies, weights, risk/trade, scale factor, critical day, MAE stress.
  Right panel: equity curve (before/after rescaling + red critical day dot) + drawdown.

Tab 2 — Ranking:
  Sortable table of all valid portfolios with key metrics + MAE stress status.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.validator import ValidPortfolio


# ── Theme ─────────────────────────────────────────────────────────────────────

BG      = "#0f1117"
PANEL   = "#1a1a2e"
BORDER  = "#2a2a4a"
CYAN    = "#00d4ff"
AMBER   = "#f0a500"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
TEXT    = "#e0e0e0"
DIM     = "#888888"
BEFORE  = "#4a4a7a"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _equity_curve(portfolio_df: pd.DataFrame) -> pd.Series:
    daily = (
        portfolio_df
        .groupby(portfolio_df["Close time"].dt.date)["Profit/Loss"]
        .sum()
    )
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(date_range, fill_value=0.0).cumsum()


def _drawdown_pct(equity: pd.Series, initial_capital: float) -> pd.Series:
    total = initial_capital + equity
    peak  = total.cummax()
    return ((peak - total) / peak * 100)


def _short(name: str) -> str:
    return name.split("/")[-1]


# ── Portfolio label for dropdown ──────────────────────────────────────────────

def _portfolio_label(vp: ValidPortfolio, rank: int) -> str:
    method = vp.method.replace("_", " ").title()
    n      = len(vp.combination)
    stress = ""
    if vp.stress is not None:
        stress = " ✓ MAE" if vp.stress.passed else " ✗ MAE"
    return f"#{rank}  {method} | {n} strats | Sharpe {vp.sharpe:.2f}{stress}"


# ── Equity + Drawdown figure ──────────────────────────────────────────────────

def _build_chart(vp: ValidPortfolio, initial_capital: float) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
    )

    # Before-scaling equity (divide rescaled P&L by scale factor)
    before_df = vp.portfolio_df.copy()
    if vp.scale_factor != 0:
        before_df["Profit/Loss"] = before_df["Profit/Loss"] / vp.scale_factor
    eq_before = _equity_curve(before_df)
    eq_after  = _equity_curve(vp.portfolio_df)
    dd        = _drawdown_pct(eq_after, initial_capital)

    x_before = eq_before.index.astype(str).tolist()
    x_after  = eq_after.index.astype(str).tolist()

    # Before (dashed, muted)
    fig.add_trace(go.Scatter(
        x=x_before, y=eq_before.values.tolist(),
        mode="lines", name="Before rescaling",
        line=dict(color=BEFORE, width=1.5, dash="dash"),
        opacity=0.7,
    ), row=1, col=1)

    # After (solid, bright)
    fig.add_trace(go.Scatter(
        x=x_after, y=eq_after.values.tolist(),
        mode="lines", name="After rescaling",
        line=dict(color=CYAN, width=2),
    ), row=1, col=1)

    # Critical day dot
    crit_ts  = str(pd.Timestamp(vp.critical_day).date())
    crit_val = eq_after.get(pd.Timestamp(vp.critical_day), None)
    if crit_val is not None:
        fig.add_trace(go.Scatter(
            x=[crit_ts], y=[float(crit_val)],
            mode="markers",
            name=f"Critical day ({vp.critical_day})",
            marker=dict(color=RED, size=10, symbol="circle"),
            hovertemplate=(
                f"<b>Critical Day</b><br>"
                f"Date: {vp.critical_day}<br>"
                f"Daily loss: ${vp.worst_day_loss:,.2f}<extra></extra>"
            ),
        ), row=1, col=1)

    # MAE stress critical day dot (if available)
    if vp.stress is not None:
        mae_ts  = str(pd.Timestamp(vp.stress.critical_day_mae).date())
        mae_val = eq_after.get(pd.Timestamp(vp.stress.critical_day_mae), None)
        if mae_val is not None:
            fig.add_trace(go.Scatter(
                x=[mae_ts], y=[float(mae_val)],
                mode="markers",
                name=f"MAE critical ({vp.stress.critical_day_mae})",
                marker=dict(color=AMBER, size=10, symbol="diamond"),
                hovertemplate=(
                    f"<b>MAE Critical Day</b><br>"
                    f"Date: {vp.stress.critical_day_mae}<br>"
                    f"Daily loss (MAE): ${vp.stress.worst_day_loss_mae:,.2f}<extra></extra>"
                ),
            ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=x_after, y=dd.values.tolist(),
        mode="lines", name="Drawdown %",
        line=dict(color=RED, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.15)",
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(
        y=0, line_color=BORDER, line_width=1, row=1, col=1,
    )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=11),
        legend=dict(
            bgcolor=PANEL, bordercolor=BORDER, borderwidth=1,
            font=dict(size=10), orientation="h", y=1.04,
        ),
        margin=dict(l=60, r=20, t=30, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, zeroline=False)
    fig.update_yaxes(title_text="Cumulative P&L ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", autorange="reversed", row=2, col=1)

    return fig


# ── Info panel (left side) ────────────────────────────────────────────────────

def _info_panel(vp: ValidPortfolio) -> html.Div:
    method = vp.method.replace("_", " ").title()
    m      = vp.metrics

    def row(label, value, color=TEXT):
        return html.Tr([
            html.Td(label, style={"color": DIM, "padding": "3px 8px", "fontSize": "12px"}),
            html.Td(value, style={"color": color, "padding": "3px 8px", "fontSize": "12px", "textAlign": "right"}),
        ])

    # Strategies + weights + risk/trade
    strat_rows = []
    for name in vp.combination:
        w   = vp.weights[name]
        rpt = vp.risk_per_trade[name]
        strat_rows.append(html.Tr([
            html.Td(_short(name), style={"color": CYAN,  "padding": "2px 8px", "fontSize": "11px"}),
            html.Td(f"{w:.3f}",   style={"color": TEXT,  "padding": "2px 8px", "fontSize": "11px", "textAlign": "right"}),
            html.Td(f"${rpt:,.0f}", style={"color": AMBER, "padding": "2px 8px", "fontSize": "11px", "textAlign": "right"}),
        ]))

    # MAE stress block
    if vp.stress is not None:
        s = vp.stress
        stress_color  = GREEN if s.passed else RED
        stress_label  = "PASSED" if s.passed else "FAILED"
        stress_block  = [
            html.Hr(style={"borderColor": BORDER, "margin": "8px 0"}),
            html.P("MAE Stress Test", style={"color": DIM, "fontSize": "11px", "margin": "4px 0"}),
            html.Table([
                row("Status",          stress_label,                       stress_color),
                row("Worst day (MAE)", f"${s.worst_day_loss_mae:,.2f}",   RED if s.worst_day_loss_mae < 0 else GREEN),
                row("Max DD (MAE)",    f"${s.max_drawdown_usd_mae:,.0f}"),
                row("Max DD% (MAE)",   f"{s.max_drawdown_pct_mae:.2f}%"),
                row("MAE coverage",    f"{s.mae_coverage_pct:.0f}% of trades"),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ]
    else:
        stress_block = [html.P("MAE stress not run yet.", style={"color": DIM, "fontSize": "11px"})]

    return html.Div([
        html.P(f"Method: {method}", style={"color": CYAN, "fontWeight": "bold", "marginBottom": "4px"}),

        html.Table([
            html.Tr([
                html.Th("Strategy",   style={"color": DIM, "fontSize": "11px", "padding": "2px 8px", "textAlign": "left"}),
                html.Th("Weight",     style={"color": DIM, "fontSize": "11px", "padding": "2px 8px", "textAlign": "right"}),
                html.Th("Risk/Trade", style={"color": DIM, "fontSize": "11px", "padding": "2px 8px", "textAlign": "right"}),
            ])
        ] + strat_rows, style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "8px"}),

        html.Hr(style={"borderColor": BORDER, "margin": "8px 0"}),

        html.Table([
            row("Scale factor",    f"×{vp.scale_factor:.4f}"),
            row("Critical day",    str(vp.critical_day)),
            row("Worst day loss",  f"${vp.worst_day_loss:,.2f}", RED),
            html.Tr([html.Td(colSpan=2)]),
            row("Total profit",    f"${m['total_profit']:,.0f}", GREEN),
            row("CAGR",            f"{m['cagr']:.2f}%"),
            row("Sharpe",          f"{m['sharpe_ratio']:.3f}", CYAN),
            row("Profit factor",   f"{m['profit_factor']:.2f}"),
            row("Return/DD",       f"{m['return_dd_ratio']:.2f}"),
            row("Win rate",        f"{m['winning_percentage']:.1f}%"),
            row("Max DD ($)",      f"${m['drawdown']:,.0f}", RED),
            row("Max DD (%)",      f"{m['pct_drawdown']:.2f}%", RED),
            row("Avg trade",       f"${m['average_trade']:.2f}"),
            row("Num trades",      f"{m['num_trades']:,}"),
        ], style={"width": "100%", "borderCollapse": "collapse"}),

        *stress_block,
    ], style={"padding": "12px"})


# ── Ranking table ─────────────────────────────────────────────────────────────

def _ranking_data(portfolios: list[ValidPortfolio]) -> list[dict]:
    rows = []
    for i, vp in enumerate(portfolios, 1):
        m   = vp.metrics
        stress_ok = (
            "✓" if (vp.stress and vp.stress.passed) else
            "✗" if (vp.stress and not vp.stress.passed) else "—"
        )
        rows.append({
            "Rank":          i,
            "Method":        vp.method.replace("_", " ").title(),
            "N":             len(vp.combination),
            "Strategies":    " | ".join(_short(n) for n in vp.combination),
            "Sharpe":        round(m["sharpe_ratio"], 3),
            "CAGR %":        round(m["cagr"], 2),
            "Profit ($)":    round(m["total_profit"], 0),
            "Max DD %":      round(m["pct_drawdown"], 2),
            "Profit Factor": round(m["profit_factor"], 2),
            "Return/DD":     round(m["return_dd_ratio"], 2),
            "Win %":         round(m["winning_percentage"], 1),
            "Scale ×":       round(vp.scale_factor, 4),
            "MAE Stress":    stress_ok,
        })
    return rows


# ── App builder ───────────────────────────────────────────────────────────────

def run_portfolio_dashboard(
    portfolios: list[ValidPortfolio],
    config: PortfolioConfig,
    port: int = 8060,
) -> None:
    """
    Launch the Portfolio Explorer Dash app.

    Args:
        portfolios : list of ValidPortfolio from the pipeline (sorted by Sharpe)
        config     : PortfolioConfig (for account_balance etc.)
        port       : local port to serve on (default 8060)
    """
    if not portfolios:
        print("  No valid portfolios to display.")
        return

    dropdown_options = [
        {"label": _portfolio_label(vp, i + 1), "value": i}
        for i, vp in enumerate(portfolios)
    ]

    ranking_rows    = _ranking_data(portfolios)
    ranking_columns = [{"name": c, "id": c} for c in ranking_rows[0].keys()]

    app = dash.Dash(__name__, title="AlphaForge — Portfolio Explorer")

    app.layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "monospace"}, children=[

        # Header
        html.Div(style={"backgroundColor": PANEL, "padding": "12px 24px", "borderBottom": f"1px solid {BORDER}"}, children=[
            html.H2("AlphaForge — Portfolio Explorer", style={"color": CYAN, "margin": 0, "fontSize": "18px"}),
            html.Span(f"{len(portfolios)} valid portfolios", style={"color": DIM, "fontSize": "12px"}),
        ]),

        # Tabs
        dcc.Tabs(
            style={"backgroundColor": PANEL},
            colors={"border": BORDER, "primary": CYAN, "background": PANEL},
            children=[

                # ── Tab 1: Portfolio View ──────────────────────────────────
                dcc.Tab(label="Portfolio View", style={"color": DIM}, selected_style={"color": CYAN, "backgroundColor": BG}, children=[
                    html.Div(style={"padding": "12px 24px"}, children=[

                        # Dropdown
                        html.Div(style={"marginBottom": "12px"}, children=[
                            html.Label("Select portfolio:", style={"color": DIM, "fontSize": "12px", "marginRight": "8px"}),
                            dcc.Dropdown(
                                id="portfolio-dropdown",
                                options=dropdown_options,
                                value=0,
                                clearable=False,
                                style={
                                    "width": "600px", "display": "inline-block",
                                    "backgroundColor": PANEL, "color": TEXT,
                                    "border": f"1px solid {BORDER}",
                                },
                            ),
                        ]),

                        # Main layout: left info + right chart
                        html.Div(style={"display": "flex", "gap": "16px"}, children=[

                            # Left info panel
                            html.Div(
                                id="info-panel",
                                style={
                                    "width": "320px", "flexShrink": "0",
                                    "backgroundColor": PANEL,
                                    "border": f"1px solid {BORDER}",
                                    "borderRadius": "6px",
                                    "overflowY": "auto",
                                    "maxHeight": "680px",
                                },
                            ),

                            # Right chart
                            html.Div(style={"flexGrow": "1"}, children=[
                                dcc.Graph(
                                    id="equity-chart",
                                    style={"height": "680px"},
                                    config={"displayModeBar": False},
                                ),
                            ]),
                        ]),
                    ]),
                ]),

                # ── Tab 2: Ranking ─────────────────────────────────────────
                dcc.Tab(label="Ranking", style={"color": DIM}, selected_style={"color": CYAN, "backgroundColor": BG}, children=[
                    html.Div(style={"padding": "24px"}, children=[
                        html.P("All valid portfolios sorted by Sharpe ratio. Click a column header to sort.",
                               style={"color": DIM, "fontSize": "12px", "marginBottom": "12px"}),
                        dash_table.DataTable(
                            id="ranking-table",
                            columns=ranking_columns,
                            data=ranking_rows,
                            sort_action="native",
                            filter_action="native",
                            page_size=25,
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": PANEL,
                                "color": CYAN,
                                "fontWeight": "bold",
                                "border": f"1px solid {BORDER}",
                                "fontSize": "11px",
                            },
                            style_cell={
                                "backgroundColor": BG,
                                "color": TEXT,
                                "border": f"1px solid {BORDER}",
                                "fontSize": "11px",
                                "fontFamily": "monospace",
                                "padding": "6px 10px",
                            },
                            style_data_conditional=[
                                {"if": {"filter_query": '{MAE Stress} = "✓"'},
                                 "color": GREEN},
                                {"if": {"filter_query": '{MAE Stress} = "✗"'},
                                 "color": RED},
                            ],
                        ),
                    ]),
                ]),
            ],
        ),
    ])

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output("equity-chart", "figure"),
        Output("info-panel",   "children"),
        Input("portfolio-dropdown", "value"),
    )
    def update_portfolio(idx):
        vp  = portfolios[idx]
        fig = _build_chart(vp, config.account_balance)
        info = _info_panel(vp)
        return fig, info

    print(f"\n  Portfolio Explorer running at http://localhost:{port}\n")
    app.run(debug=False, port=port)
