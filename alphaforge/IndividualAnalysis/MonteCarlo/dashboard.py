"""
mc_dashboard.py — Monte Carlo analysis dashboard for AlphaForge.

Opens on port 8051.  No computation runs on load — configure and press Run.
Progress bar updates in real time via a background thread + dcc.Interval.

Callback architecture (zero duplicate-output conflicts):
  _pick_periods  → mc-periods-store          (sole writer)
  _pick_nsims    → mc-nsims-store            (sole writer)
  _start_run     → mc-status-bar             (sole writer)
                 → mc-interval.disabled      (primary writer)
  _poll          → mc-progress-div           (sole writer)
                 → mc-interval.disabled      (allow_duplicate)
                 → mc-results-store          (sole writer)
  _render        → mc-content               (sole writer)
"""

import math
import threading

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import dash
from dash import Input, Output, State, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alphaforge.IndividualAnalysis.MonteCarlo.monte_carlo import (
    run_bootstrap, run_reshuffle, run_time_block_bootstrap,
    run_trade_block_bootstrap, run_best_trade_removal,
    run_rolling, get_equity_curves,
    rolling_summary, simulation_summary,
)

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
_MEAN   = "#ffd32a"

_CI_COLORS = {"90%": _GREEN, "95%": _WARN, "99%": _RED}

_AXIS = dict(gridcolor=_BOX, showgrid=True, zeroline=False,
             tickfont=dict(color=_DIM, size=11))
_LAYOUT = dict(
    paper_bgcolor=_BG, plot_bgcolor=_PANEL,
    font=dict(color=_TEXT, family="'Courier New', monospace", size=12),
    margin=dict(l=10, r=10, t=44, b=10),
    hovermode="closest",
    hoverlabel=dict(bgcolor=_CARD, bordercolor=_BOX, font=dict(color=_TEXT)),
    legend=dict(bgcolor=_CARD, bordercolor=_BOX, borderwidth=1,
                font=dict(color=_TEXT, size=11)),
)

_METRICS = [
    ("return_pct",   "Return %"),
    ("max_drawdown", "Max Drawdown ($)"),
    ("ret_dd",       "Return / Drawdown"),
    ("final_equity", "Final Equity ($)"),
]

_TESTS = [
    ("reshuffle",          "Trade Reshuffling"),
    ("bootstrap",          "Trade Bootstrapping"),
    ("time_block",         "Time Block Bootstrap"),
    ("trade_block",        "Trade Block Bootstrap"),
    ("best_trade_removal", "Best Trade Removal"),
]

# ── Thread-safe simulation state ───────────────────────────────────────────────
_sim_lock  = threading.Lock()
_sim_state: dict = {
    "running": False, "n_done": 0, "n_total": 0,
    "phase": "overall",  # "overall" | "rolling"
    "roll_done": 0, "roll_total": 0,
    "result": None, "error": None,
}


# Only reshuffle keeps all trades unchanged — skip would alter final equity
_NO_SKIP_TESTS = {"reshuffle"}

_MC_FN = {
    "reshuffle":          run_reshuffle,
    "bootstrap":          run_bootstrap,
    "time_block":         run_time_block_bootstrap,
    "trade_block":        run_trade_block_bootstrap,
    "best_trade_removal": run_best_trade_removal,
}


def _run_thread(df, test, n_sims, n_periods, ic):
    global _sim_state
    try:
        skip = 0.0 if test in _NO_SKIP_TESTS else 0.05
        fn   = _MC_FN[test]
        n    = len(df["Profit/Loss"].dropna())

        # ── Phase 1: Overall (tracked in progress bar) ────────────────────────
        def _overall_pf(done, _total):
            with _sim_lock:
                _sim_state["n_done"] = done

        overall_df = fn(
            df,
            n_simulations=n_sims,
            skip_trade_probability=skip,
            initial_capital=ic,
            progress_fn=_overall_pf,
        )

        # Equity curves — 100 clean sequences from full dataset, no skip
        eq_curves = get_equity_curves(
            df, test=test, n_curves=min(100, n_sims),
            skip_trade_probability=0.0, initial_capital=ic, seed=42,
        )

        # ── Phase 2: Rolling periods ───────────────────────────────────────────
        rolling_dfs = {}
        if n_periods > 1:
            roll_total = n_sims * n_periods
            with _sim_lock:
                _sim_state.update({"phase": "rolling",
                                   "roll_done": 0, "roll_total": roll_total})

            def _rolling_pf(done, _total):
                with _sim_lock:
                    _sim_state["roll_done"] = done

            rolling_dfs = run_rolling(
                df,
                n_periods=n_periods,
                test=test,
                n_simulations=n_sims,
                skip_trade_probability=skip,
                initial_capital=ic,
                progress_fn=_rolling_pf,
            )

        # Summaries
        roll_summ = rolling_summary(rolling_dfs)
        roll_dict: dict = {}
        for (period, metric), row in roll_summ.iterrows():
            roll_dict.setdefault(period, {})[metric] = {
                k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                for k, v in row.to_dict().items()
            }

        store = {
            "test":        test,
            "test_label":  dict(_TESTS).get(test, test),
            "n_sims":      n_sims,
            "n_periods":   n_periods,
            "strat":       _sim_state.get("strat", ""),
            "overall":     overall_df.to_json(orient="split"),
            "rolling":     roll_dict,
            "eq_curves":   eq_curves,
            "trade_count": n,
        }
        with _sim_lock:
            _sim_state["result"]  = store
            _sim_state["n_done"]  = n_sims
            _sim_state["phase"]   = "overall"
            _sim_state["running"] = False

    except Exception as exc:
        import traceback; traceback.print_exc()
        with _sim_lock:
            _sim_state["error"]   = str(exc)
            _sim_state["running"] = False


# ══════════════════════════════════════════════════════════════════════════════
#  Figure builders
# ══════════════════════════════════════════════════════════════════════════════

def _dist_fig(sim_df: pd.DataFrame, test_label: str, strat: str) -> go.Figure:
    titles = [m[1] for m in _METRICS]
    fig    = make_subplots(rows=2, cols=2, subplot_titles=titles,
                           vertical_spacing=0.16, horizontal_spacing=0.10)
    summ   = simulation_summary(sim_df)

    for idx, (col, label) in enumerate(_METRICS):
        row, c = idx // 2 + 1, idx % 2 + 1
        if col not in sim_df.columns:
            continue
        vals = sim_df[col].dropna().values
        if len(vals) < 2:
            continue

        # Detect constant distribution (e.g. reshuffle with no skip)
        is_const = float(np.std(vals)) < 1e-9
        val0     = float(vals[0])

        if is_const:
            # Single tall bar at the exact value
            y_top = float(len(vals)) * 1.08
            fig.add_trace(go.Bar(x=[val0], y=[len(vals)],
                                 marker_color=_ACCENT, opacity=0.7,
                                 showlegend=False,
                                 hovertemplate=f"%{{x:,.4f}}: %{{y}}<extra></extra>"),
                          row=row, col=c)
            fig.add_vline(x=val0, line_color=_MEAN, line_width=2.0,
                          line_dash="dot", row=row, col=c)
            continue

        counts, edges = np.histogram(vals, bins=60)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_w   = float(edges[1] - edges[0])
        y_top   = float(counts.max() * 1.08)

        fig.add_trace(go.Bar(x=centers, y=counts, marker_color=_ACCENT,
                             opacity=0.38, showlegend=False,
                             hovertemplate="%{x:,.2f}: %{y}<extra></extra>"),
                      row=row, col=c)
        xr    = np.linspace(vals.min(), vals.max(), 500)
        kde_y = scipy_stats.gaussian_kde(vals)(xr) * len(vals) * bin_w
        fig.add_trace(go.Scatter(x=xr, y=kde_y, mode="lines",
                                 line=dict(color=_ACCENT, width=2.2),
                                 showlegend=False), row=row, col=c)

        if col not in summ.index:
            continue
        s = summ.loc[col]

        for lo_k, hi_k, color, alpha in [
            ("ci99_lo", "ci99_hi", _CI_COLORS["99%"], 0.07),
            ("ci95_lo", "ci95_hi", _CI_COLORS["95%"], 0.12),
            ("ci90_lo", "ci90_hi", _CI_COLORS["90%"], 0.18),
        ]:
            if lo_k not in s or hi_k not in s:
                continue
            lo, hi = float(s[lo_k]), float(s[hi_k])
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            for xs in [[vals.min(), lo, lo, vals.min()],
                       [hi, vals.max(), vals.max(), hi]]:
                fig.add_trace(go.Scatter(
                    x=xs, y=[0, 0, y_top, y_top], fill="toself", mode="none",
                    fillcolor=f"rgba({r},{g},{b},{alpha})",
                    showlegend=False, hoverinfo="skip"), row=row, col=c)
            for xv in [lo, hi]:
                fig.add_vline(x=xv, line_color=color, line_width=1.1,
                              line_dash="dash", row=row, col=c)

        if "mean" in s:
            fig.add_vline(x=float(s["mean"]), line_color=_MEAN,
                          line_width=1.6, line_dash="dot", row=row, col=c)
        if "p50" in s:
            fig.add_vline(x=float(s["p50"]), line_color=_GREEN,
                          line_width=1.6, row=row, col=c)

    for ann in fig.layout.annotations:
        ann.font.color = _ACCENT
        ann.font.size  = 13

    fig.update_layout(**{**_LAYOUT, "height": 760, "bargap": 0.04,
                         "showlegend": False,
                         "title": dict(text=f"{test_label}  |  {strat}",
                                       font=dict(color=_TEXT, size=13), x=0.5)})
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS, title_text="Count", title_font=dict(size=11))
    return fig


def _forest_fig(roll_df: pd.DataFrame, metric: str, label: str) -> go.Figure:
    try:
        sub = roll_df.xs(metric, level="metric")
    except KeyError:
        return go.Figure()
    periods = list(sub.index)
    n       = len(periods)
    if n == 0:
        return go.Figure()

    fig = go.Figure()
    for (lo_k, hi_k), width, color, name in [
        (("ci99_lo", "ci99_hi"), 2,  _CI_COLORS["99%"], "99% CI"),
        (("ci95_lo", "ci95_hi"), 5,  _CI_COLORS["95%"], "95% CI"),
        (("ci90_lo", "ci90_hi"), 10, _CI_COLORS["90%"], "90% CI"),
    ]:
        xs, ys = [], []
        for i, (_, row) in enumerate(sub.iterrows()):
            if lo_k in row and hi_k in row and row[lo_k] is not None:
                xs += [float(row[lo_k]), float(row[hi_k]), None]
                ys += [i, i, None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(width=width, color=color),
                                     name=name, showlegend=False))
    if "p50" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["p50"].values, y=list(range(n)),
                                 mode="markers", showlegend=False, name="Median",
                                 marker=dict(size=12, color=_GREEN, symbol="circle",
                                             line=dict(color="#000", width=1)),
                                 customdata=periods,
                                 hovertemplate="Median %{x:,.2f}<extra>%{customdata}</extra>"))
    if "mean" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["mean"].values, y=list(range(n)),
                                 mode="markers", showlegend=False, name="Mean",
                                 marker=dict(size=10, color=_MEAN, symbol="diamond",
                                             line=dict(color="#000", width=1)),
                                 customdata=periods,
                                 hovertemplate="Mean %{x:,.2f}<extra>%{customdata}</extra>"))
    fig.add_vline(x=0, line_color="white", line_width=1.0, opacity=0.3)
    fig.update_layout(**{**_LAYOUT, "height": max(180, 72 + n * 58),
                         "title": dict(text=label, font=dict(color=_ACCENT, size=13), x=0),
                         "margin": dict(l=10, r=10, t=38, b=8),
                         "xaxis": dict(**_AXIS),
                         "yaxis": dict(tickvals=list(range(n)), ticktext=periods,
                                       tickfont=dict(color=_DIM, size=10),
                                       gridcolor=_BOX, showgrid=False,
                                       autorange="reversed"),
                         "showlegend": False})
    return fig


def _equity_fig(eq_curves: list, real_pnl: np.ndarray,
                strat: str, test_label: str) -> go.Figure:
    fig = go.Figure()
    x_sim = list(range(len(eq_curves[0]))) if eq_curves else []
    for i, curve in enumerate(eq_curves):
        fig.add_trace(go.Scatter(
            x=x_sim, y=curve, mode="lines",
            line=dict(color=_ACCENT, width=0.6), opacity=0.18,
            showlegend=(i == 0),
            name="Simulations" if i == 0 else None,
            hoverinfo="skip"))
    if len(eq_curves) >= 5:
        arr = np.array(eq_curves)
        p5, p95 = np.percentile(arr, 5, axis=0), np.percentile(arr, 95, axis=0)
        xs = list(range(arr.shape[1]))
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=p95.tolist() + p5.tolist()[::-1],
            fill="toself", mode="none",
            fillcolor="rgba(0,212,255,0.07)",
            name="90% sim band", showlegend=True, hoverinfo="skip"))
    real_eq = np.cumsum(real_pnl)
    fig.add_trace(go.Scatter(
        x=list(range(len(real_eq))), y=real_eq.tolist(),
        mode="lines", name="Real equity",
        line=dict(color=_WARN, width=2.5),
        hovertemplate="Trade %{x}: $%{y:,.0f}<extra>Real</extra>"))
    fig.add_hline(y=0, line_color="white", line_width=1.0, opacity=0.3)
    fig.update_layout(**{**_LAYOUT, "height": 520,
                         "title": dict(text=f"Equity Curves  |  {test_label}  |  {strat}",
                                       font=dict(color=_TEXT, size=13), x=0.5),
                         "xaxis": dict(**_AXIS, title="Trade index"),
                         "yaxis": dict(**_AXIS, title="Cumulative PnL ($)",
                                       tickprefix="$"),
                         "legend": dict(orientation="h", x=0, y=1.06,
                                        bgcolor="rgba(0,0,0,0)",
                                        font=dict(color=_TEXT, size=11))})
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _placeholder():
    return html.Div(
        style={"display": "flex", "flexDirection": "column",
               "alignItems": "center", "justifyContent": "center",
               "height": "60vh", "gap": "18px"},
        children=[
            html.Div("No results yet",
                     style={"color": _ACCENT, "fontSize": "24px",
                            "fontWeight": "bold"}),
            html.Div("Configure options above and press  ▶ Run",
                     style={"color": _DIM, "fontSize": "14px"}),
        ])


def _progress_ui(n_done: int, n_total: int, label: str = "Running") -> html.Div:
    pct = min(100, round(n_done / max(n_total, 1) * 100))
    return html.Div(
        style={"padding": "8px 20px"},
        children=[
            html.Div(f"{label}  {pct}%  ({n_done:,} / {n_total:,} simulations)",
                     style={"color": _TEXT, "fontSize": "13px",
                            "marginBottom": "8px"}),
            html.Div(style={"width": "100%", "height": "10px",
                            "backgroundColor": _BOX, "borderRadius": "5px",
                            "overflow": "hidden"},
                     children=[html.Div(style={
                         "width": f"{pct}%", "height": "100%",
                         "backgroundColor": _ACCENT,
                         "transition": "width 0.2s ease",
                         "borderRadius": "5px"})]),
        ])


_BTN_BASE = {"backgroundColor": _BOX, "color": _TEXT,
             "border": f"1px solid {_BOX}", "borderRadius": "4px",
             "padding": "4px 10px", "cursor": "pointer",
             "fontSize": "12px", "fontFamily": "monospace"}
_BTN_SEL  = {**_BTN_BASE, "color": "#ffd32a", "border": "1px solid #ffd32a"}


def _per_btn(n: int, selected: bool = False) -> html.Button:
    return html.Button(
        str(n), id=f"mc-per-{n}", n_clicks=0,
        style=_BTN_SEL if selected else _BTN_BASE)


def _ci_tbl(summ: pd.DataFrame) -> html.Table:
    rows = []
    for col, label in _METRICS:
        if col not in summ.index:
            continue
        s = summ.loc[col]
        rows.append(html.Tr([
            html.Td(label, style={"color": _DIM, "padding": "5px 14px",
                                  "fontSize": "12px"}),
            html.Td(f"{s.get('mean', 0):,.3f}",
                    style={"color": _TEXT, "padding": "5px 14px",
                           "fontSize": "12px", "textAlign": "right"}),
            html.Td(f"{s.get('p50', 0):,.3f}",
                    style={"color": _GREEN, "padding": "5px 14px",
                           "fontSize": "12px", "textAlign": "right"}),
            *[html.Td(f"[{s.get(lo, 0):,.3f},  {s.get(hi, 0):,.3f}]",
                      style={"color": color, "padding": "5px 14px",
                             "fontSize": "12px", "textAlign": "right"})
              for (lo, hi), color in [
                  (("ci90_lo", "ci90_hi"), _CI_COLORS["90%"]),
                  (("ci95_lo", "ci95_hi"), _CI_COLORS["95%"]),
                  (("ci99_lo", "ci99_hi"), _CI_COLORS["99%"]),
              ]],
        ]))
    hdr = html.Thead(html.Tr([
        html.Th(h, style={"color": _ACCENT, "padding": "7px 14px",
                          "fontSize": "11px", "textAlign": aj,
                          "borderBottom": f"1px solid {_BOX}",
                          "textTransform": "uppercase"})
        for h, aj in [("Metric", "left"), ("Mean", "right"), ("Median", "right"),
                      ("90% CI", "right"), ("95% CI", "right"), ("99% CI", "right")]
    ]))
    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse",
               "backgroundColor": _CARD, "borderRadius": "6px",
               "marginTop": "10px"},
        children=[hdr, html.Tbody(rows)])


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_mc_dashboard(
    strategies,
    *,
    initial_capital: float = 10_000,
    port: int = 8051,
    debug: bool = False,
) -> None:
    """Launch the Monte Carlo dashboard on port 8051."""
    if isinstance(strategies, pd.DataFrame):
        df   = strategies
        name = df["strategy"].iloc[0] if "strategy" in df.columns else "Strategy"
        strategies = {name: df}

    if len(strategies) > 1:
        combined   = pd.concat(strategies.values(), ignore_index=True)
        strategies = {"★ Portfolio (All)": combined, **strategies}

    names   = list(strategies.keys())
    default = names[0]

    app = dash.Dash(__name__, title="AlphaForge | Monte Carlo")

    _TAB  = {"backgroundColor": _CARD, "color": _DIM, "border": "none",
             "padding": "8px 20px", "fontSize": "12px", "fontFamily": "monospace"}
    _TSEL = {**_TAB, "color": _ACCENT, "borderBottom": f"2px solid {_ACCENT}"}

    _dd_style = {"fontSize": "12px", "color": "#111"}

    # ── Layout ────────────────────────────────────────────────────────────────
    app.layout = html.Div(
        style={"backgroundColor": _BG, "minHeight": "100vh",
               "fontFamily": "'Courier New', monospace", "color": _TEXT},
        children=[

            # Top bar
            html.Div(
                style={"backgroundColor": _CARD, "padding": "10px 20px",
                       "display": "flex", "alignItems": "center", "gap": "10px",
                       "borderBottom": f"2px solid {_BOX}", "flexWrap": "wrap"},
                children=[
                    html.Span("AlphaForge | Monte Carlo",
                              style={"color": _ACCENT, "fontSize": "15px",
                                     "fontWeight": "bold", "whiteSpace": "nowrap",
                                     "marginRight": "8px"}),

                    dcc.Dropdown(id="mc-strat-dd", clearable=False,
                                 options=[{"label": n, "value": n} for n in names],
                                 value=default,
                                 style={"width": "240px", **_dd_style}),

                    dcc.Dropdown(id="mc-test-dd", clearable=False,
                                 options=[{"label": lbl, "value": val}
                                          for val, lbl in _TESTS],
                                 value="reshuffle",
                                 style={"width": "215px", **_dd_style}),

                    # N-sims: preset dropdown
                    dcc.Dropdown(
                        id="mc-nsim-dd", clearable=False,
                        options=[
                            {"label": "500  — quick",  "value": 500},
                            {"label": "1 000",         "value": 1_000},
                            {"label": "5 000",         "value": 5_000},
                            {"label": "10 000  — full","value": 10_000},
                            {"label": "Custom…",       "value": 0},
                        ],
                        value=1_000,
                        style={"width": "175px", **_dd_style},
                    ),

                    # Custom N-sims text field (always visible, enabled on demand)
                    dcc.Input(
                        id="mc-nsim-custom", type="number",
                        min=100, max=100_000, step=100,
                        placeholder="# sims", debounce=True,
                        style={"width": "95px", "fontSize": "12px",
                               "backgroundColor": _PANEL, "color": _TEXT,
                               "border": f"1px solid {_BOX}",
                               "borderRadius": "4px", "padding": "6px 8px",
                               "fontFamily": "monospace",
                               "opacity": "0.4", "cursor": "not-allowed"},
                    ),

                    # Divider + periods
                    html.Div(style={"width": "1px", "height": "28px",
                                    "backgroundColor": _BOX, "margin": "0 8px"}),
                    html.Span("Periods:",
                              style={"color": _DIM, "fontSize": "12px",
                                     "whiteSpace": "nowrap"}),
                    *[_per_btn(n, selected=(n == 4)) for n in [1, 2, 3, 4, 5, 6]],

                    # Stores + interval
                    dcc.Store(id="mc-periods-store", data=4),
                    dcc.Store(id="mc-nsims-store",   data=1_000),
                    dcc.Store(id="mc-results-store", data=None),
                    dcc.Interval(id="mc-interval", interval=300,
                                 disabled=True, n_intervals=0),

                    html.Div(style={"flex": "1"}),

                    html.Button(
                        "▶  Run", id="mc-run-btn", n_clicks=0,
                        style={"backgroundColor": _GREEN, "color": "#000",
                               "border": "none", "borderRadius": "6px",
                               "padding": "8px 26px", "cursor": "pointer",
                               "fontSize": "13px", "fontFamily": "monospace",
                               "fontWeight": "bold", "whiteSpace": "nowrap"}),
                ],
            ),

            # Status bar
            html.Div(id="mc-status-bar",
                     style={"backgroundColor": _PANEL, "padding": "5px 20px",
                            "borderBottom": f"1px solid {_BOX}",
                            "fontSize": "12px", "color": _DIM, "minHeight": "26px"},
                     children="Ready — configure options and press Run."),

            # Progress bar (separate from content — sole responsibility of _poll)
            html.Div(id="mc-progress-div",
                     style={"backgroundColor": _PANEL,
                            "borderBottom": f"1px solid {_BOX}"}),

            # Tabs
            dcc.Tabs(id="mc-tabs", value="dist",
                     style={"backgroundColor": _CARD},
                     children=[
                         dcc.Tab(label="Distribution",    value="dist",
                                 style=_TAB, selected_style=_TSEL),
                         dcc.Tab(label="Rolling Periods", value="rolling",
                                 style=_TAB, selected_style=_TSEL),
                         dcc.Tab(label="Equity Curves",   value="equity",
                                 style=_TAB, selected_style=_TSEL),
                         dcc.Tab(label="Summary Table",   value="table",
                                 style=_TAB, selected_style=_TSEL),
                     ]),

            # Content (sole responsibility of _render)
            html.Div(id="mc-content", style={"padding": "12px"},
                     children=[_placeholder()]),
        ],
    )

    # ── 1. Period buttons ──────────────────────────────────────────────────────
    @app.callback(
        [Output("mc-periods-store", "data"),
         Output("mc-per-1", "style"),
         Output("mc-per-2", "style"),
         Output("mc-per-3", "style"),
         Output("mc-per-4", "style"),
         Output("mc-per-5", "style"),
         Output("mc-per-6", "style")],
        [Input("mc-per-1", "n_clicks"),
         Input("mc-per-2", "n_clicks"),
         Input("mc-per-3", "n_clicks"),
         Input("mc-per-4", "n_clicks"),
         Input("mc-per-5", "n_clicks"),
         Input("mc-per-6", "n_clicks")],
        prevent_initial_call=True,
    )
    def _pick_periods(*_):
        tid = dash.ctx.triggered_id
        if tid is None:
            return [dash.no_update] * 7
        selected = int(str(tid).replace("mc-per-", ""))
        def _s(n):
            base = {"backgroundColor": _BOX, "borderRadius": "4px",
                    "padding": "4px 10px", "cursor": "pointer",
                    "fontSize": "12px", "fontFamily": "monospace"}
            if n == selected:
                return {**base, "color": "#ffd32a", "border": "1px solid #ffd32a"}
            return {**base, "color": _TEXT, "border": f"1px solid {_BOX}"}
        return [selected, _s(1), _s(2), _s(3), _s(4), _s(5), _s(6)]

    # ── 2. N-sims (dropdown + custom input) ───────────────────────────────────
    @app.callback(
        [Output("mc-nsims-store",  "data"),
         Output("mc-nsim-custom",  "style")],
        [Input("mc-nsim-dd",       "value"),
         Input("mc-nsim-custom",   "value")],
        prevent_initial_call=True,
    )
    def _pick_nsims(dd_val, custom_val):
        enabled  = {"width": "95px", "fontSize": "12px",
                    "backgroundColor": _PANEL, "color": _TEXT,
                    "border": f"1px solid {_ACCENT}",
                    "borderRadius": "4px", "padding": "6px 8px",
                    "fontFamily": "monospace", "opacity": "1",
                    "cursor": "text"}
        disabled = {**enabled, "border": f"1px solid {_BOX}",
                    "opacity": "0.4", "cursor": "not-allowed"}

        ctx  = dash.callback_context
        trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        if "mc-nsim-custom" in trig:
            v = int(custom_val) if custom_val and int(custom_val) >= 100 else dash.no_update
            return v, enabled

        # dropdown triggered
        if dd_val == 0:   # "Custom…"
            return dash.no_update, enabled
        return int(dd_val), disabled

    # ── 3. Run button → start thread (sole writer: mc-status-bar, mc-interval) ─
    @app.callback(
        [Output("mc-status-bar",  "children"),
         Output("mc-interval",    "disabled"),
         Output("mc-interval",    "n_intervals")],
        Input("mc-run-btn",        "n_clicks"),
        [State("mc-strat-dd",      "value"),
         State("mc-test-dd",       "value"),
         State("mc-nsims-store",   "data"),
         State("mc-periods-store", "data")],
        prevent_initial_call=True,
    )
    def _start_run(_, strat_name, test, n_sims, n_periods):
        global _sim_state
        if _sim_state.get("running"):
            return ("Already running — wait for it to finish.",
                    False, dash.no_update)

        n_sims    = int(n_sims    or 1_000)
        n_periods = int(n_periods or 4)
        df        = strategies[strat_name].sort_values("Close time").reset_index(drop=True)

        with _sim_lock:
            _sim_state.update({
                "running": True, "n_done": 0, "n_total": n_sims,
                "phase": "overall", "roll_done": 0, "roll_total": 0,
                "result": None, "error": None,
                "strat": strat_name,
            })

        threading.Thread(
            target=_run_thread,
            args=(df, test, n_sims, n_periods, initial_capital),
            daemon=True,
        ).start()

        test_lbl = dict(_TESTS).get(test, test)
        return (f"Running  {test_lbl}  |  {strat_name}  |  "
                f"{n_sims:,} sims × {n_periods} periods …",
                False,   # enable interval
                0)       # reset n_intervals so _poll fires from 0

    # ── 4. Poll → progress bar + completion (sole writer: mc-progress-div,
    #                                         mc-results-store;
    #                                         allow_dup: mc-interval.disabled) ──
    @app.callback(
        [Output("mc-progress-div",   "children"),
         Output("mc-interval",       "disabled", allow_duplicate=True),
         Output("mc-results-store",  "data")],
        Input("mc-interval",         "n_intervals"),
        prevent_initial_call=True,
    )
    def _poll(_):
        with _sim_lock:
            running    = _sim_state["running"]
            n_done     = _sim_state["n_done"]
            n_total    = _sim_state["n_total"]
            phase      = _sim_state.get("phase", "overall")
            roll_done  = _sim_state.get("roll_done", 0)
            roll_total = _sim_state.get("roll_total", 0)
            result     = _sim_state["result"]
            error      = _sim_state.get("error")

        if error:
            return (html.Div(f"Error: {error}",
                             style={"color": _RED, "padding": "8px 20px",
                                    "fontSize": "12px"}),
                    True, dash.no_update)

        if running:
            if phase == "rolling" and roll_total > 0:
                pct = min(100, round(roll_done / roll_total * 100))
                bar = html.Div(
                    style={"padding": "8px 20px"},
                    children=[
                        html.Div(f"Rolling periods  {pct}%  ({roll_done:,} / {roll_total:,} sims)",
                                 style={"color": _TEXT, "fontSize": "13px",
                                        "marginBottom": "8px"}),
                        html.Div(style={"width": "100%", "height": "10px",
                                        "backgroundColor": _BOX, "borderRadius": "5px",
                                        "overflow": "hidden"},
                                 children=[html.Div(style={
                                     "width": f"{pct}%", "height": "100%",
                                     "backgroundColor": _WARN,
                                     "transition": "width 0.2s ease",
                                     "borderRadius": "5px"})]),
                    ])
                return bar, False, dash.no_update
            return _progress_ui(n_done, n_total), False, dash.no_update

        if result is not None:
            with _sim_lock:
                _sim_state["result"] = None
            return html.Div(), True, result

        return dash.no_update, True, dash.no_update

    # ── 5. Render results (sole writer: mc-content) ───────────────────────────
    @app.callback(
        Output("mc-content", "children"),
        [Input("mc-results-store", "data"),
         Input("mc-tabs",          "value")],
    )
    def _render(store, tab):
        if store is None:
            return [_placeholder()]

        overall_df = pd.read_json(store["overall"], orient="split")
        test_lbl   = store.get("test_label", store["test"])
        strat      = store["strat"]

        # Distribution ─────────────────────────────────────────────────────────
        if tab == "dist":
            legend = html.Div(
                style={"display": "flex", "gap": "4px", "alignItems": "center",
                       "padding": "5px 10px", "backgroundColor": _CARD,
                       "borderRadius": "4px", "marginBottom": "8px",
                       "flexWrap": "wrap"},
                children=[
                    *[x for sym, col, lbl in [
                        ("━━━━━━━", _CI_COLORS["90%"], "90% CI"),
                        ("━━━━",   _CI_COLORS["95%"], "95% CI"),
                        ("━━",     _CI_COLORS["99%"], "99% CI"),
                        ("━━",     _MEAN,             "Mean (dashed)"),
                        ("━━",     _GREEN,            "Median"),
                    ] for x in [
                        html.Span(sym, style={"color": col, "fontSize": "14px"}),
                        html.Span(f" {lbl}  ",
                                  style={"color": _DIM, "fontSize": "12px"}),
                    ]],
                ])
            return [legend,
                    dcc.Graph(figure=_dist_fig(overall_df, test_lbl, strat),
                              config={"displayModeBar": False}),
                    _ci_tbl(simulation_summary(overall_df))]

        # Rolling ──────────────────────────────────────────────────────────────
        elif tab == "rolling":
            roll_dict = store.get("rolling", {})
            if not roll_dict:
                return [html.Div("No rolling data.", style={"color": _DIM})]
            rows = [{"period": p, "metric": m, **{k: v for k, v in s.items()
                                                   if v is not None}}
                    for p, md in roll_dict.items()
                    for m, s in md.items()]
            if not rows:
                return [html.Div("No rolling data.", style={"color": _DIM})]
            roll_df = pd.DataFrame(rows).set_index(["period", "metric"])
            charts = []
            for col, label in _METRICS:
                charts.append(dcc.Graph(
                    figure=_forest_fig(roll_df, col, label),
                    config={"displayModeBar": False},
                    style={"marginBottom": "4px"}))
            return charts

        # Equity curves ────────────────────────────────────────────────────────
        elif tab == "equity":
            eq  = store.get("eq_curves", [])
            df_ = strategies[strat].sort_values("Close time").reset_index(drop=True)
            rpnl = df_["Profit/Loss"].dropna().values
            if not eq:
                return [html.Div("No equity curves stored.",
                                 style={"color": _DIM})]
            return [
                html.Div(f"{len(eq)} simulation curves  |  "
                         f"{store['trade_count']:,} trades  |  {test_lbl}",
                         style={"color": _DIM, "fontSize": "12px",
                                "marginBottom": "8px"}),
                dcc.Graph(figure=_equity_fig(eq, rpnl, strat, test_lbl),
                          config={"displayModeBar": False}),
            ]

        # Summary table ────────────────────────────────────────────────────────
        else:
            summ   = simulation_summary(overall_df)
            cols   = ["mean", "std", "p50", "ci90_lo", "ci90_hi",
                      "ci95_lo", "ci95_hi", "ci99_lo", "ci99_hi", "min", "max"]
            exist  = [c for c in cols if c in summ.columns]

            def _cc(k):
                if "99" in k: return _CI_COLORS["99%"]
                if "95" in k: return _CI_COLORS["95%"]
                if "90" in k: return _CI_COLORS["90%"]
                if k == "p50": return _GREEN
                return _TEXT

            th = {"color": _ACCENT, "padding": "7px 14px",
                  "borderBottom": f"1px solid {_BOX}",
                  "fontSize": "11px", "textTransform": "uppercase"}
            data_rows = []
            for col, label in _METRICS:
                if col not in summ.index:
                    continue
                s = summ.loc[col]
                data_rows.append(html.Tr([
                    html.Td(label, style={"color": _DIM, "padding": "6px 14px",
                                          "fontSize": "12px",
                                          "borderBottom": f"1px solid {_BOX}"}),
                    *[html.Td(f"{s[c]:,.4f}" if c in s and s[c] is not None else "—",
                              style={"color": _cc(c), "padding": "6px 14px",
                                     "fontSize": "12px", "textAlign": "right",
                                     "borderBottom": f"1px solid {_BOX}"})
                      for c in exist],
                ]))
            return [
                html.Div(f"{test_lbl}  |  {strat}  |  {store['n_sims']:,} sims",
                         style={"color": _DIM, "fontSize": "12px",
                                "marginBottom": "10px"}),
                html.Table(
                    style={"width": "100%", "borderCollapse": "collapse",
                           "backgroundColor": _CARD, "borderRadius": "6px"},
                    children=[
                        html.Thead(html.Tr(
                            [html.Th("Metric", style=th)] +
                            [html.Th(c, style={**th, "textAlign": "right"})
                             for c in exist])),
                        html.Tbody(data_rows),
                    ]),
            ]

    # ── Launch ────────────────────────────────────────────────────────────────
    url = f"http://127.0.0.1:{port}"
    print(f"\nMonte Carlo Dashboard\n  {url}\n  Ctrl+C to stop.\n")
    app.run(port=port, debug=debug)
