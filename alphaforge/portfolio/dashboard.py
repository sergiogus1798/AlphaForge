"""
dashboard.py — Portfolio Explorer dashboard (matplotlib, local windows).

Two windows:
  Window 1 — Combination Explorer:
    Navigate through the top-N strategy combinations (< Prev / Next >).
    Left panel  : scrollable visual tables (mouse-wheel) — strategies+asset,
                  weights, risk/trade (+ total), scale, DD status.
    Right top   : 4 equity curves on the same axes (one colour per method).
                  DD-failed methods shown dashed with reduced opacity.
    Right bottom: metrics comparison table (rows = metrics, cols = 4 methods).

  Window 2 — Ranking overview:
    One row per combination, columns show each method's Sharpe (coloured by method).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.combo_result import (
    CombinationResult, METHODS, METHOD_COLORS, METHOD_LABELS,
)
from alphaforge.portfolio.generator.wf import WF_METHODS


# ── Theme ──────────────────────────────────────────────────────────────────────

BG       = "#0f1117"
AX_BG    = "#1a1a2e"
BORDER   = "#2a2a4a"
CYAN     = "#00d4ff"
AMBER    = "#f0a500"
GREEN    = "#2ecc71"
RED      = "#e74c3c"
TEXT     = "#e0e0e0"
DIM      = "#888888"
ROW_A    = "#0d1020"
ROW_B    = "#111828"
SEC_BG   = "#1e2240"
COL_HDR  = "#141830"

# Full-length names for metrics table columns
_METHOD_FULL = {
    "equal":        "Equal Weight",
    "min_variance": "Min Variance",
    "risk_parity":  "Risk Parity",
    "hrp":          "HRP",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _equity(portfolio_df: pd.DataFrame) -> pd.Series:
    daily = (
        portfolio_df
        .groupby(portfolio_df["Close time"].dt.date)["Profit/Loss"]
        .sum()
    )
    dr = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(dr, fill_value=0.0).cumsum()


def _short(name: str) -> str:
    return name.split("/")[-1]


def _asset(name: str, strategies: dict) -> str:
    df = strategies.get(name)
    if df is not None and "Symbol" in df.columns:
        try:
            symbol = str(df["Symbol"].mode()[0])
            return symbol.split("_")[0]
        except Exception:
            pass
    parts = name.split("/")
    return parts[-2] if len(parts) >= 2 else "?"


def _style(ax, hide_ticks: bool = False) -> None:
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


# ── Left info panel — data-coordinate drawing (enables scroll) ─────────────────

_INFO_VISIBLE = 24     # rows shown at once; scroll reveals the rest

# Column layout in data-x space (xlim = 0-100)
# Reserve x=96-100 for the scroll track
_CX  = [1,  44,  57,  70,  83]    # left edge of each column
_CW  = [42,  12,  12,  12,  12]   # width of each column
_TRACK_X = 96.5
_TRACK_W = 3.0


def _dp(ax, x, y, w, h, facecolor, edgecolor=None, lw=0.4, zorder=1) -> None:
    """Add a Rectangle patch in data coordinates."""
    ax.add_patch(Rectangle(
        (x, y), w, h,
        facecolor=facecolor,
        edgecolor=edgecolor or "none",
        linewidth=lw,
        zorder=zorder,
        clip_on=True,
    ))


def _dt(ax, x, y, text, color, fontsize=7.4, ha="left", bold=False,
        va="center") -> None:
    """Add text in data coordinates."""
    ax.text(
        x, y, text,
        color=color,
        fontsize=fontsize,
        ha=ha, va=va,
        fontfamily="monospace",
        fontweight="bold" if bold else "normal",
        zorder=2,
        clip_on=True,
    )


def _draw_info(
    ax, cr: CombinationResult, rank: int, n_total: int,
    strategies: dict,
) -> tuple[int, object | None]:
    """
    Draw the info panel using data coordinates so mouse-wheel scrolling works.

    Returns (n_rows, thumb_patch) so the caller can update the thumb on scroll.
    """
    ax.cla()
    ax.set_facecolor(AX_BG)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ── Build row list ─────────────────────────────────────────────────────────
    rows: list[tuple[str, object]] = []
    rows.append(("title",   None))
    rows.append(("rdd",     None))
    rows.append(("gap",     None))
    rows.append(("sec",     "STRATEGIES"))
    for name in cr.combination:
        rows.append(("strategy", name))
    rows.append(("gap",     None))
    rows.append(("col_hdr", None))
    rows.append(("sec",     "WEIGHTS"))
    for name in cr.combination:
        rows.append(("w_row", name))
    rows.append(("gap",     None))
    rows.append(("sec",     "RISK / TRADE  ($)"))
    for name in cr.combination:
        rows.append(("r_row", name))
    rows.append(("r_total", None))
    rows.append(("gap",     None))
    rows.append(("sec",     "SCALE FACTOR"))
    rows.append(("scale",   None))
    rows.append(("gap",     None))
    rows.append(("sec",     "DD STATUS"))
    rows.append(("dd",      None))

    n_rows = len(rows)

    # Row i occupies y ∈ [n_rows-i-1,  n_rows-i)  (top row first)
    def _yb(i: int) -> float:   # bottom y of row i
        return float(n_rows - i - 1)

    def _ym(i: int) -> float:   # mid y of row i
        return _yb(i) + 0.5

    # ── Draw rows ──────────────────────────────────────────────────────────────
    for i, (rtype, data) in enumerate(rows):
        yb = _yb(i)
        ym = _ym(i)

        if rtype == "title":
            _dp(ax, 0, yb, 100, 1, "#141a30")
            _dt(ax, 2, ym, f"Combination #{rank} / {n_total}",
                CYAN, fontsize=8.2, bold=True)

        elif rtype == "rdd":
            _dp(ax, 0, yb, 100, 1, "#0f1220")
            _dt(ax, 3, ym, f"  Raw Return/DD   {cr.raw_return_dd:.3f}",
                TEXT, fontsize=7.4)

        elif rtype == "gap":
            pass

        elif rtype == "sec":
            _dp(ax, 0, yb, 100, 1, SEC_BG)
            _dt(ax, 2, ym, data, AMBER, fontsize=7.0, bold=True)

        elif rtype == "strategy":
            name = data
            bg = ROW_A if i % 2 == 0 else ROW_B
            _dp(ax, 0, yb, 100, 1, bg)
            short = _short(name)
            asset = _asset(name, strategies)
            _dt(ax, _CX[0] + 2, ym, f"  •  {short}", TEXT, fontsize=7.2)
            # Asset badge — patch at z=2, text at z=3 so text sits on top
            bx = 70
            _dp(ax, bx, yb + 0.12, 24, 0.76, "#1e3050",
                edgecolor="#2060a0", lw=0.5, zorder=2)
            ax.text(bx + 12, ym, asset, color=CYAN, fontsize=7.0,
                    ha="center", va="center", fontfamily="monospace",
                    zorder=3, clip_on=True)

        elif rtype == "col_hdr":
            _dp(ax, 0, yb, _CX[0] + _CW[0], 1, COL_HDR)
            for k, m in enumerate(METHODS):
                cx, cw = _CX[k + 1], _CW[k + 1]
                _dp(ax, cx, yb, cw, 1, COL_HDR,
                    edgecolor=BORDER, lw=0.5)
                _dt(ax, cx + cw / 2, ym, METHOD_LABELS[m],
                    METHOD_COLORS[m], fontsize=7.2, ha="center", bold=True)

        elif rtype in ("w_row", "r_row"):
            name = data
            bg = ROW_A if i % 2 == 0 else ROW_B
            _dp(ax, 0, yb, _CX[0] + _CW[0], 1, bg)
            _dt(ax, _CX[0] + 1, ym, _short(name)[:18], TEXT, fontsize=7.2)
            for k, m in enumerate(METHODS):
                cx, cw = _CX[k + 1], _CW[k + 1]
                vp = cr.portfolios.get(m)
                if vp:
                    if rtype == "w_row":
                        val = f"{vp.weights.get(name, 0):.3f}"
                    else:
                        val = f"{vp.risk_per_trade.get(name, 0):,.0f}"
                    fg = TEXT
                else:
                    val, fg = "—", DIM
                _dp(ax, cx, yb, cw, 1, bg, edgecolor=BORDER, lw=0.3)
                _dt(ax, cx + cw / 2, ym, val, fg,
                    fontsize=7.2, ha="center")

        elif rtype == "r_total":
            bg = "#1a2535"
            _dp(ax, 0, yb, 100, 1, bg)
            _dp(ax, _CX[0], yb, _CW[0], 1, bg, edgecolor=BORDER, lw=0.3)
            _dt(ax, _CX[0] + 1, ym, "TOTAL",
                AMBER, fontsize=7.2, bold=True)
            for k, m in enumerate(METHODS):
                cx, cw = _CX[k + 1], _CW[k + 1]
                vp = cr.portfolios.get(m)
                if vp:
                    total = sum(vp.risk_per_trade.values())
                    val, fg = f"{total:,.0f}", CYAN
                else:
                    val, fg = "—", DIM
                _dp(ax, cx, yb, cw, 1, bg, edgecolor=BORDER, lw=0.3)
                _dt(ax, cx + cw / 2, ym, val, fg,
                    fontsize=7.2, ha="center", bold=True)

        elif rtype == "scale":
            bg = ROW_A
            _dp(ax, 0, yb, 100, 1, bg)
            for k, m in enumerate(METHODS):
                cx, cw = _CX[k + 1], _CW[k + 1]
                vp = cr.portfolios.get(m)
                val = f"×{vp.scale_factor:.3f}" if vp else "—"
                fg  = TEXT if vp else DIM
                _dp(ax, cx, yb, cw, 1, bg, edgecolor=BORDER, lw=0.3)
                _dt(ax, cx + cw / 2, ym, val, fg,
                    fontsize=7.2, ha="center")

        elif rtype == "dd":
            _dp(ax, 0, yb, 100, 1, ROW_B)
            for k, m in enumerate(METHODS):
                cx, cw = _CX[k + 1], _CW[k + 1]
                vp = cr.portfolios.get(m)
                if vp is None:
                    val, fg, cbg = "—",    DIM,   ROW_B
                elif vp.dd_failed:
                    val, fg, cbg = "FAIL", RED,   "#2a0808"
                else:
                    val, fg, cbg = "PASS", GREEN, "#082a12"
                _dp(ax, cx, yb, cw, 1, cbg, edgecolor=BORDER, lw=0.5)
                _dt(ax, cx + cw / 2, ym, val, fg,
                    fontsize=7.2, ha="center", bold=True)

    # ── Set initial ylim (show topmost rows) ───────────────────────────────────
    ax.set_ylim(max(0, n_rows - _INFO_VISIBLE), n_rows)

    # ── Scroll track + thumb ───────────────────────────────────────────────────
    thumb = None
    if n_rows > _INFO_VISIBLE:
        # Track
        _dp(ax, _TRACK_X, 0, _TRACK_W, n_rows,
            "#0a0d1a", edgecolor=BORDER, lw=0.5, zorder=5)
        # Thumb (initial position = top)
        thumb = Rectangle(
            (_TRACK_X, n_rows - _INFO_VISIBLE),
            _TRACK_W, _INFO_VISIBLE,
            facecolor=AMBER, edgecolor="none",
            linewidth=0, zorder=6, clip_on=True,
            alpha=0.7,
        )
        ax.add_patch(thumb)

        # Hint label
        _dt(ax, _TRACK_X + _TRACK_W / 2, -0.6,
            "↕", DIM, fontsize=8, ha="center")

    return n_rows, thumb


# ── Equity curves ──────────────────────────────────────────────────────────────

def _draw_equity(ax, cr: CombinationResult, wf_visible: bool = True) -> list:
    """Draw equity curves. Returns list of WF line objects for toggle control."""
    ax.cla()
    _style(ax)

    for m in METHODS:
        vp = cr.portfolios.get(m)
        if vp is None:
            continue
        eq    = _equity(vp.portfolio_df)
        label = f"{METHOD_LABELS[m]}  Sharpe {vp.sharpe:.2f}"
        if vp.dd_failed:
            ax.plot(eq.index, eq.values,
                    color=METHOD_COLORS[m], lw=1.5, ls="--", alpha=0.55,
                    label=f"{label}  [DD>limit]")
        else:
            ax.plot(eq.index, eq.values,
                    color=METHOD_COLORS[m], lw=2,
                    label=label)

    # Walk-forward equity curves (dotted, same color, no legend entry)
    # Offset each WF curve so it starts at the same level as the static equity
    # of the same method at the WF start date — making them directly comparable.
    wf_lines = []
    wf_equity = getattr(cr, "wf_equity", {})
    for m, eq_wf in wf_equity.items():
        if eq_wf is None:
            continue
        vp = cr.portfolios.get(m)
        if vp is not None:
            eq_static = _equity(vp.portfolio_df)
            wf_start  = eq_wf.index[0]
            # Find static value at or just before WF start
            static_at_start = eq_static.asof(wf_start) if hasattr(eq_static.index, 'asof') else (
                eq_static[eq_static.index <= wf_start].iloc[-1]
                if (eq_static.index <= wf_start).any() else 0.0
            )
            offset = float(static_at_start) if not (static_at_start != static_at_start) else 0.0
        else:
            offset = 0.0
        line, = ax.plot(
            eq_wf.index, eq_wf.values + offset,
            color=METHOD_COLORS[m], lw=1.2, ls=":", alpha=0.7,
            visible=wf_visible,
        )
        wf_lines.append(line)

    ax.axhline(0, color=BORDER, lw=0.8)

    ax.set_ylabel("Cumul. P&L ($)", color=TEXT, fontsize=8.5, labelpad=4)
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.tick_params(axis="y", direction="in", pad=3, colors=TEXT)
    ax.tick_params(axis="x", colors=TEXT)

    ax.legend(fontsize=7.5, facecolor=AX_BG, labelcolor=TEXT,
              loc="upper left", framealpha=0.8)
    ax.grid(True, color=BORDER, lw=0.5)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    plt.setp(ax.get_xticklabels(), visible=False)

    return wf_lines


# ── Metrics comparison table ───────────────────────────────────────────────────

_METRIC_ROWS = [
    ("Sharpe",       "sharpe_ratio",        "{:.3f}",  True),
    ("CAGR %",       "cagr",                "{:.2f}",  True),
    ("Profit ($)",   "total_profit",        "{:,.0f}", True),
    ("Return / DD",  "return_dd_ratio",     "{:.2f}",  True),
    ("Win %",        "winning_percentage",  "{:.1f}",  True),
    ("Pft Factor",   "profit_factor",       "{:.2f}",  True),
    ("Max DD %",     "pct_drawdown",        "{:.2f}",  False),
    ("Max DD ($)",   "drawdown",            "{:,.0f}", False),
    ("Num Trades",   "num_trades",          "{:,.0f}", True),
    ("DD Status",    "__dd__",              "{}",      None),
    ("MAE Stress",   "__mae__",             "{}",      None),
]


def _draw_metrics(ax, cr: CombinationResult) -> None:
    ax.cla()
    ax.set_facecolor(BG)
    ax.axis("off")

    col_labels  = ["Metric"] + [_METHOD_FULL[m] for m in METHODS]
    rows_data   = []
    cell_colors = []
    text_colors = []

    for label, key, fmt, higher_better in _METRIC_ROWS:
        row   = [label]
        c_row = [AX_BG]
        t_row = [DIM]

        if key == "__dd__":
            for m in METHODS:
                vp = cr.portfolios.get(m)
                if vp is None:
                    row.append("—");    c_row.append(AX_BG);    t_row.append(DIM)
                elif vp.dd_failed:
                    row.append("FAIL"); c_row.append("#2a0808"); t_row.append(RED)
                else:
                    row.append("PASS"); c_row.append("#082a12"); t_row.append(GREEN)

        elif key == "__mae__":
            for m in METHODS:
                vp = cr.portfolios.get(m)
                if vp is None or vp.stress is None:
                    row.append("—");    c_row.append(AX_BG);    t_row.append(DIM)
                elif vp.stress.passed:
                    row.append("PASS"); c_row.append("#082a12"); t_row.append(GREEN)
                else:
                    row.append("FAIL"); c_row.append("#2a0808"); t_row.append(RED)

        else:
            vals = []
            for m in METHODS:
                vp = cr.portfolios.get(m)
                if vp is None:
                    row.append("—"); c_row.append(AX_BG); t_row.append(DIM)
                    vals.append(None)
                else:
                    v = vp.metrics.get(key, 0.0)
                    row.append(fmt.format(v))
                    c_row.append(AX_BG)
                    t_row.append(TEXT)
                    vals.append(v)

            valid_vals = [(i, v) for i, v in enumerate(vals) if v is not None]
            if valid_vals and higher_better is not None:
                best_i = (max if higher_better else min)(valid_vals, key=lambda x: x[1])[0]
                c_row[best_i + 1] = "#0e2535"
                t_row[best_i + 1] = CYAN

        rows_data.append(row)
        cell_colors.append(c_row)
        text_colors.append(t_row)

    tbl = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(range(len(col_labels)))
    tbl.scale(1, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            if c > 0:
                m  = METHODS[c - 1]
                vp = cr.portfolios.get(m)
                tc = METHOD_COLORS[m] if (vp and not vp.dd_failed) else DIM
            else:
                tc = CYAN
            cell.set_facecolor(COL_HDR)
            cell.get_text().set_color(tc)
            cell.get_text().set_fontweight("bold")
        else:
            bg       = ROW_A if r % 2 == 0 else ROW_B
            override = cell_colors[r - 1][c]
            cell.set_facecolor(override if override != AX_BG else bg)
            cell.get_text().set_color(text_colors[r - 1][c])


# ── WF metrics table ───────────────────────────────────────────────────────────

# Keys that cannot be derived from an equity curve — shown as "—" in WF table
_WF_NOT_AVAILABLE = {"profit_factor", "num_trades", "__mae__"}

_WF_METHOD_COL_LABELS = {
    "equal":        "Equal (static)",
    "min_variance": "Min Var WF",
    "risk_parity":  "Risk Par WF",
    "hrp":          "HRP WF",
}


def _draw_wf_metrics(ax, cr: CombinationResult, config: PortfolioConfig) -> None:
    ax.cla()
    ax.set_facecolor(BG)
    ax.axis("off")

    # cr.wf_metrics pre-computed by compute_all_wf_equities using same compute_metrics()
    stored_wf = getattr(cr, "wf_metrics", {})

    # Build per-method metric dicts — equal uses static vp (WF = static for equal weight)
    wf_metrics: dict[str, dict] = {}
    for m in METHODS:
        if m == "equal":
            vp = cr.portfolios.get("equal")
            if vp:
                d = dict(vp.metrics)
                d["dd_failed"] = vp.dd_failed
                wf_metrics["equal"] = d
            else:
                wf_metrics["equal"] = {}
        else:
            md = stored_wf.get(m, {})
            if md:
                # dd_failed derived same way as validator
                dd_usd = md.get("drawdown", 0.0)
                md = dict(md)
                md["dd_failed"] = dd_usd > config.total_drawdown_limit_usd
            wf_metrics[m] = md

    col_labels  = ["Metric (WF)"] + [_WF_METHOD_COL_LABELS[m] for m in METHODS]
    rows_data   = []
    cell_colors = []
    text_colors = []

    # Use exact same rows as the static table
    for label, key, fmt, higher_better in _METRIC_ROWS:
        row   = [label]
        c_row = [AX_BG]
        t_row = [DIM]

        if key in _WF_NOT_AVAILABLE:
            # Row exists but values unavailable for WF methods
            for m in METHODS:
                if key == "__mae__":
                    row.append("—"); c_row.append(AX_BG); t_row.append(DIM)
                else:
                    row.append("—"); c_row.append(AX_BG); t_row.append(DIM)
        elif key == "__dd__":
            for m in METHODS:
                md = wf_metrics.get(m, {})
                if not md:
                    row.append("—");    c_row.append(AX_BG);    t_row.append(DIM)
                elif md.get("dd_failed", False):
                    row.append("FAIL"); c_row.append("#2a0808"); t_row.append(RED)
                else:
                    row.append("PASS"); c_row.append("#082a12"); t_row.append(GREEN)
        else:
            vals = []
            for m in METHODS:
                md = wf_metrics.get(m, {})
                if not md or key not in md:
                    row.append("—"); c_row.append(AX_BG); t_row.append(DIM)
                    vals.append(None)
                else:
                    v = md[key]
                    row.append(fmt.format(v))
                    c_row.append(AX_BG)
                    t_row.append(TEXT)
                    vals.append(v)

            valid_vals = [(i, v) for i, v in enumerate(vals) if v is not None]
            if valid_vals and higher_better is not None:
                best_i = (max if higher_better else min)(valid_vals, key=lambda x: x[1])[0]
                c_row[best_i + 1] = "#0e2535"
                t_row[best_i + 1] = CYAN

        rows_data.append(row)
        cell_colors.append(c_row)
        text_colors.append(t_row)

    tbl = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(range(len(col_labels)))
    tbl.scale(1, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            if c > 0:
                m  = METHODS[c - 1]
                vp = cr.portfolios.get(m)
                tc = METHOD_COLORS[m] if (vp and not vp.dd_failed) else DIM
            else:
                tc = AMBER
            cell.set_facecolor(COL_HDR)
            cell.get_text().set_color(tc)
            cell.get_text().set_fontweight("bold")
        else:
            bg       = ROW_A if r % 2 == 0 else ROW_B
            override = cell_colors[r - 1][c]
            cell.set_facecolor(override if override != AX_BG else bg)
            cell.get_text().set_color(text_colors[r - 1][c])


# ── Full render ────────────────────────────────────────────────────────────────

def _render(
    fig, ax_info, ax_eq, ax_metrics, ax_wf_metrics, nav_label,
    cr: CombinationResult, idx: int, n: int,
    strategies: dict, scroll_info: dict, config: PortfolioConfig,
) -> None:
    n_rows, thumb = _draw_info(ax_info, cr, idx + 1, n, strategies)
    scroll_info["n_rows"]  = n_rows
    scroll_info["offset"]  = 0
    scroll_info["thumb"]   = thumb

    wf_visible = scroll_info.get("wf_visible", True)
    wf_lines   = _draw_equity(ax_eq, cr, wf_visible=wf_visible)
    scroll_info["wf_lines"] = wf_lines
    _draw_metrics(ax_metrics, cr)
    _draw_wf_metrics(ax_wf_metrics, cr, config)

    n_pass = cr.n_valid_methods
    n_fail = len(METHODS) - n_pass
    status = f"{n_pass}/4 DD pass" + (f"  {n_fail} fail (shown dashed)" if n_fail else "")

    fig.suptitle(
        f"Combination #{idx+1}/{n}  |  "
        f"Raw R/DD {cr.raw_return_dd:.2f}  |  {status}",
        color=CYAN, fontsize=11, y=0.99,
    )
    nav_label.set_text(f"Combination {idx + 1} of {n}")
    fig.canvas.draw_idle()


# ── Overview table (Window 2) ──────────────────────────────────────────────────

def _show_overview(combinations: list[CombinationResult]) -> None:
    col_labels = [
        "#", "Strategies", "Raw R/DD",
        f"Sharpe ({METHOD_LABELS['equal']})",
        f"Sharpe ({METHOD_LABELS['min_variance']})",
        f"Sharpe ({METHOD_LABELS['risk_parity']})",
        f"Sharpe ({METHOD_LABELS['hrp']})",
        "Best CAGR%", "Best MaxDD%", "Best R/DD", "DD Pass",
    ]

    rows_data = []
    for cr in combinations:
        def _s(m, _cr=cr):
            vp = _cr.portfolios.get(m)
            return f"{vp.metrics['sharpe_ratio']:.3f}" if vp else "—"

        def _best(key, higher=True, _cr=cr):
            vals = [vp.metrics.get(key, 0) for vp in _cr.portfolios.values() if vp]
            if not vals:
                return "—"
            v = (max if higher else min)(vals)
            return f"{v:.2f}"

        rows_data.append([
            str(cr.rank),
            f"Combination {cr.rank}",
            f"{cr.raw_return_dd:.2f}",
            _s("equal"), _s("min_variance"), _s("risk_parity"), _s("hrp"),
            _best("cagr"),
            _best("pct_drawdown", higher=False),
            _best("return_dd_ratio"),
            f"{cr.n_valid_methods}/4",
        ])

    n     = len(combinations)
    fig_h = max(5.0, n * 0.42 + 3.0)
    fig2, ax2 = plt.subplots(figsize=(24, fig_h), facecolor=BG)
    ax2.set_facecolor(BG)
    ax2.axis("off")
    fig2.suptitle(
        "Portfolio Overview  —  combinations ranked by equal-weight Return/DD  "
        "|  dashed = DD>limit (still usable at lower sizing)",
        color=CYAN, fontsize=11, y=0.98,
    )

    tbl = ax2.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(range(len(col_labels)))
    tbl.scale(1, 1.3)

    sharpe_cols = {3: "equal", 4: "min_variance", 5: "risk_parity", 6: "hrp"}
    dd_col = 10

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(COL_HDR)
            if c in sharpe_cols:
                cell.get_text().set_color(METHOD_COLORS[sharpe_cols[c]])
            else:
                cell.get_text().set_color(CYAN)
            cell.get_text().set_fontweight("bold")
        else:
            bg = ROW_A if r % 2 == 0 else ROW_B
            cell.set_facecolor(bg)
            cr = combinations[r - 1]
            if c in sharpe_cols:
                m  = sharpe_cols[c]
                vp = cr.portfolios.get(m)
                color = METHOD_COLORS[m] if (vp and not vp.dd_failed) else DIM
                cell.get_text().set_color(color)
            elif c == dd_col:
                n_ok = cr.n_valid_methods
                cell.get_text().set_color(GREEN if n_ok == 4 else AMBER if n_ok >= 2 else RED)
            else:
                cell.get_text().set_color(TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.95])


# ── WF Weight History table ────────────────────────────────────────────────────

_WF_METHOD_FULL = {
    "min_variance": "Min Variance",
    "risk_parity":  "Risk Parity",
    "hrp":          "HRP",
}


def _show_wf_weights(cr: CombinationResult) -> None:
    """
    Open a new figure showing the walk-forward weight history for one combination.

    Rows    = strategies (short names)
    Columns = OOS windows (e.g. "2019–2022")
    RadioButtons switch between methods.
    """
    wf_weights = getattr(cr, "wf_weights", {})

    # Collect all windows across methods to find a consistent column set
    all_windows: list[str] = []
    for m in WF_METHODS:
        for win in wf_weights.get(m, []):
            if win["label"] not in all_windows:
                all_windows.append(win["label"])

    if not all_windows:
        print("  No walk-forward weight history available for this combination.")
        return

    short_names = [n.split("/")[-1] for n in cr.combination]

    fig_w = max(10.0, len(all_windows) * 1.8 + 3.0)
    fig_h = max(4.0,  len(short_names) * 0.5 + 3.5)
    fig_wf = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    fig_wf.suptitle(
        f"Walk-Forward Weight History  |  Combination #{cr.rank}",
        color=CYAN, fontsize=11, y=0.98,
    )

    # Reserve left margin for RadioButtons
    ax_tbl = fig_wf.add_axes([0.18, 0.10, 0.80, 0.82])
    ax_tbl.set_facecolor(BG)
    ax_tbl.axis("off")

    # RadioButtons for method selection
    ax_radio = fig_wf.add_axes([0.01, 0.30, 0.14, 0.40], facecolor=AX_BG)
    from matplotlib.widgets import RadioButtons
    radio = RadioButtons(
        ax_radio,
        labels=[_WF_METHOD_FULL[m] for m in WF_METHODS],
        activecolor=CYAN,
    )
    for lbl, m in zip(radio.labels, WF_METHODS):
        lbl.set_color(METHOD_COLORS[m])
        lbl.set_fontsize(8.5)

    tbl_ref = [None]   # mutable container so the callback can replace the table

    def _draw_table(method: str) -> None:
        if tbl_ref[0] is not None:
            tbl_ref[0].remove()
            tbl_ref[0] = None

        history = wf_weights.get(method, [])
        win_map  = {w["label"]: w["weights"] for w in history}

        col_labels = ["Strategy"] + all_windows
        rows_data  : list[list[str]] = []
        for name, short in zip(cr.combination, short_names):
            row = [short]
            for win_label in all_windows:
                w = win_map.get(win_label, {}).get(name, None)
                row.append(f"{w:.3f}" if w is not None else "—")
            rows_data.append(row)

        tbl = ax_tbl.table(
            cellText=rows_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(range(len(col_labels)))
        tbl.scale(1, 1.6)

        mc = METHOD_COLORS[method]
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(BORDER)
            if r == 0:
                cell.set_facecolor(COL_HDR)
                cell.get_text().set_color(mc if c > 0 else CYAN)
                cell.get_text().set_fontweight("bold")
            else:
                bg = ROW_A if r % 2 == 0 else ROW_B
                cell.set_facecolor(bg)
                cell.get_text().set_color(TEXT if c > 0 else DIM)

        tbl_ref[0] = tbl
        fig_wf.canvas.draw_idle()

    def on_method(label: str) -> None:
        m = next(m for m in WF_METHODS if _WF_METHOD_FULL[m] == label)
        _draw_table(m)

    radio.on_clicked(on_method)
    fig_wf._refs = [radio, tbl_ref]

    _draw_table(WF_METHODS[0])
    fig_wf.show()


# ── Entry point ────────────────────────────────────────────────────────────────

def run_portfolio_dashboard(
    combinations: list[CombinationResult],
    config: PortfolioConfig,
    strategies: dict | None = None,
    _show: bool = True,
) -> None:
    """
    Open two local matplotlib windows.

    Window 1: combination explorer (< Prev / Next >).
    Window 2: ranking overview table.
    """
    if not combinations:
        print("  No combinations to display.")
        return

    strategies = strategies or {}
    n          = len(combinations)
    state      = {"idx": 0}
    scroll_info: dict = {"offset": 0, "n_rows": 0, "thumb": None, "wf_visible": True, "wf_lines": []}

    # ── Figure & layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(28, 12), facecolor=BG)

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1.0, 1.5, 1.5],  # info | static metrics | WF metrics
        height_ratios=[2.6, 2.4],
        hspace=0.06,
        wspace=0.06,
        left=0.01, right=0.99,
        top=0.93, bottom=0.10,
    )

    ax_info       = fig.add_subplot(gs[0:2, 0])
    ax_eq         = fig.add_subplot(gs[0, 1:3])   # equity spans both metric cols
    ax_metrics    = fig.add_subplot(gs[1, 1])
    ax_wf_metrics = fig.add_subplot(gs[1, 2])

    ax_nav = fig.add_axes([0.42, 0.02, 0.16, 0.045])
    ax_nav.set_facecolor(BG)
    ax_nav.axis("off")
    nav_label = ax_nav.text(0.5, 0.5, "", transform=ax_nav.transAxes,
                            color=TEXT, ha="center", va="center", fontsize=10)

    ax_prev    = fig.add_axes([0.34, 0.02, 0.07, 0.045])
    ax_next    = fig.add_axes([0.59, 0.02, 0.07, 0.045])
    ax_wf      = fig.add_axes([0.70, 0.02, 0.10, 0.045])
    ax_wf_tbl  = fig.add_axes([0.81, 0.02, 0.10, 0.045])
    btn_prev   = Button(ax_prev,   "< Prev",     color=AX_BG,    hovercolor=BORDER)
    btn_next   = Button(ax_next,   "Next >",     color=AX_BG,    hovercolor=BORDER)
    btn_wf     = Button(ax_wf,     "WF: ON",     color="#0a2010", hovercolor="#0d2a18")
    btn_wf_tbl = Button(ax_wf_tbl, "WF Weights", color=AX_BG,    hovercolor=BORDER)
    btn_prev.label.set_color(TEXT)
    btn_next.label.set_color(TEXT)
    btn_wf.label.set_color(GREEN)
    btn_wf_tbl.label.set_color(AMBER)

    # ── Navigation callbacks ───────────────────────────────────────────────────
    def on_prev(event):
        state["idx"] = max(state["idx"] - 1, 0)
        _render(fig, ax_info, ax_eq, ax_metrics, ax_wf_metrics, nav_label,
                combinations[state["idx"]], state["idx"], n,
                strategies, scroll_info, config)

    def on_next(event):
        state["idx"] = min(state["idx"] + 1, n - 1)
        _render(fig, ax_info, ax_eq, ax_metrics, ax_wf_metrics, nav_label,
                combinations[state["idx"]], state["idx"], n,
                strategies, scroll_info, config)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    def on_wf_toggle(event):
        current = scroll_info.get("wf_visible", True)
        new_vis = not current
        scroll_info["wf_visible"] = new_vis
        for line in scroll_info.get("wf_lines", []):
            line.set_visible(new_vis)
        if new_vis:
            btn_wf.label.set_text("WF: ON")
            btn_wf.label.set_color(GREEN)
            ax_wf.set_facecolor("#0a2010")
        else:
            btn_wf.label.set_text("WF: OFF")
            btn_wf.label.set_color(DIM)
            ax_wf.set_facecolor(AX_BG)
        fig.canvas.draw_idle()

    btn_wf.on_clicked(on_wf_toggle)

    def on_wf_weights(event):
        _show_wf_weights(combinations[state["idx"]])

    btn_wf_tbl.on_clicked(on_wf_weights)

    # ── Scroll callback ────────────────────────────────────────────────────────
    def on_scroll(event):
        if event.inaxes != ax_info:
            return
        n_rows = scroll_info["n_rows"]
        if n_rows <= _INFO_VISIBLE:
            return

        step = -int(event.step) * 2   # scroll up = negative step = move thumb up
        new_offset = max(
            0,
            min(n_rows - _INFO_VISIBLE, scroll_info["offset"] + step),
        )
        scroll_info["offset"] = new_offset

        top = n_rows - new_offset
        ax_info.set_ylim(top - _INFO_VISIBLE, top)

        # Move thumb
        thumb = scroll_info["thumb"]
        if thumb is not None:
            thumb.set_y(top - _INFO_VISIBLE)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Keep widget references alive — prevents GC when _show=False returns early
    fig._refs = [btn_prev, btn_next, btn_wf, btn_wf_tbl, state, scroll_info]

    # ── Initial render ─────────────────────────────────────────────────────────
    _show_overview(combinations)
    _render(fig, ax_info, ax_eq, ax_metrics, ax_wf_metrics, nav_label,
            combinations[0], 0, n, strategies, scroll_info, config)

    if _show:
        plt.show()
