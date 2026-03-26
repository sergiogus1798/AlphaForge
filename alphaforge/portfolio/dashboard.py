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
            return str(df["Symbol"].mode()[0])
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

def _draw_equity(ax, cr: CombinationResult) -> None:
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

    ax.axhline(0, color=BORDER, lw=0.8)

    # Y-axis label close to axis, ticks facing inward
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


# ── Full render ────────────────────────────────────────────────────────────────

def _render(
    fig, ax_info, ax_eq, ax_metrics, nav_label,
    cr: CombinationResult, idx: int, n: int,
    strategies: dict, scroll_info: dict,
) -> None:
    n_rows, thumb = _draw_info(ax_info, cr, idx + 1, n, strategies)
    scroll_info["n_rows"]  = n_rows
    scroll_info["offset"]  = 0
    scroll_info["thumb"]   = thumb

    _draw_equity(ax_eq, cr)
    _draw_metrics(ax_metrics, cr)

    n_pass = cr.n_valid_methods
    n_fail = len(METHODS) - n_pass
    status = f"{n_pass}/4 DD pass" + (f"  {n_fail} fail (shown dashed)" if n_fail else "")

    fig.suptitle(
        f"Combination #{idx+1}/{n}  |  {cr.combo_label}  |  "
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
    scroll_info: dict = {"offset": 0, "n_rows": 0, "thumb": None}

    # ── Figure & layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 12), facecolor=BG)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[1.0, 3.0],    # wider right column for equity + metrics
        height_ratios=[2.6, 2.4],
        hspace=0.06,
        wspace=0.04,
        left=0.01, right=0.99,
        top=0.93, bottom=0.10,
    )

    ax_info    = fig.add_subplot(gs[0:2, 0])
    ax_eq      = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[1, 1])

    ax_nav = fig.add_axes([0.42, 0.02, 0.16, 0.045])
    ax_nav.set_facecolor(BG)
    ax_nav.axis("off")
    nav_label = ax_nav.text(0.5, 0.5, "", transform=ax_nav.transAxes,
                            color=TEXT, ha="center", va="center", fontsize=10)

    ax_prev = fig.add_axes([0.34, 0.02, 0.07, 0.045])
    ax_next = fig.add_axes([0.59, 0.02, 0.07, 0.045])
    btn_prev = Button(ax_prev, "< Prev", color=AX_BG, hovercolor=BORDER)
    btn_next = Button(ax_next, "Next >", color=AX_BG, hovercolor=BORDER)
    btn_prev.label.set_color(TEXT)
    btn_next.label.set_color(TEXT)

    # ── Navigation callbacks ───────────────────────────────────────────────────
    def on_prev(event):
        state["idx"] = max(state["idx"] - 1, 0)
        _render(fig, ax_info, ax_eq, ax_metrics, nav_label,
                combinations[state["idx"]], state["idx"], n,
                strategies, scroll_info)

    def on_next(event):
        state["idx"] = min(state["idx"] + 1, n - 1)
        _render(fig, ax_info, ax_eq, ax_metrics, nav_label,
                combinations[state["idx"]], state["idx"], n,
                strategies, scroll_info)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

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
    fig._refs = [btn_prev, btn_next, state, scroll_info]

    # ── Initial render ─────────────────────────────────────────────────────────
    _show_overview(combinations)
    _render(fig, ax_info, ax_eq, ax_metrics, nav_label,
            combinations[0], 0, n, strategies, scroll_info)

    if _show:
        plt.show()
