"""
exporter.py — Export pipeline results to portfolios/ output directory.

For each valid (combination, method) pair produces:
    portfolios/
    └── Combination001/
        ├── Equal/
        │   ├── portfolio.xlsx   (Trades | Metrics | Weights & Sizing | Stress Test)
        │   └── report.pdf       (cover · equity + drawdown · monthly heatmap ·
        │                         full metrics table · weights & sizing)
        ├── MinVariance/
        ├── RiskParity/
        └── HRP/
"""

from __future__ import annotations

import os
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle as _Rect

from alphaforge.paths import PORTFOLIOS_OUTPUT, ensure_method_dir
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.combo_result import (
    CombinationResult, METHODS,
)


# ── Theme (light / print-friendly) ─────────────────────────────────────────────

PDF_BG     = "#ffffff"
PDF_PANEL  = "#f0f2f8"
PDF_TEXT   = "#1a1a2e"
PDF_DIM    = "#555570"
PDF_ACCENT = "#1a3a8a"
PDF_GREEN  = "#1a7a3a"
PDF_RED    = "#b02020"
PDF_GRID   = "#d8dcea"
PDF_ROW_A  = "#f7f8fc"
PDF_ROW_B  = "#eceef5"
PDF_SEC    = "#d4daf0"

METHOD_FULL = {
    "equal":        "Equal Weight",
    "min_variance": "Min Variance",
    "risk_parity":  "Risk Parity",
    "hrp":          "HRP",
}

A4 = (8.27, 11.69)   # inches, portrait


# ── Data helpers ────────────────────────────────────────────────────────────────

def _equity(vp) -> pd.Series:
    df    = vp.portfolio_df
    daily = df.groupby(df["Close time"].dt.date)["Profit/Loss"].sum()
    dr    = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(dr, fill_value=0.0).cumsum()


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity - equity.cummax()


def _monthly_pivot(vp) -> pd.DataFrame:
    df          = vp.portfolio_df.copy()
    df["year"]  = df["Close time"].dt.year
    df["month"] = df["Close time"].dt.month
    pivot       = df.groupby(["year", "month"])["Profit/Loss"].sum().unstack(fill_value=0.0)
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0.0
    return pivot[sorted(pivot.columns)]


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


# ── Shared figure helpers ───────────────────────────────────────────────────────

def _style_ts(ax) -> None:
    """Style a time-series axes for the PDF (light theme)."""
    ax.set_facecolor(PDF_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(PDF_GRID)
    ax.tick_params(colors=PDF_TEXT, labelsize=8)
    ax.grid(True, color=PDF_GRID, lw=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=20)


def _footer(fig, cr: CombinationResult, vp) -> None:
    method = METHOD_FULL.get(vp.method, vp.method)
    fig.add_artist(_Rect(
        (0, 0), 1, 0.038,
        transform=fig.transFigure,
        facecolor="#e8ecf5", edgecolor=PDF_GRID, lw=0.3, zorder=5,
    ))
    fig.text(
        0.5, 0.014,
        f"AlphaForge  ·  Combination #{cr.rank}  ·  {method}  ·  Confidential",
        fontsize=7.5, color=PDF_DIM, ha="center",
        transform=fig.transFigure, zorder=6,
    )


def _table_style(tbl, col_labels, section_rows: set) -> None:
    """Apply consistent styling to a matplotlib table."""
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PDF_GRID)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(PDF_ACCENT)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(9)
        elif (r - 1) in section_rows:
            cell.set_facecolor(PDF_SEC)
            cell.get_text().set_color(PDF_ACCENT)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(8.5)
        else:
            cell.get_text().set_color(PDF_TEXT)
            cell.get_text().set_fontsize(9)


# ── Page 1 — Cover page ─────────────────────────────────────────────────────────

def _page_cover(
    pdf: PdfPages,
    vp,
    cr: CombinationResult,
    config: PortfolioConfig,
    strategies: dict,
) -> None:
    fig = plt.figure(figsize=A4, facecolor=PDF_BG)

    # Header bar
    fig.add_artist(_Rect(
        (0, 0.880), 1, 0.120,
        transform=fig.transFigure,
        facecolor=PDF_ACCENT, edgecolor="none", zorder=1,
    ))
    fig.text(0.07, 0.962, "AlphaForge",
             fontsize=30, fontweight="bold", color="white",
             transform=fig.transFigure, zorder=2)
    fig.text(0.07, 0.918, "Portfolio Analysis Report",
             fontsize=13, color="#aac0e8",
             transform=fig.transFigure, zorder=2)
    fig.text(0.93, 0.962, date.today().strftime("%d %B %Y"),
             fontsize=9, color="#aac0e8", ha="right",
             transform=fig.transFigure, zorder=2)
    method_name = METHOD_FULL.get(vp.method, vp.method)
    fig.text(0.93, 0.918, f"Method: {method_name}",
             fontsize=11, color="white", ha="right", fontweight="bold",
             transform=fig.transFigure, zorder=2)

    # Content canvas
    ax = fig.add_axes([0.07, 0.055, 0.86, 0.810])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    y       = 96.0
    row_h   = 5.2
    ind     = 3.0
    val_x   = 58.0
    _row_i  = [0]   # mutable counter for alternating row colours

    def _sec(label: str) -> None:
        nonlocal y
        ax.add_patch(_Rect((0, y - 0.4), 100, row_h * 0.86,
                            facecolor=PDF_SEC, edgecolor=PDF_GRID, lw=0.4))
        ax.text(ind, y + 1.6, label, fontsize=9.5, fontweight="bold",
                color=PDF_ACCENT, va="bottom")
        y -= row_h
        _row_i[0] = 0

    def _row(label: str, value: str, value_color: str | None = None) -> None:
        nonlocal y
        bg = PDF_ROW_A if _row_i[0] % 2 == 0 else PDF_ROW_B
        ax.add_patch(_Rect((0, y - 0.3), 100, row_h * 0.86,
                            facecolor=bg, edgecolor="none"))
        ax.text(ind + 1.5, y + 1.5, label, fontsize=8.5, color=PDF_TEXT, va="bottom")
        ax.text(val_x, y + 1.5, value, fontsize=8.5,
                color=value_color or PDF_TEXT, va="bottom", fontweight="bold")
        y -= row_h
        _row_i[0] += 1

    # ── Combination strategies ─────────────────────────────────────────────────
    _sec(f"COMBINATION  #{cr.rank}")
    for name in cr.combination:
        _row(_short(name), _asset(name, strategies))

    y -= 0.8

    # ── Performance snapshot ───────────────────────────────────────────────────
    m = vp.metrics
    _sec("PERFORMANCE SUMMARY")
    _row("Sharpe Ratio",       f"{m.get('sharpe_ratio', 0):.3f}")
    _row("CAGR",               f"{m.get('cagr', 0):.2f}%")
    _row("Total Profit",       f"${m.get('total_profit', 0):,.0f}")
    _row("Profit Factor",      f"{m.get('profit_factor', 0):.2f}")
    _row("Return / DD Ratio",  f"{m.get('return_dd_ratio', 0):.2f}")
    _row("Win Rate",           f"{m.get('winning_percentage', 0):.1f}%")
    _row("Max Drawdown ($)",   f"${m.get('drawdown', 0):,.0f}")
    _row("Max Drawdown (%)",   f"{m.get('pct_drawdown', 0):.2f}%")
    _row("Num Trades",         f"{m.get('num_trades', 0):.0f}")

    y -= 0.8

    # ── Sizing ─────────────────────────────────────────────────────────────────
    _sec("SIZING")
    _row("Scale Factor",   f"×{vp.scale_factor:.4f}")
    _row("Critical Day",   str(vp.critical_day))
    _row("Worst Day Loss", f"${vp.worst_day_loss:,.2f}")

    y -= 0.8

    # ── Account ────────────────────────────────────────────────────────────────
    _sec("ACCOUNT")
    _row("Starting Capital",  f"${config.account_balance:,.0f}")
    _row("Daily Loss Limit",  f"${config.daily_loss_limit_usd:,.0f}  ({config.daily_loss_limit_pct*100:.1f}%)")
    _row("Total DD Limit",    f"${config.total_drawdown_limit_usd:,.0f}  ({config.total_drawdown_limit_pct*100:.1f}%)")

    y -= 1.5

    # ── Status badges ──────────────────────────────────────────────────────────
    dd_ok     = not vp.dd_failed
    mae_ok    = vp.stress.passed if vp.stress else None
    dd_col    = PDF_GREEN if dd_ok else PDF_RED
    badge_h   = row_h * 1.3
    badge_w   = 26

    ax.add_patch(_Rect((2, y - 0.5), badge_w, badge_h,
                        facecolor=dd_col + "22", edgecolor=dd_col, lw=1.2))
    ax.text(2 + badge_w / 2, y + badge_h / 2 - 0.5,
            f"DD:  {'PASS' if dd_ok else 'FAIL'}",
            fontsize=11, fontweight="bold", color=dd_col,
            ha="center", va="center")

    if mae_ok is not None:
        mae_col = PDF_GREEN if mae_ok else PDF_RED
        ax.add_patch(_Rect((34, y - 0.5), badge_w, badge_h,
                            facecolor=mae_col + "22", edgecolor=mae_col, lw=1.2))
        ax.text(34 + badge_w / 2, y + badge_h / 2 - 0.5,
                f"MAE Stress:  {'PASS' if mae_ok else 'FAIL'}",
                fontsize=11, fontweight="bold", color=mae_col,
                ha="center", va="center")

    # Footer
    fig.add_artist(_Rect(
        (0, 0), 1, 0.038,
        transform=fig.transFigure,
        facecolor="#e8ecf5", edgecolor=PDF_GRID, lw=0.3, zorder=5,
    ))
    fig.text(0.5, 0.014,
             f"AlphaForge  ·  Combination #{cr.rank}  ·  {method_name}  ·  Confidential",
             fontsize=7.5, color=PDF_DIM, ha="center",
             transform=fig.transFigure, zorder=6)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 2 — Equity curve + Drawdown ───────────────────────────────────────────

def _page_equity_drawdown(pdf: PdfPages, vp, cr: CombinationResult) -> None:
    eq = _equity(vp)
    dd = _drawdown(eq)

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=A4, facecolor=PDF_BG,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )
    fig.subplots_adjust(hspace=0.38, top=0.92, bottom=0.06, left=0.13, right=0.95)
    fig.suptitle(
        f"Equity & Drawdown  ·  Combination #{cr.rank}  ·  {METHOD_FULL.get(vp.method)}",
        fontsize=11, color=PDF_TEXT, y=0.97,
    )

    # Equity
    _style_ts(ax_eq)
    ax_eq.plot(eq.index, eq.values, color=PDF_ACCENT, lw=2)
    ax_eq.fill_between(eq.index, eq.values, 0,
                        where=eq.values >= 0, alpha=0.12, color=PDF_ACCENT)
    ax_eq.axhline(0, color=PDF_GRID, lw=0.8)
    ax_eq.set_title("Cumulative P&L", color=PDF_TEXT, fontsize=11, pad=6)
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # Drawdown
    _style_ts(ax_dd)
    ax_dd.fill_between(dd.index, dd.values, 0, color=PDF_RED, alpha=0.35)
    ax_dd.plot(dd.index, dd.values, color=PDF_RED, lw=1.2)
    ax_dd.axhline(0, color=PDF_GRID, lw=0.8)
    ax_dd.set_title("Drawdown from Peak", color=PDF_TEXT, fontsize=11, pad=6)
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    _footer(fig, cr, vp)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 3 — Monthly P&L heatmap ───────────────────────────────────────────────

def _page_monthly_heatmap(pdf: PdfPages, vp, cr: CombinationResult) -> None:
    try:
        pivot = _monthly_pivot(vp)
    except Exception:
        return

    MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
    n_years = len(pivot)
    fig_h   = min(11.69, max(5.5, n_years * 1.1 + 3.5))

    fig, ax = plt.subplots(figsize=(8.27, fig_h), facecolor=PDF_BG)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.92)
    fig.suptitle(
        f"Monthly P&L Heatmap  ·  Combination #{cr.rank}  ·  {METHOD_FULL.get(vp.method)}",
        fontsize=11, color=PDF_TEXT, y=0.97,
    )

    data = pivot.values
    vmax = max(abs(data).max(), 1.0)

    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, color=PDF_TEXT, fontsize=9)
    ax.set_yticks(range(n_years))
    ax.set_yticklabels([str(y) for y in pivot.index], color=PDF_TEXT, fontsize=9)
    ax.set_facecolor(PDF_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(PDF_GRID)

    for r in range(n_years):
        for c in range(12):
            val = data[r, c]
            if val == 0.0:
                txt = "—"
            elif abs(val) >= 1000:
                txt = f"${val/1000:.1f}k"
            else:
                txt = f"${val:.0f}"
            tc = "white" if abs(val) > vmax * 0.55 else PDF_TEXT
            ax.text(c, r, txt, ha="center", va="center", fontsize=7.5, color=tc)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors=PDF_TEXT)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    _footer(fig, cr, vp)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 4 — Full metrics table ─────────────────────────────────────────────────

def _page_metrics_table(
    pdf: PdfPages,
    vp,
    cr: CombinationResult,
    config: PortfolioConfig,
) -> None:
    m = vp.metrics

    rows_spec = [
        # (label, value, is_section)
        ("RETURNS",              "",                                          True),
        ("Total Profit ($)",     f"${m.get('total_profit', 0):,.0f}",        False),
        ("CAGR (%)",             f"{m.get('cagr', 0):.2f}",                  False),
        ("Avg Annual Return (%)",f"{m.get('yearly_avg_pct_return', 0):.2f}", False),
        ("Average Trade ($)",    f"${m.get('average_trade', 0):.2f}",        False),
        ("RISK",                 "",                                          True),
        ("Sharpe Ratio",         f"{m.get('sharpe_ratio', 0):.3f}",          False),
        ("Profit Factor",        f"{m.get('profit_factor', 0):.2f}",         False),
        ("Return / DD Ratio",    f"{m.get('return_dd_ratio', 0):.2f}",       False),
        ("Win Rate (%)",         f"{m.get('winning_percentage', 0):.1f}",    False),
        ("Max Drawdown ($)",     f"${m.get('drawdown', 0):,.0f}",            False),
        ("Max Drawdown (%)",     f"{m.get('pct_drawdown', 0):.2f}",          False),
        ("TRADES",               "",                                          True),
        ("Number of Trades",     f"{m.get('num_trades', 0):.0f}",            False),
        ("SIZING",               "",                                          True),
        ("Scale Factor",         f"×{vp.scale_factor:.4f}",                  False),
        ("Critical Day",         str(vp.critical_day),                       False),
        ("Worst Day Loss ($)",   f"${vp.worst_day_loss:,.2f}",               False),
        ("Daily Loss Limit ($)", f"${config.daily_loss_limit_usd:,.0f}",     False),
        ("Total DD Limit ($)",   f"${config.total_drawdown_limit_usd:,.0f}", False),
        ("VALIDATION",           "",                                          True),
        ("DD Status",            "PASS" if not vp.dd_failed else "FAIL",     False),
    ]

    if vp.stress:
        st = vp.stress
        rows_spec += [
            ("MAE Stress",          "PASS" if st.passed else "FAIL",           False),
            ("Worst Day (MAE $)",   f"${st.worst_day_loss_mae:,.2f}",          False),
            ("Max DD (MAE $)",      f"${st.max_drawdown_usd_mae:,.0f}",        False),
            ("Max DD (MAE %)",      f"{st.max_drawdown_pct_mae:.2f}",          False),
            ("MAE Coverage (%)",    f"{st.mae_coverage_pct:.0f}",              False),
        ]

    cell_text    = [[r[0], r[1]] for r in rows_spec]
    section_rows = {i for i, r in enumerate(rows_spec) if r[2]}

    n_rows = len(rows_spec)
    cell_colours = []
    for i, (_, _, is_sec) in enumerate(rows_spec):
        if is_sec:
            cell_colours.append([PDF_SEC, PDF_SEC])
        else:
            bg = PDF_ROW_A if i % 2 == 0 else PDF_ROW_B
            cell_colours.append([bg, bg])

    fig_h = min(11.69, max(6.0, n_rows * 0.38 + 2.5))
    fig, ax = plt.subplots(figsize=(8.27, fig_h), facecolor=PDF_BG)
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.06, right=0.94)
    fig.suptitle(
        f"Performance Metrics  ·  Combination #{cr.rank}  ·  {METHOD_FULL.get(vp.method)}",
        fontsize=11, color=PDF_TEXT, y=0.97,
    )
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        cellColours=cell_colours,
        loc="upper center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.5, 1.45)
    tbl.auto_set_column_width([0, 1])

    _table_style(tbl, ["Metric", "Value"], section_rows)

    # Colour PASS/FAIL cells
    pass_fail_labels = {"DD Status", "MAE Stress"}
    for i, (label, val, _) in enumerate(rows_spec):
        if label in pass_fail_labels:
            cell = tbl[(i + 1, 1)]
            is_pass = val == "PASS"
            cell.get_text().set_color(PDF_GREEN if is_pass else PDF_RED)
            cell.get_text().set_fontweight("bold")

    _footer(fig, cr, vp)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 5 — Weights & sizing ───────────────────────────────────────────────────

def _page_weights(
    pdf: PdfPages,
    vp,
    cr: CombinationResult,
    strategies: dict,
) -> None:
    combo = cr.combination

    rows = []
    for i, name in enumerate(combo):
        rows.append([
            _short(name),
            _asset(name, strategies),
            f"{vp.weights.get(name, 0):.4f}",
            f"${vp.risk_per_trade.get(name, 0):,.0f}",
        ])
    total_risk = sum(vp.risk_per_trade.values())
    rows.append(["TOTAL", "", "1.0000", f"${total_risk:,.0f}"])

    n         = len(rows)
    sec_rows  = {n - 1}   # TOTAL row styled as section

    cell_colours = []
    for i in range(n - 1):
        bg = PDF_ROW_A if i % 2 == 0 else PDF_ROW_B
        cell_colours.append([bg, bg, bg, bg])
    cell_colours.append([PDF_SEC, PDF_SEC, PDF_SEC, PDF_SEC])

    fig, ax = plt.subplots(figsize=A4, facecolor=PDF_BG)
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.06, right=0.94)
    fig.suptitle(
        f"Weights & Sizing  ·  Combination #{cr.rank}  ·  {METHOD_FULL.get(vp.method)}",
        fontsize=11, color=PDF_TEXT, y=0.97,
    )
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Strategy", "Asset", "Weight", "Risk / Trade ($)"],
        cellColours=cell_colours,
        loc="upper center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.5, 2.0)
    tbl.auto_set_column_width([0, 1, 2, 3])

    _table_style(tbl, ["Strategy", "Asset", "Weight", "Risk / Trade ($)"], sec_rows)

    _footer(fig, cr, vp)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Excel writer ────────────────────────────────────────────────────────────────

def _write_excel(
    vp,
    cr: CombinationResult,
    config: PortfolioConfig,
    strategies: dict,
    out_dir: str,
) -> None:
    path = os.path.join(out_dir, "portfolio.xlsx")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # ── Sheet 1: Trades ────────────────────────────────────────────────────
        trades = vp.portfolio_df.copy()
        # Strip timezone info — Excel doesn't support tz-aware datetimes
        for col in trades.select_dtypes(include="datetimetz").columns:
            trades[col] = trades[col].dt.tz_localize(None)
        trades.to_excel(writer, sheet_name="Trades", index=False)

        # ── Sheet 2: Metrics ───────────────────────────────────────────────────
        m = vp.metrics
        metrics_rows = [
            ("Total Profit ($)",        round(m.get("total_profit", 0), 2)),
            ("CAGR (%)",                round(m.get("cagr", 0), 4)),
            ("Avg Annual Return (%)",   round(m.get("yearly_avg_pct_return", 0), 4)),
            ("Sharpe Ratio",            round(m.get("sharpe_ratio", 0), 4)),
            ("Profit Factor",           round(m.get("profit_factor", 0), 4)),
            ("Return / DD Ratio",       round(m.get("return_dd_ratio", 0), 4)),
            ("Win Rate (%)",            round(m.get("winning_percentage", 0), 2)),
            ("Max Drawdown ($)",        round(m.get("drawdown", 0), 2)),
            ("Max Drawdown (%)",        round(m.get("pct_drawdown", 0), 4)),
            ("Average Trade ($)",       round(m.get("average_trade", 0), 2)),
            ("Num Trades",              int(m.get("num_trades", 0))),
            ("Scale Factor",            round(vp.scale_factor, 6)),
            ("Critical Day",            str(vp.critical_day)),
            ("Worst Day Loss ($)",      round(vp.worst_day_loss, 2)),
            ("Daily Loss Limit ($)",    config.daily_loss_limit_usd),
            ("Total DD Limit ($)",      config.total_drawdown_limit_usd),
            ("DD Status",               "PASS" if not vp.dd_failed else "FAIL"),
        ]
        pd.DataFrame(metrics_rows, columns=["Metric", "Value"]).to_excel(
            writer, sheet_name="Metrics", index=False,
        )

        # ── Sheet 3: Weights & Sizing ──────────────────────────────────────────
        weight_rows = [
            {
                "Strategy":       _short(name),
                "Asset":          _asset(name, strategies),
                "Weight":         round(vp.weights.get(name, 0), 6),
                "Risk/Trade ($)": round(vp.risk_per_trade.get(name, 0), 2),
            }
            for name in cr.combination
        ]
        weight_rows.append({
            "Strategy":       "TOTAL",
            "Asset":          "",
            "Weight":         1.0,
            "Risk/Trade ($)": round(sum(vp.risk_per_trade.values()), 2),
        })
        pd.DataFrame(weight_rows).to_excel(
            writer, sheet_name="Weights & Sizing", index=False,
        )

        # ── Sheet 4: Stress Test ───────────────────────────────────────────────
        if vp.stress:
            st = vp.stress
            stress_rows = [
                ("Passed",                  st.passed),
                ("Worst Day Loss MAE ($)",  round(st.worst_day_loss_mae, 2)),
                ("Critical Day (MAE)",      str(st.critical_day_mae)),
                ("Max Drawdown MAE ($)",    round(st.max_drawdown_usd_mae, 2)),
                ("Max Drawdown MAE (%)",    round(st.max_drawdown_pct_mae, 4)),
                ("MAE Coverage (%)",        round(st.mae_coverage_pct, 2)),
                ("Trades with MAE",         st.trades_with_mae),
                ("Total Trades",            st.trades_total),
            ]
            pd.DataFrame(stress_rows, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="Stress Test", index=False,
            )


# ── PDF writer ──────────────────────────────────────────────────────────────────

def _write_pdf(
    vp,
    cr: CombinationResult,
    config: PortfolioConfig,
    strategies: dict,
    out_dir: str,
) -> None:
    path = os.path.join(out_dir, "report.pdf")
    with PdfPages(path) as pdf:
        _page_cover(pdf, vp, cr, config, strategies)
        _page_equity_drawdown(pdf, vp, cr)
        _page_monthly_heatmap(pdf, vp, cr)
        _page_metrics_table(pdf, vp, cr, config)
        _page_weights(pdf, vp, cr, strategies)

        d = pdf.infodict()
        d["Title"]   = f"AlphaForge — Portfolio Report — Combination #{cr.rank}"
        d["Author"]  = "AlphaForge"
        d["Subject"] = f"Combination #{cr.rank} — {METHOD_FULL.get(vp.method)}"
        d["Creator"] = "AlphaForge / matplotlib"


# ── Public API ──────────────────────────────────────────────────────────────────

def export_combination(
    cr: CombinationResult,
    config: PortfolioConfig,
    strategies: dict,
    verbose: bool = True,
) -> None:
    """Export all valid method portfolios for one combination."""
    for method in METHODS:
        vp = cr.portfolios.get(method)
        if vp is None:
            continue
        out_dir = ensure_method_dir(cr.rank, method)
        label   = METHOD_FULL.get(method, method)
        if verbose:
            print(f"    Combination #{cr.rank:03d} / {label} …")
        try:
            _write_excel(vp, cr, config, strategies, out_dir)
            _write_pdf(vp, cr, config, strategies, out_dir)
        except Exception as e:
            if verbose:
                print(f"      ERROR exporting {label}: {e}")


def export_all(
    combinations: list[CombinationResult],
    config: PortfolioConfig,
    strategies: dict,
    verbose: bool = True,
) -> None:
    """Export every valid portfolio to the portfolios/ output directory."""
    if not combinations:
        return
    n_portfolios = sum(cr.n_valid_methods for cr in combinations)
    if verbose:
        print(f"\n  Exporting {len(combinations)} combination(s), "
              f"{n_portfolios} portfolio(s) to {PORTFOLIOS_OUTPUT}/")
    for cr in combinations:
        export_combination(cr, config, strategies, verbose=verbose)
    if verbose:
        print(f"  Export complete.\n")
