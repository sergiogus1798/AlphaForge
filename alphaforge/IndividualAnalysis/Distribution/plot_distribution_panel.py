"""
plot_distribution_panel.py — Matplotlib interactive distribution dashboard.

Replaces the Dash/Plotly browser-based dashboard with a native dark-theme
matplotlib panel that matches the style of all other AlphaForge modules.

Panels (Overview mode):
  [Trade Distribution]  [Win vs Loss]
  [Rolling KDE]         [MAE / MFE]

Panels (Rolling Breakdown mode):
  Up to 6 period subplots, each with histogram + KDE + normal fit.

Controls:
  ▾ Strategy  |  Periods: [2][3][4][5][6]  |  [Overview] [Rolling]  |  [? Explain]  [▶ Run]
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import font as tk_font

import matplotlib
import matplotlib.dates
import matplotlib.ticker
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── Palette ───────────────────────────────────────────────────────────────────

_BG      = "#0d1117"
_AX      = "#161b22"
_BORDER  = "#21262d"
_ACCENT  = "#00d4ff"
_GREEN   = "#3fb950"
_RED     = "#f85149"
_AMBER   = "#d29922"
_PURPLE  = "#bc8cff"
_TEXT    = "#c9d1d9"
_DIM     = "#8b949e"
_YELLOW  = "#e6c84a"
_ORANGE  = "#ff9f43"
_GRID    = "#1f2937"

PERIOD_COLORS = ["#00d4ff", "#d29922", "#3fb950", "#bc8cff", "#f85149", "#e6c84a"]

_BTN_KW  = dict(color=_BORDER, hovercolor=_BORDER)
_BTN_TXT = dict(color=_TEXT,   fontsize=9, fontfamily="monospace")

_EXPLAIN_TEXT = """\
DISTRIBUTION PANEL — User Guide
════════════════════════════════════════════════════

OVERVIEW MODE (4 panels)
─────────────────────────────────────────────────────

▸ Trade Distribution (top-left)
  Histogram of all trade PnL (green = profit, red = loss).
  Amber line: KDE (kernel density estimate) — the true
  shape of your distribution. Purple dashed line: what a
  Gaussian with the same mean/std would look like.
  Watch the divergence between KDE and Normal:
    - Positive skew (tail right)  → upside surprises
    - Negative skew (tail left)   → hidden downside risk
    - High kurtosis               → fat tails, rare big moves
  Vertical lines:
    Yellow   = mean trade
    Orange   = VaR 99% (worst 1% of trades)
    Cyan     = 95% confidence interval

▸ Win vs Loss (top-right)
  Overlaid KDEs for winning and losing trades separately.
  The area overlap tells you about trade clarity: a wide
  overlap means wins and losses are hard to distinguish
  (noisy signal). Minimal overlap = clean edge.

▸ Rolling Distribution (bottom-left)
  The full dataset split into N equal-sized time periods,
  each drawn as its own KDE. Drifts in the KDE center or
  shape over time reveal non-stationarity: your edge may
  be changing, improving, or decaying.

▸ MAE / MFE (bottom-right)
  MAE = Maximum Adverse Excursion (how far against you
  the trade went before closing — your "pain" per trade).
  MFE = Maximum Favourable Excursion (the furthest the
  trade moved in your favour — your "potential").
  Efficiency = Avg PnL / Avg MFE. High efficiency means
  you're capturing most of the available profit.
  PnL/MAE > 1 means your reward > your pain on average.

ROLLING BREAKDOWN MODE
─────────────────────────────────────────────────────
  Each panel shows one time period with its own histogram
  + KDE + normal fit. Lets you see how the distribution
  shape evolved period by period. Are the later periods
  looking more Gaussian (edge maturing)? Fewer fat tails?
  More negative skew (increasing downside risk)?

CONTROLS
─────────────────────────────────────────────────────
  ▾ Strategy  — opens strategy picker
  2/3/4/5/6   — number of time periods for rolling KDE
                and breakdown panels
  Overview    — switch to 2×2 overview mode
  Rolling     — switch to period breakdown mode
  ▶ Run       — load selected strategy and redraw
"""


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX)
    ax.tick_params(colors=_DIM, labelsize=8)
    for s in ax.spines.values():
        s.set_color(_BORDER)
    ax.xaxis.label.set_color(_DIM)
    ax.yaxis.label.set_color(_DIM)
    ax.title.set_color(_TEXT)
    ax.margins(x=0.01)


def _grid(ax: plt.Axes) -> None:
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def _stats_box(ax: plt.Axes, text: str, x: float = 0.98, y: float = 0.97,
               va: str = "top") -> None:
    ax.text(x, y, text, transform=ax.transAxes,
            va=va, ha="right", fontsize=7.5, color=_TEXT,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=_BORDER,
                      alpha=0.92, edgecolor=_ACCENT, linewidth=0.8))


def _vline(ax: plt.Axes, x: float, color: str,
           lw: float = 1.2, ls: str = "--", label: str = "") -> None:
    ax.axvline(x, color=color, linewidth=lw, linestyle=ls,
               alpha=0.85, label=label or None)


# ── Chart drawing functions ───────────────────────────────────────────────────

def _fmt_val(v: float, unit: str) -> str:
    """Format a single value in the current unit."""
    return f"\\${v:,.2f}" if unit == "$" else f"{v:.3f}%"


def _fmt_val0(v: float, unit: str) -> str:
    return f"\\${v:,.0f}" if unit == "$" else f"{v:.2f}%"


def _axis_fmt(unit: str):
    if unit == "$":
        return matplotlib.ticker.FuncFormatter(lambda x, _: f"\\${x:,.0f}")
    return matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.2f}%")


def _xlabel(unit: str) -> str:
    return "Trade PnL ($)" if unit == "$" else "Trade Return (%)"


def _draw_histogram(ax: plt.Axes, pnl: np.ndarray, title: str,
                    unit: str = "$") -> None:
    """Trade PnL histogram + KDE + normal fit + markers."""
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 2:
        ax.set_title(title, color=_TEXT, fontsize=9)
        _style_ax(ax)
        return

    counts, edges = np.histogram(pnl, bins=60)
    centers   = (edges[:-1] + edges[1:]) / 2
    bin_width = edges[1] - edges[0]
    bar_colors = [_GREEN if c >= 0 else _RED for c in centers]
    ax.bar(centers, counts, width=bin_width * 0.9,
           color=bar_colors, alpha=0.75, zorder=2)

    x_range = np.linspace(pnl.min(), pnl.max(), 600)
    scale   = len(pnl) * bin_width
    kde_y   = scipy_stats.gaussian_kde(pnl)(x_range) * scale
    norm_y  = scipy_stats.norm.pdf(x_range, pnl.mean(), pnl.std()) * scale

    ax.plot(x_range, kde_y,  color=_AMBER,  linewidth=2.0, label="KDE",    zorder=4)
    ax.fill_between(x_range, norm_y, alpha=0.08, color=_PURPLE, zorder=3)
    ax.plot(x_range, norm_y, color=_PURPLE, linewidth=1.5, linestyle="--",
            label="Normal", zorder=4)

    mu       = float(pnl.mean())
    sigma    = float(pnl.std())
    skewness = float(scipy_stats.skew(pnl))
    kurt     = float(scipy_stats.kurtosis(pnl))
    var_99   = float(np.percentile(pnl, 1))
    ci_low   = float(np.percentile(pnl, 2.5))
    ci_high  = float(np.percentile(pnl, 97.5))

    _vline(ax, 0,      "white",  lw=1.6, ls="-")
    _vline(ax, mu,     _YELLOW,  lw=1.3, ls="--",  label=f"Mean {_fmt_val0(mu, unit)}")
    _vline(ax, var_99, _ORANGE,  lw=1.3, ls=":",   label=f"VaR99% {_fmt_val0(var_99, unit)}")
    _vline(ax, ci_low, _ACCENT,  lw=1.0, ls="--",  label="95% CI")
    _vline(ax, ci_high,_ACCENT,  lw=1.0, ls="--")

    skew_w = "  !" if abs(skewness) > 0.5 else ""
    kurt_w = "  !" if abs(kurt)     > 1.0 else ""
    _stats_box(ax,
        f"mu    = {_fmt_val(mu, unit)}\n"
        f"sigma = {_fmt_val(sigma, unit)}\n"
        f"VaR99 = {_fmt_val(var_99, unit)}\n"
        f"Skew  = {skewness:.3f}{skew_w}\n"
        f"Kurt  = {kurt:.3f}{kurt_w}")

    ax.legend(fontsize=7, labelcolor=_DIM, framealpha=0.0, loc="center right")
    ax.set_title(title, color=_TEXT, fontsize=9, pad=4)
    ax.set_xlabel(_xlabel(unit), fontsize=8)
    ax.set_ylabel("Count",       fontsize=8)
    _style_ax(ax); _grid(ax)
    ax.xaxis.set_major_formatter(_axis_fmt(unit))


def _draw_win_loss(ax: plt.Axes, pnl: np.ndarray, title: str,
                   unit: str = "$") -> None:
    """Overlaid KDE for wins and losses."""
    pnl    = pnl[~np.isnan(pnl)]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(wins) < 2 or len(losses) < 2:
        ax.set_title(title, color=_TEXT, fontsize=9); _style_ax(ax); return

    x_range = np.linspace(pnl.min(), pnl.max(), 600)

    for subset, color, label in [(wins, _GREEN, "Wins"), (losses, _RED, "Losses")]:
        kde_y = scipy_stats.gaussian_kde(subset)(x_range)
        ax.fill_between(x_range, kde_y, alpha=0.18, color=color)
        ax.plot(x_range, kde_y, color=color, linewidth=2.0, label=label)
        ax.axvline(float(subset.mean()), color=color,
                   linewidth=1.2, linestyle="--", alpha=0.8)

    ax.axvline(0, color="white", linewidth=1.6)

    ratio  = len(wins) / len(losses) if losses.size else float("nan")
    payoff = (wins.mean() / abs(losses.mean())
              if wins.size and losses.size else float("nan"))
    _stats_box(ax,
        f"Wins   n={len(wins):,}  avg={_fmt_val(wins.mean(), unit)}\n"
        f"Losses n={len(losses):,}  avg={_fmt_val(losses.mean(), unit)}\n"
        f"W/L ratio: {ratio:.2f}   Payoff: {payoff:.2f}")

    ax.legend(fontsize=7, labelcolor=_DIM, framealpha=0.0, loc="upper left")
    ax.set_title(title, color=_TEXT, fontsize=9, pad=4)
    ax.set_xlabel(_xlabel(unit), fontsize=8)
    ax.set_ylabel("Density",     fontsize=8)
    _style_ax(ax); _grid(ax)
    ax.xaxis.set_major_formatter(_axis_fmt(unit))


def _draw_rolling_dist(ax: plt.Axes, df: pd.DataFrame,
                       n_periods: int, title: str,
                       unit: str = "$") -> None:
    """Overlaid KDE per time period."""
    df = df.sort_values("Close time").reset_index(drop=True)
    chunks = np.array_split(df, n_periods)
    pnl_all = df["Profit/Loss"].dropna().values
    if len(pnl_all) < 2:
        ax.set_title(title, color=_TEXT, fontsize=9); _style_ax(ax); return

    x_range = np.linspace(pnl_all.min(), pnl_all.max(), 600)

    for i, chunk in enumerate(chunks):
        pnl = chunk["Profit/Loss"].dropna().values
        if len(pnl) < 2:
            continue
        color = PERIOD_COLORS[i % len(PERIOD_COLORS)]
        start = pd.to_datetime(chunk["Close time"].min()).strftime("%Y-%m")
        end   = pd.to_datetime(chunk["Close time"].max()).strftime("%Y-%m")
        label = (f"P{i+1} {start}\u2192{end}  n={len(pnl)}"
                 f"  avg={_fmt_val0(pnl.mean(), unit)}")
        kde_y = scipy_stats.gaussian_kde(pnl)(x_range)
        ax.fill_between(x_range, kde_y, alpha=0.10, color=color)
        ax.plot(x_range, kde_y, color=color, linewidth=1.8, label=label)
        ax.axvline(float(pnl.mean()), color=color,
                   linewidth=1.0, linestyle=":", alpha=0.7)

    ax.axvline(0, color="white", linewidth=1.6)
    ax.legend(fontsize=6.5, labelcolor=_DIM, framealpha=0.0,
              loc="upper right", handlelength=1.2)
    ax.set_title(title, color=_TEXT, fontsize=9, pad=4)
    ax.set_xlabel(_xlabel(unit), fontsize=8)
    ax.set_ylabel("Density",     fontsize=8)
    _style_ax(ax); _grid(ax)
    ax.xaxis.set_major_formatter(_axis_fmt(unit))


def _draw_mae_mfe(ax: plt.Axes, df: pd.DataFrame, title: str,
                  unit: str = "$") -> None:
    """Overlaid KDE for MAE vs MFE + avg PnL marker."""
    mae_col = next((c for c in df.columns if "mae" in c.lower()), None)
    mfe_col = next((c for c in df.columns if "mfe" in c.lower()), None)
    if mae_col is None or mfe_col is None:
        ax.set_title(f"{title}  (no MAE/MFE columns)", color=_DIM, fontsize=9)
        _style_ax(ax); return

    mae = df[mae_col].dropna().abs().values
    mfe = df[mfe_col].dropna().abs().values
    pnl = df["Profit/Loss"].dropna().values
    if len(mae) < 2 or len(mfe) < 2:
        ax.set_title(title, color=_TEXT, fontsize=9); _style_ax(ax); return

    x_max   = max(mae.max(), mfe.max())
    x_range = np.linspace(0, x_max, 600)

    for values, color, label in [(mae, _RED, "MAE"), (mfe, _GREEN, "MFE")]:
        kde_y = scipy_stats.gaussian_kde(values)(x_range)
        ax.fill_between(x_range, kde_y, alpha=0.18, color=color)
        ax.plot(x_range, kde_y, color=color, linewidth=2.0, label=label)
        ax.axvline(float(values.mean()), color=color,
                   linewidth=1.2, linestyle="--", alpha=0.8)

    avg_pnl = float(pnl.mean())
    avg_mae = float(mae.mean())
    avg_mfe = float(mfe.mean())
    ax.axvline(avg_pnl, color=_YELLOW, linewidth=1.3, linestyle=":")

    eff = (avg_pnl / avg_mfe * 100) if avg_mfe > 0 else float("nan")
    cap = (avg_pnl / avg_mae * 100) if avg_mae > 0 else float("nan")
    _stats_box(ax,
        f"Avg MAE = {_fmt_val(avg_mae, unit)}\n"
        f"Avg MFE = {_fmt_val(avg_mfe, unit)}\n"
        f"Avg PnL = {_fmt_val(avg_pnl, unit)}\n"
        f"Efficiency (PnL/MFE) = {eff:.1f}%\n"
        f"PnL/MAE ratio        = {cap:.1f}%",
        x=0.98, y=0.5, va="center")

    ax.legend(fontsize=7, labelcolor=_DIM, framealpha=0.0, loc="upper right")
    ax.set_title(title, color=_TEXT, fontsize=9, pad=4)
    xlabel = "$ amount" if unit == "$" else "% amount"
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density",  fontsize=8)
    _style_ax(ax); _grid(ax)
    ax.xaxis.set_major_formatter(_axis_fmt(unit))


def _draw_equity(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    """Cumulative PnL equity curve with drawdown shading."""
    df_s = df.sort_values("Close time").reset_index(drop=True)
    pnl  = df_s["Profit/Loss"].dropna()
    if len(pnl) < 2:
        ax.set_title(title, color=_TEXT, fontsize=9); _style_ax(ax); return

    try:
        dates = pd.to_datetime(df_s.loc[pnl.index, "Close time"])
        x_vals = dates.values
        use_dates = True
    except Exception:
        x_vals = np.arange(len(pnl))
        use_dates = False

    eq          = pnl.cumsum().values
    running_max = np.maximum.accumulate(eq)
    drawdown    = eq - running_max

    # Equity line
    ax.plot(x_vals, eq, color=_GREEN, linewidth=1.4, zorder=3)
    ax.fill_between(x_vals, eq, 0,
                    where=eq >= 0, alpha=0.10, color=_GREEN, zorder=2)
    ax.fill_between(x_vals, eq, 0,
                    where=eq < 0,  alpha=0.12, color=_RED,   zorder=2)

    # Drawdown shading (between equity and its running max)
    ax.fill_between(x_vals, eq, running_max,
                    alpha=0.18, color=_RED, zorder=1, label="Drawdown")

    ax.axhline(0, color=_DIM, linewidth=0.8, linestyle="--", alpha=0.6)

    # Stats
    total_pnl = float(eq[-1])
    max_dd    = float(abs(drawdown.min())) if drawdown.min() < 0 else 0.0
    ret_dd    = (total_pnl / max_dd) if max_dd > 1e-8 else float("nan")
    peak_eq   = float(running_max.max())

    stats = (
        f"Net PnL = \\${total_pnl:,.0f}   "
        f"Peak = \\${peak_eq:,.0f}   "
        f"Max DD = \\${max_dd:,.0f}   "
        f"Return/DD = {ret_dd:.2f}"
    )
    ax.text(0.01, 0.97, stats, transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5, color=_TEXT,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=_BORDER,
                      alpha=0.88, edgecolor=_ACCENT, linewidth=0.7))

    ax.set_title(title, color=_TEXT, fontsize=9, pad=4)
    ax.set_ylabel("Cumul. PnL (\\$)", fontsize=8)
    _style_ax(ax); _grid(ax)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"\\${x:,.0f}"))
    if use_dates:
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(
            matplotlib.dates.AutoDateLocator())
        ax.tick_params(axis="x", labelsize=7)


def _draw_period(ax: plt.Axes, chunk: pd.DataFrame, color: str,
                 period_idx: int, unit: str = "$") -> None:
    """One period's histogram + KDE + normal fit."""
    pnl = chunk["Profit/Loss"].dropna().values
    if len(pnl) < 2:
        _style_ax(ax); return

    counts, edges = np.histogram(pnl, bins=40)
    centers   = (edges[:-1] + edges[1:]) / 2
    bin_width = edges[1] - edges[0]
    bar_colors = [_GREEN if c >= 0 else _RED for c in centers]
    ax.bar(centers, counts, width=bin_width * 0.9,
           color=bar_colors, alpha=0.65, zorder=2)

    x_range = np.linspace(pnl.min(), pnl.max(), 400)
    scale   = len(pnl) * bin_width
    kde_y   = scipy_stats.gaussian_kde(pnl)(x_range) * scale
    norm_y  = scipy_stats.norm.pdf(x_range, pnl.mean(), pnl.std()) * scale

    ax.plot(x_range, kde_y,  color=color,   linewidth=1.8, zorder=4)
    ax.fill_between(x_range, norm_y, alpha=0.07, color=_PURPLE, zorder=3)
    ax.plot(x_range, norm_y, color=_PURPLE, linewidth=1.2,
            linestyle="--", zorder=4)
    ax.axvline(0,              color="white", linewidth=1.4)
    ax.axvline(float(pnl.mean()), color=_YELLOW, linewidth=1.1,
               linestyle="--", alpha=0.8)

    start = pd.to_datetime(chunk["Close time"].min()).strftime("%Y-%m")
    end   = pd.to_datetime(chunk["Close time"].max()).strftime("%Y-%m")
    net   = pnl.sum()
    win   = (pnl > 0).mean() * 100
    eq    = pd.Series(pnl).cumsum()
    mdd   = float((eq.cummax() - eq).max())

    title = (f"P{period_idx+1}  {start} \u2192 {end}  |  {len(pnl)} trades  |  "
             f"Net: {_fmt_val0(net, unit)}  Win: {win:.1f}%  MDD: {_fmt_val0(mdd, unit)}")
    ax.set_title(title, color=color, fontsize=7.5, pad=3)
    ax.set_xlabel(_xlabel(unit), fontsize=7)
    _style_ax(ax); _grid(ax)
    ax.xaxis.set_major_formatter(_axis_fmt(unit))
    ax.tick_params(labelsize=7)


# ── Metrics strip ─────────────────────────────────────────────────────────────

def _draw_metrics(fig: plt.Figure, df: pd.DataFrame,
                  initial_capital: float, strat_name: str,
                  metric_texts: list) -> None:
    """Draw / refresh the metrics strip (top of figure)."""
    for t in metric_texts:
        try:
            t.remove()
        except Exception:
            pass
    metric_texts.clear()

    try:
        from alphaforge.metrics import compute_metrics
        m = compute_metrics(df, initial_capital=initial_capital)
    except Exception:
        m = {}

    pnl  = df["Profit/Loss"].dropna().values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    # Return/DD
    eq      = pnl.cumsum()
    max_dd  = float(abs((eq - np.maximum.accumulate(eq)).min())) if len(eq) else 0.0
    ret_dd  = (pnl.sum() / max_dd) if max_dd > 1e-8 else float("nan")
    ret_dd_str = f"{ret_dd:.2f}" if not math.isnan(ret_dd) else "N/A"

    items = [
        ("Net P&L",      f"\\${m.get('total_profit', pnl.sum()):,.0f}",
         m.get('total_profit', pnl.sum()) >= 0),
        ("Avg Trade",    f"\\${m.get('average_trade', pnl.mean()):,.2f}",
         m.get('average_trade', pnl.mean()) >= 0),
        ("Win Rate",     f"{m.get('winning_percentage', (pnl>0).mean()*100):.1f}%",
         m.get('winning_percentage', (pnl>0).mean()*100) >= 50),
        ("Profit Factor",f"{m.get('profit_factor', wins.sum()/abs(losses.sum()) if losses.size else 0):.2f}",
         m.get('profit_factor', 0) >= 1.5),
        ("R Expectancy", f"{m.get('r_expectancy', 0):.3f}",
         m.get('r_expectancy', 0) >= 0),
        ("Return/DD",    ret_dd_str, ret_dd >= 1.5 if not math.isnan(ret_dd) else None),
        ("Trades",       f"{len(pnl):,}", None),
    ]

    xs = [0.07, 0.18, 0.29, 0.40, 0.51, 0.62, 0.73]
    for (label, value, positive), x in zip(items, xs):
        color = (_GREEN if positive is True else
                 _RED   if positive is False else _TEXT)
        t1 = fig.text(x, 0.955, label, ha="left", color=_DIM,
                      fontsize=7, fontfamily="monospace")
        t2 = fig.text(x, 0.935, value, ha="left", color=color,
                      fontsize=10, fontweight="bold", fontfamily="monospace")
        metric_texts.extend([t1, t2])

    # Strategy name right side
    short = strat_name.split("/")[-1]
    pair  = strat_name.split("/")[0] if "/" in strat_name else ""
    t3 = fig.text(0.98, 0.955, pair,  ha="right", color=_ACCENT,
                  fontsize=8, fontfamily="monospace")
    t4 = fig.text(0.98, 0.935, short, ha="right", color=_DIM,
                  fontsize=7, fontfamily="monospace")
    metric_texts.extend([t3, t4])


# ── Main panel ────────────────────────────────────────────────────────────────

def plot_distribution_panel(
    all_strategies: dict,
    initial_key:    str,
    initial_capital: float = 10_000.0,
) -> None:
    """
    Launch the interactive distribution panel.

    Parameters
    ----------
    all_strategies : dict
        Mapping of strategy_key → trades DataFrame (from load_folder).
    initial_key : str
        Key of the first strategy to display.
    initial_capital : float
        Used for metric calculations.
    """
    # ── State ─────────────────────────────────────────────────────────────────
    state = {
        "key":       initial_key,
        "df":        all_strategies[initial_key].sort_values("Close time").reset_index(drop=True),
        "mode":      "overview",     # "overview" | "rolling"
        "n_periods": 4,
        "unit":      "$",            # "$" | "%"
    }
    _last_run = {"key": initial_key}

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=_BG)
    fig.canvas.manager.set_window_title("AlphaForge — Distribution")

    # Track non-control elements to clear between redraws
    _plot_axes:   list = []
    _metric_texts: list = []

    # ── Controls layout ───────────────────────────────────────────────────────
    CTRL_Y  = 0.02
    CTRL_H  = 0.052
    ctrl_axes = []

    def _make_btn(x, w, label, callback, color=None):
        ax = fig.add_axes([x, CTRL_Y, w, CTRL_H])
        ax.set_facecolor(color or _BORDER)
        for sp in ax.spines.values():
            sp.set_color(_DIM)
        btn = Button(ax, label, color=color or _BORDER, hovercolor="#2d333b")
        btn.label.set(**_BTN_TXT)
        btn.on_clicked(callback)
        ctrl_axes.append(ax)
        return btn, ax

    # Strategy selector
    _strat_label_text = [None]

    btn_strat, _ = _make_btn(0.02, 0.11, u"\u25be Select Strategy",
                             lambda _: None)  # filled below

    strat_label_ax = fig.add_axes([0.14, CTRL_Y, 0.09, CTRL_H])
    strat_label_ax.set_facecolor(_BG)
    for sp in strat_label_ax.spines.values():
        sp.set_visible(False)
    _strat_label_text[0] = strat_label_ax.text(
        0, 0.5, state["key"].split("/")[-1][:20],
        color=_ACCENT, fontsize=7.5, fontfamily="monospace",
        va="center", ha="left", transform=strat_label_ax.transAxes)
    ctrl_axes.append(strat_label_ax)

    # Period buttons
    fig.text(0.245, CTRL_Y + CTRL_H * 0.65, "Periods:", color=_DIM,
             fontsize=8, fontfamily="monospace")
    period_btns = {}
    for i, n in enumerate([2, 3, 4, 5, 6]):
        x = 0.300 + i * 0.038
        btn, ax_p = _make_btn(x, 0.030, str(n), lambda _, n=n: _set_periods(n))
        period_btns[n] = (btn, ax_p)

    # Mode buttons
    _mode_axes = {}
    btn_ov,  ax_ov  = _make_btn(0.495, 0.074, "Overview", lambda _: _set_mode("overview"))
    btn_rol, ax_rol = _make_btn(0.573, 0.074, "Rolling",  lambda _: _set_mode("rolling"))
    _mode_axes["overview"] = ax_ov
    _mode_axes["rolling"]  = ax_rol

    # Unit toggle [$] [%]
    _unit_axes = {}
    btn_usd, ax_usd = _make_btn(0.651, 0.030, "[$]", lambda _: _set_unit("$"))
    btn_pct, ax_pct = _make_btn(0.684, 0.030, "[%]", lambda _: _set_unit("%"))
    _unit_axes["$"] = ax_usd
    _unit_axes["%"] = ax_pct

    # Equity launcher
    import subprocess, sys as _sys, pathlib as _pathlib
    _root_dir   = str(_pathlib.Path(__file__).resolve().parents[3])
    _run_equity = str(_pathlib.Path(_root_dir) / "run_equity.py")

    def _open_equity(_e=None):
        try:
            subprocess.Popen(
                [_sys.executable, _run_equity, state["key"]],
                cwd=_root_dir,
            )
        except Exception as exc:
            print(f"[Equity] launch failed: {exc}")

    btn_eq, _ = _make_btn(0.718, 0.060, "Equity", _open_equity)

    # Explain + Run
    btn_exp, _ = _make_btn(0.782, 0.074, "? Explain", lambda _: _show_explain())
    btn_run, _ = _make_btn(0.860, 0.082, u"\u25b6 Run",   lambda _: _on_run())

    # ── Visual state for active buttons ───────────────────────────────────────
    def _refresh_btn_highlights():
        for n, (btn, ax_p) in period_btns.items():
            c = _ACCENT if n == state["n_periods"] else _BORDER
            ax_p.set_facecolor(c)
            btn.color = c
        for mode, ax_m in _mode_axes.items():
            c = _ACCENT if mode == state["mode"] else _BORDER
            ax_m.set_facecolor(c)
        for u, ax_u in _unit_axes.items():
            c = _AMBER if u == state["unit"] else _BORDER
            ax_u.set_facecolor(c)

    # ── Redraw ────────────────────────────────────────────────────────────────
    def _redraw():
        # Remove old plot axes
        for ax in list(_plot_axes):
            try:
                ax.remove()
            except Exception:
                pass
        _plot_axes.clear()

        df        = state["df"]
        mode      = state["mode"]
        n_periods = state["n_periods"]
        unit      = state["unit"]
        key       = state["key"]
        sym       = key.split("/")[0] if "/" in key else key
        short     = key.split("/")[-1]

        # Build display dataframe (scale to % if needed)
        if unit == "%":
            scale = 100.0 / initial_capital
            df_display = df.copy()
            df_display["Profit/Loss"] = df_display["Profit/Loss"] * scale
            for col in df_display.columns:
                if "mae" in col.lower() or "mfe" in col.lower():
                    df_display[col] = df_display[col] * scale
        else:
            df_display = df

        # Refresh metrics (always in $)
        _draw_metrics(fig, df, initial_capital, key, _metric_texts)

        # Refresh strategy label
        _strat_label_text[0].set_text(short[:22])

        _refresh_btn_highlights()

        pnl = df_display["Profit/Loss"].dropna().values

        if mode == "overview":
            gs = gridspec.GridSpec(
                2, 2, figure=fig,
                left=0.05, right=0.98, top=0.87, bottom=0.12,
                hspace=0.46, wspace=0.26,
            )
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            _plot_axes.extend([ax1, ax2, ax3, ax4])

            _draw_histogram(ax1, pnl,
                            f"Trade Distribution — {short} | {sym}", unit=unit)
            _draw_win_loss(ax2, pnl,
                            f"Win vs Loss — {short} | {sym}", unit=unit)
            _draw_rolling_dist(ax3, df_display, n_periods,
                               f"Rolling Distribution ({n_periods} periods) — {short}",
                               unit=unit)
            _draw_mae_mfe(ax4, df_display,
                           f"MAE / MFE — {short} | {sym}", unit=unit)

        else:  # rolling breakdown
            n = n_periods
            ncols = 2 if n <= 4 else 3
            nrows = math.ceil(n / ncols)
            gs = gridspec.GridSpec(
                nrows, ncols, figure=fig,
                left=0.05, right=0.98, top=0.92, bottom=0.12,
                hspace=0.55, wspace=0.26,
            )
            df_s   = df_display.sort_values("Close time").reset_index(drop=True)
            chunks = np.array_split(df_s, n)
            for i, chunk in enumerate(chunks):
                row = i // ncols
                col = i % ncols
                ax  = fig.add_subplot(gs[row, col])
                _plot_axes.append(ax)
                _draw_period(ax, chunk, PERIOD_COLORS[i % len(PERIOD_COLORS)], i,
                             unit=unit)

            # Hide unused axes if n is odd with 3 cols
            for j in range(n, nrows * ncols):
                ax_blank = fig.add_subplot(gs[j // ncols, j % ncols])
                ax_blank.set_visible(False)
                _plot_axes.append(ax_blank)

        fig.canvas.draw_idle()

    # ── Event handlers ────────────────────────────────────────────────────────
    def _set_periods(n: int):
        state["n_periods"] = n
        _redraw()

    def _set_mode(mode: str):
        state["mode"] = mode
        _redraw()

    def _set_unit(unit: str):
        state["unit"] = unit
        _redraw()

    def _on_run():
        key = state["key"]
        if key != _last_run["key"]:
            state["df"] = (all_strategies[key]
                           .sort_values("Close time")
                           .reset_index(drop=True))
            _last_run["key"] = key
        _redraw()

    def _open_strategy_selector(_event=None):
        root = tk.Tk()
        root.title("Select Strategy")
        root.configure(bg="#1a1a2e")
        root.geometry("500x400")

        filter_var = tk.StringVar()
        tk.Label(root, text="Filter:", bg="#1a1a2e", fg="#c9d1d9",
                 font=("Courier New", 10)).pack(padx=8, pady=(8, 2), anchor="w")
        entry = tk.Entry(root, textvariable=filter_var,
                         bg="#0f3460", fg="#c9d1d9", insertbackground="#c9d1d9",
                         font=("Courier New", 10))
        entry.pack(fill=tk.X, padx=8, pady=(0, 4))

        lb_frame = tk.Frame(root, bg="#1a1a2e")
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        scrollbar = tk.Scrollbar(lb_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(lb_frame, yscrollcommand=scrollbar.set,
                        bg="#0f3460", fg="#c9d1d9", selectbackground="#00d4ff",
                        selectforeground="#0d1117", font=("Courier New", 9),
                        activestyle="none")
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=lb.yview)

        all_keys = sorted(all_strategies.keys())

        def _populate(keys):
            lb.delete(0, tk.END)
            for k in keys:
                lb.insert(tk.END, k)

        _populate(all_keys)

        def _on_filter(*_):
            q = filter_var.get().lower()
            _populate([k for k in all_keys if q in k.lower()])

        filter_var.trace_add("write", _on_filter)

        def _on_select(_event=None):
            sel = lb.curselection()
            if not sel:
                return
            chosen = lb.get(sel[0])
            state["key"] = chosen
            _strat_label_text[0].set_text(chosen.split("/")[-1][:22])
            fig.canvas.draw_idle()
            root.destroy()

        lb.bind("<Double-1>", _on_select)
        tk.Button(root, text="Select", command=_on_select,
                  bg="#00d4ff", fg="#0d1117", font=("Courier New", 10, "bold")
                  ).pack(pady=(0, 8))
        root.mainloop()

    btn_strat.on_clicked(lambda _: _open_strategy_selector())

    def _show_explain(_event=None):
        root = tk.Tk()
        root.title("Distribution Panel — Explain")
        root.configure(bg="#0d1117")
        root.geometry("640x620")

        frame = tk.Frame(root, bg="#0d1117")
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frame, yscrollcommand=sb.set,
                      bg="#161b22", fg="#c9d1d9", insertbackground="#c9d1d9",
                      font=("Courier New", 9), wrap=tk.WORD, relief=tk.FLAT)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=txt.yview)
        txt.insert(tk.END, _EXPLAIN_TEXT)
        txt.config(state=tk.DISABLED)
        root.mainloop()

    # ── Initial draw ──────────────────────────────────────────────────────────
    # Top separator line
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.92, 0.92],
                              transform=fig.transFigure,
                              color=_BORDER, linewidth=0.8))
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.105, 0.105],
                              transform=fig.transFigure,
                              color=_BORDER, linewidth=0.8))

    _on_run()
    plt.show()
