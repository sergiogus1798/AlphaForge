"""
plot_equity_panel.py — Matplotlib interactive equity curve panel.

Two stacked panels sharing the x-axis:
  Top (70%): Equity curve + benchmark line + stats box
  Bottom (30%): Drawdown %

Controls:
  ▾ Strategy | [3M] [6M] [1Y] [3Y] [All] | [? Explain] [▶ Run]

Metrics strip: Net P&L | CAGR | Sharpe | Max DD | Win Rate | Return/DD | Trades
"""

from __future__ import annotations

import math
import tkinter as tk

import matplotlib
import matplotlib.dates
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from alphaforge.metrics import compute_metrics, sharpe_ratio


# ── Palette ───────────────────────────────────────────────────────────────────

_BG      = "#0d1117"
_AX      = "#161b22"
_BORDER  = "#21262d"
_ACCENT  = "#00d4ff"
_GREEN   = "#3fb950"
_RED     = "#f85149"
_TEXT    = "#c9d1d9"
_DIM     = "#8b949e"
_YELLOW  = "#e6c84a"
_BENCH   = "#6e7681"
_GRID    = "#1f2937"
_DD_FILL = "#f8514922"

_BTN_KW = dict(color=_BORDER, hovercolor="#2d333b")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(_AX)
    ax.tick_params(colors=_DIM, labelsize=8)
    for s in ax.spines.values():
        s.set_color(_BORDER)
    ax.xaxis.label.set_color(_DIM)
    ax.yaxis.label.set_color(_DIM)
    ax.title.set_color(_TEXT)
    ax.margins(x=0.01)


def _grid(ax):
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def _fmt(v, fmt=".2f", prefix="", suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{prefix}{v:{fmt}}{suffix}"


def _filter_range(df: pd.DataFrame, rng: str) -> pd.DataFrame:
    if rng == "All":
        return df
    end   = df["Close time"].max()
    delta = {"3M": pd.DateOffset(months=3),
             "6M": pd.DateOffset(months=6),
             "1Y": pd.DateOffset(years=1),
             "3Y": pd.DateOffset(years=3)}[rng]
    return df[df["Close time"] >= end - delta].reset_index(drop=True)


def _equity_series(df: pd.DataFrame, initial_capital: float):
    s      = df.sort_values("Close time").reset_index(drop=True)
    dates  = pd.to_datetime(s["Close time"])
    equity = initial_capital + s["Profit/Loss"].cumsum().values
    return dates, equity


def _benchmark(dates: pd.Series, initial_capital: float,
               annual_return: float = 0.05) -> np.ndarray:
    t0    = dates.iloc[0]
    years = (dates - t0).dt.total_seconds() / (365.25 * 86_400)
    return initial_capital * (1 + annual_return) ** years.values


def _sortino(df: pd.DataFrame, initial_capital: float = 10_000) -> float:
    start = df["Close time"].min().date()
    end   = df["Close time"].max().date()
    daily = (
        df.groupby(df["Close time"].dt.date)["Profit/Loss"]
        .sum()
        .reindex(pd.bdate_range(start, end).date, fill_value=0.0)
    )
    rf_d   = initial_capital * 0.01 / 252
    excess = daily - rf_d
    neg    = excess[excess < 0]
    if len(neg) < 2:
        return float("nan")
    dstd = float(np.sqrt((neg ** 2).mean()))
    return float((daily.mean() - rf_d) / dstd * np.sqrt(252)) if dstd > 0 else float("nan")


def _calmar(df: pd.DataFrame, initial_capital: float) -> float:
    try:
        from alphaforge.metrics import cagr as _cagr, pct_drawdown as _dd
        c = _cagr(df, initial_capital=initial_capital)
        d = _dd(df,  initial_capital=initial_capital)
        return float(c / d) if d and d > 0 else float("nan")
    except Exception:
        return float("nan")


_EXPLAIN_TEXT = """\
EQUITY CURVE PANEL — User Guide
════════════════════════════════════════════════════

TOP PANEL — Equity Curve
─────────────────────────────────────────────────────
  Shows cumulative PnL added to the starting capital
  over time. Each point represents the account value
  after closing that trade.

  Cyan line     = your strategy's equity
  Grey dashed   = simple buy-and-hold benchmark (5%/yr)

  When your equity is above the benchmark line,
  you are generating excess return (alpha). When it
  falls below, the strategy is underperforming passive
  exposure.

  Green fill    = equity above starting capital
  Red fill      = equity below starting capital

BOTTOM PANEL — Drawdown %
─────────────────────────────────────────────────────
  Shows how far (in %) the equity has fallen from its
  most recent peak at any given point in time.

  - Max Drawdown is the deepest the red ever goes.
  - Duration matters as much as depth: a long shallow
    drawdown is often more damaging psychologically
    than a deep but brief one.
  - A drawdown never rising back to 0 means the
    strategy has not recovered to new highs.

METRICS STRIP
─────────────────────────────────────────────────────
  Net P&L    — total $ profit/loss
  CAGR       — compound annual growth rate
  Sharpe     — risk-adjusted return (>1 is good)
  Max DD     — peak-to-trough equity drop (%)
  Win Rate   — % of trades closed in profit
  Return/DD  — net profit / max drawdown in $
               (similar to Calmar but using raw $)
  Trades     — number of closed trades

RANGE BUTTONS
─────────────────────────────────────────────────────
  Filter the chart to the most recent N months/years.
  The metrics strip always shows full-history values.
"""


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_panels(ax_eq, ax_dd, df_full, df_view,
                 initial_capital, benchmark_return, short, sym):
    """Draw equity (ax_eq) and drawdown (ax_dd) panels."""

    # ── Equity ────────────────────────────────────────────────────────────────
    ax_eq.cla()
    dates, equity = _equity_series(df_view, initial_capital)
    bench         = _benchmark(dates, initial_capital, benchmark_return)

    ax_eq.plot(dates, equity, color=_ACCENT, linewidth=1.6,
               label="Equity", zorder=4)
    ax_eq.fill_between(dates, equity, initial_capital,
                        where=equity >= initial_capital,
                        alpha=0.10, color=_GREEN, zorder=2)
    ax_eq.fill_between(dates, equity, initial_capital,
                        where=equity < initial_capital,
                        alpha=0.15, color=_RED, zorder=2)
    ax_eq.plot(dates, bench, color=_BENCH, linewidth=1.2,
               linestyle="--", alpha=0.7,
               label=f"Benchmark ({benchmark_return*100:.0f}%/yr)", zorder=3)
    ax_eq.axhline(initial_capital, color=_DIM,
                  linewidth=0.8, linestyle=":", alpha=0.5)

    # Stats text box
    m         = compute_metrics(df_full, initial_capital=initial_capital)
    shr       = sharpe_ratio(df_full)
    sor       = _sortino(df_full, initial_capital)
    cal       = _calmar(df_full, initial_capital)
    pnl_all   = df_full["Profit/Loss"].dropna().values
    eq_all    = pnl_all.cumsum()
    max_dd_d  = float(abs((eq_all - np.maximum.accumulate(eq_all)).min()))
    ret_dd    = (pnl_all.sum() / max_dd_d) if max_dd_d > 1e-8 else float("nan")
    pnl_f     = df_view["Profit/Loss"].dropna().values
    skew      = float(scipy_stats.skew(pnl_f))
    kurt      = float(scipy_stats.kurtosis(pnl_f))
    var99     = float(np.percentile(pnl_f, 1))

    stats = (
        f"Sharpe   {_fmt(shr, '.3f')}\n"
        f"Sortino  {_fmt(sor, '.3f')}\n"
        f"Calmar   {_fmt(cal, '.3f')}\n"
        f"Ret/DD   {_fmt(ret_dd, '.2f')}\n"
        f"VaR 99%  \\${var99:,.2f}\n"
        f"Skew     {skew:.3f}{'  !' if abs(skew)>0.5 else ''}\n"
        f"Kurt     {kurt:.3f}{'  !' if abs(kurt)>1.0 else ''}"
    )
    ax_eq.text(0.995, 0.98, stats,
               transform=ax_eq.transAxes,
               va="top", ha="right", fontsize=7.5, color=_TEXT,
               fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.5", facecolor=_BORDER,
                         alpha=0.92, edgecolor=_ACCENT, linewidth=0.8))

    ax_eq.legend(fontsize=7.5, labelcolor=_DIM, framealpha=0.0,
                 loc="upper left")
    ax_eq.set_title(f"Equity Curve — {short} | {sym}", color=_TEXT,
                    fontsize=9, pad=4)
    ax_eq.set_ylabel("Account Value (\\$)", fontsize=8)
    ax_eq.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"\\${x:,.0f}"))
    _style_ax(ax_eq); _grid(ax_eq)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax_dd.cla()
    dd_pct = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100

    ax_dd.fill_between(dates, dd_pct, 0, alpha=0.55, color=_RED, zorder=2)
    ax_dd.plot(dates, dd_pct, color=_RED, linewidth=1.0, zorder=3)
    ax_dd.axhline(0, color=_DIM, linewidth=0.8, linestyle="--", alpha=0.5)

    max_dd_pct = float(dd_pct.min())
    ax_dd.text(0.995, 0.04, f"Max DD: {max_dd_pct:.2f}%",
               transform=ax_dd.transAxes,
               va="bottom", ha="right", fontsize=7.5, color=_RED,
               fontfamily="monospace")

    ax_dd.set_ylabel("Drawdown %", fontsize=8)
    ax_dd.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_dd.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    ax_dd.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
    ax_dd.tick_params(axis="x", labelsize=7)
    _style_ax(ax_dd); _grid(ax_dd)


def _draw_metrics(fig, df, initial_capital, strat_name, metric_texts):
    for t in metric_texts:
        try:
            t.remove()
        except Exception:
            pass
    metric_texts.clear()

    m      = compute_metrics(df, initial_capital=initial_capital)
    pnl    = df["Profit/Loss"].dropna().values
    eq     = pnl.cumsum()
    max_dd = float(abs((eq - np.maximum.accumulate(eq)).min()))
    ret_dd = (pnl.sum() / max_dd) if max_dd > 1e-8 else float("nan")

    items = [
        ("Net P&L",  f"\\${m.get('total_profit', pnl.sum()):,.0f}",
         m.get("total_profit", 0) >= 0),
        ("CAGR",     f"{m.get('cagr', 0):.2f}%",
         m.get("cagr", 0) >= 0),
        ("Sharpe",   f"{m.get('sharpe_ratio', 0):.2f}",
         m.get("sharpe_ratio", 0) >= 1.0),
        ("Max DD",   f"-{m.get('pct_drawdown', 0):.2f}%", False),
        ("Win Rate", f"{m.get('winning_percentage', (pnl>0).mean()*100):.1f}%",
         m.get("winning_percentage", (pnl>0).mean()*100) >= 50),
        ("Return/DD",f"{ret_dd:.2f}" if not math.isnan(ret_dd) else "N/A",
         ret_dd >= 1.5 if not math.isnan(ret_dd) else None),
        ("Trades",   f"{len(pnl):,}", None),
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

    short = strat_name.split("/")[-1]
    pair  = strat_name.split("/")[0] if "/" in strat_name else ""
    t3 = fig.text(0.98, 0.955, pair,  ha="right", color=_ACCENT,
                  fontsize=8, fontfamily="monospace")
    t4 = fig.text(0.98, 0.935, short, ha="right", color=_DIM,
                  fontsize=7, fontfamily="monospace")
    metric_texts.extend([t3, t4])


# ── Main panel ────────────────────────────────────────────────────────────────

def plot_equity_panel(
    all_strategies:   dict,
    initial_key:      str,
    initial_capital:  float = 10_000.0,
    benchmark_return: float = 0.05,
) -> None:
    """Launch the interactive equity curve panel."""

    state = {
        "key":   initial_key,
        "df":    all_strategies[initial_key].sort_values("Close time").reset_index(drop=True),
        "range": "All",
    }
    _last_run = {"key": initial_key}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=_BG)
    fig.canvas.manager.set_window_title("AlphaForge — Equity Curve")

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.70, 0.30],
        left=0.06, right=0.98, top=0.88, bottom=0.12,
        hspace=0.08,
    )
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

    _metric_texts: list = []
    _strat_label  = [None]

    # ── Controls ──────────────────────────────────────────────────────────────
    CTRL_Y = 0.02
    CTRL_H = 0.052
    ctrl_axes = []

    def _make_btn(x, w, label, callback):
        ax = fig.add_axes([x, CTRL_Y, w, CTRL_H])
        ax.set_facecolor(_BORDER)
        for sp in ax.spines.values():
            sp.set_color(_DIM)
        btn = Button(ax, label, color=_BORDER, hovercolor="#2d333b")
        btn.label.set(color=_TEXT, fontsize=9, fontfamily="monospace")
        btn.on_clicked(callback)
        ctrl_axes.append(ax)
        return btn, ax

    btn_strat, _ = _make_btn(0.02, 0.13, "\u25be Select Strategy",
                             lambda _: None)

    lbl_ax = fig.add_axes([0.16, CTRL_Y, 0.11, CTRL_H])
    lbl_ax.set_facecolor(_BG)
    for sp in lbl_ax.spines.values():
        sp.set_visible(False)
    _strat_label[0] = lbl_ax.text(
        0, 0.5, state["key"].split("/")[-1][:24],
        color=_ACCENT, fontsize=7.5, fontfamily="monospace",
        va="center", ha="left", transform=lbl_ax.transAxes)
    ctrl_axes.append(lbl_ax)

    # Range buttons
    fig.text(0.305, CTRL_Y + CTRL_H * 0.65, "Range:", color=_DIM,
             fontsize=8, fontfamily="monospace")
    _range_axes = {}
    for i, rng in enumerate(["3M", "6M", "1Y", "3Y", "All"]):
        x = 0.360 + i * 0.052
        _, ax_r = _make_btn(x, 0.043, rng, lambda _, r=rng: _set_range(r))
        _range_axes[rng] = ax_r

    btn_exp, _ = _make_btn(0.660, 0.09, "? Explain", lambda _: _show_explain())
    btn_run, _ = _make_btn(0.760, 0.10, "\u25b6 Run",   lambda _: _on_run())

    # ── Button highlights ─────────────────────────────────────────────────────
    def _refresh_highlights():
        for rng, ax_r in _range_axes.items():
            c = _ACCENT if rng == state["range"] else _BORDER
            ax_r.set_facecolor(c)

    # ── Redraw ────────────────────────────────────────────────────────────────
    def _redraw():
        df_full = state["df"]
        df_view = _filter_range(df_full, state["range"])
        key     = state["key"]
        short   = key.split("/")[-1]
        sym     = key.split("/")[0] if "/" in key else key

        _strat_label[0].set_text(short[:24])
        _refresh_highlights()
        _draw_metrics(fig, df_full, initial_capital, key, _metric_texts)
        _draw_panels(ax_eq, ax_dd, df_full, df_view,
                     initial_capital, benchmark_return, short, sym)
        fig.canvas.draw_idle()

    # ── Handlers ──────────────────────────────────────────────────────────────
    def _set_range(rng):
        state["range"] = rng
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
        tk.Label(root, text="Filter:", bg="#1a1a2e", fg=_TEXT,
                 font=("Courier New", 10)).pack(padx=8, pady=(8, 2), anchor="w")
        entry = tk.Entry(root, textvariable=filter_var,
                         bg="#0f3460", fg=_TEXT, insertbackground=_TEXT,
                         font=("Courier New", 10))
        entry.pack(fill=tk.X, padx=8, pady=(0, 4))

        lb_frame  = tk.Frame(root, bg="#1a1a2e")
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        sb = tk.Scrollbar(lb_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(lb_frame, yscrollcommand=sb.set,
                        bg="#0f3460", fg=_TEXT, selectbackground=_ACCENT,
                        selectforeground="#0d1117",
                        font=("Courier New", 9), activestyle="none")
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=lb.yview)

        all_keys = sorted(all_strategies.keys())

        def _populate(keys):
            lb.delete(0, tk.END)
            for k in keys:
                lb.insert(tk.END, k)

        _populate(all_keys)
        filter_var.trace_add("write", lambda *_: _populate(
            [k for k in all_keys if filter_var.get().lower() in k.lower()]))

        def _select(_e=None):
            sel = lb.curselection()
            if not sel:
                return
            state["key"] = lb.get(sel[0])
            _strat_label[0].set_text(state["key"].split("/")[-1][:24])
            fig.canvas.draw_idle()
            root.destroy()

        lb.bind("<Double-1>", _select)
        tk.Button(root, text="Select", command=_select,
                  bg=_ACCENT, fg="#0d1117",
                  font=("Courier New", 10, "bold")).pack(pady=(0, 8))
        root.mainloop()

    btn_strat.on_clicked(lambda _: _open_strategy_selector())

    def _show_explain(_e=None):
        root = tk.Tk()
        root.title("Equity Curve — Explain")
        root.configure(bg="#0d1117")
        root.geometry("600x520")
        frame = tk.Frame(root, bg="#0d1117")
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frame, yscrollcommand=sb.set,
                      bg="#161b22", fg=_TEXT, insertbackground=_TEXT,
                      font=("Courier New", 9), wrap=tk.WORD, relief=tk.FLAT)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=txt.yview)
        txt.insert(tk.END, _EXPLAIN_TEXT)
        txt.config(state=tk.DISABLED)
        root.mainloop()

    # ── Separator lines ───────────────────────────────────────────────────────
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.92, 0.92],
                              transform=fig.transFigure,
                              color=_BORDER, linewidth=0.8))
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.105, 0.105],
                              transform=fig.transFigure,
                              color=_BORDER, linewidth=0.8))

    # ── Initial draw ──────────────────────────────────────────────────────────
    _on_run()
    plt.show()
