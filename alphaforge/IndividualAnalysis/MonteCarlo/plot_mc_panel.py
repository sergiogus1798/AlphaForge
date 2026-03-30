"""
plot_mc_panel.py — Matplotlib Monte Carlo dashboard for AlphaForge.

Two views (toggled with KDE / Evolution buttons):

  KDE        — 2×2 grid; each subplot = one metric; curves = per-period KDE
               overlaid + CI bands + mean/median vlines + real-value line.

  Evolution  — 2×2 grid; each subplot = one rolling period; equity fan chart
               (faint sim paths + CI fill + real equity in amber).

Controls (top):
  ◄ strategy name (N trades) ►
  [Reshuffle][Bootstrap][Time Block][Trade Block][Best Trade Removal]  [KDE][Evolution]  [► RUN]
  Periods: [1][2][3][4][5][6]     CI Band (Evolution): [80%][90%][95%][98%][99%]
  Sims:[__]  Time Block:[__] block period  Trade Block:[__] trades  Best Removal:[__] % top trades
"""

from __future__ import annotations

import math
import threading

import matplotlib
import matplotlib.ticker
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from alphaforge.IndividualAnalysis.MonteCarlo.monte_carlo import (
    run_bootstrap, run_reshuffle, run_time_block_bootstrap,
    run_trade_block_bootstrap, run_best_trade_removal,
    run_rolling, get_equity_curves,
)

# ── Palette ───────────────────────────────────────────────────────────────────
_BG     = "#0d1117"
_AX     = "#16213e"
_BORDER = "#0f3460"
_ACCENT = "#00d4ff"
_GREEN  = "#3fb950"
_RED    = "#f85149"
_AMBER  = "#d29922"
_YELLOW = "#e6c84a"
_TEXT   = "#c9d1d9"
_DIM    = "#8b949e"
_GRID   = "#1f2937"

PERIOD_COLORS = ["#00d4ff", "#bc8cff", "#3fb950", "#d29922", "#f85149", "#e6c84a"]
CI_90  = "#3fb950"
CI_95  = "#d29922"
CI_99  = "#f85149"

_BTN_KW  = dict(color=_BORDER, hovercolor="#1a2a4a")
_BTN_TXT = dict(color=_TEXT, fontsize=8.5, fontfamily="monospace")

_METRICS = [
    ("return_pct",   "Return (%)"),
    ("max_drawdown", "Max Drawdown ($)"),
    ("ret_dd",       "Return / Drawdown"),
    ("final_equity", "Final Equity ($)"),
]

_TESTS = [
    ("reshuffle",          "Trade Reshuffle"),
    ("bootstrap",          "Bootstrap"),
    ("time_block",         "Time Block"),
    ("trade_block",        "Trade Block"),
    ("best_trade_removal", "Best Trade Removal"),
]
_TEST_DICT = {k: v for k, v in _TESTS}
_NO_SKIP   = {"reshuffle"}

_MC_FN = {
    "reshuffle":          run_reshuffle,
    "bootstrap":          run_bootstrap,
    "time_block":         run_time_block_bootstrap,
    "trade_block":        run_trade_block_bootstrap,
    "best_trade_removal": run_best_trade_removal,
}

# ── Axis helpers ──────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(_AX)
    ax.tick_params(colors=_DIM, labelsize=8)
    for s in ax.spines.values():
        s.set_color(_BORDER)
    ax.xaxis.label.set_color(_DIM)
    ax.yaxis.label.set_color(_DIM)
    ax.title.set_color(_ACCENT)
    ax.margins(x=0.01)


def _grid(ax):
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


# ── Chart builders ────────────────────────────────────────────────────────────

def _real_metric(chunk_df, col, initial_capital):
    """Compute the real (non-simulated) value of a metric for one period chunk."""
    pnl = chunk_df["Profit/Loss"].dropna().values
    if len(pnl) == 0:
        return None
    eq       = np.concatenate([[initial_capital], initial_capital + np.cumsum(pnl)])
    peak     = np.maximum.accumulate(eq)
    max_dd   = float((peak - eq).max())
    final_eq = float(eq[-1])
    ret_pct  = (final_eq - initial_capital) / initial_capital * 100
    ret_dd   = (ret_pct / (max_dd / initial_capital * 100)) if max_dd > 0 else float("nan")
    return {
        "return_pct":   ret_pct,
        "max_drawdown": max_dd,
        "ret_dd":       ret_dd,
        "final_equity": final_eq,
    }


def _draw_kde_subplot(ax, period_data, col, title, initial_capital):
    """One KDE subplot: per-period overlaid KDE + CI shading + real vline."""
    _style(ax)
    _grid(ax)
    ax.set_title(title, color=_ACCENT, fontsize=9, pad=4)

    # collect all simulation values across periods for CI reference
    all_vals = []
    for pd_item in period_data:
        sim_df = pd_item.get("sim_df")
        if sim_df is not None and col in sim_df.columns:
            v = sim_df[col].dropna().values
            all_vals.append(v[np.isfinite(v)])

    if not all_vals:
        ax.text(0.5, 0.5, "No data\nPress ▶ RUN", ha="center", va="center",
                transform=ax.transAxes, color=_DIM, fontsize=9, fontfamily="monospace")
        return

    combined = np.concatenate(all_vals)
    if len(combined) < 2:
        return

    x_min, x_max = float(combined.min()), float(combined.max())
    if x_min >= x_max:
        x_max = x_min + 1.0
    x_range = np.linspace(x_min, x_max, 500)

    # CI shading from combined distribution
    for lo_p, hi_p, color, alpha in [
        (0.5,  99.5, CI_99, 0.06),
        (2.5,  97.5, CI_95, 0.10),
        (5.0,  95.0, CI_90, 0.16),
    ]:
        lo = np.percentile(combined, lo_p)
        hi = np.percentile(combined, hi_p)
        ax.axvspan(lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)
        for xv in [lo, hi]:
            ax.axvline(xv, color=color, linewidth=0.8, linestyle="--", alpha=0.55)

    # Per-period KDE
    for i, pd_item in enumerate(period_data):
        sim_df = pd_item.get("sim_df")
        chunk  = pd_item.get("chunk")
        color  = pd_item["color"]

        if sim_df is None or col not in sim_df.columns:
            continue
        vals = sim_df[col].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue

        is_const = float(np.std(vals)) < 1e-9
        if is_const:
            ax.axvline(vals[0], color=color, linewidth=2.0,
                       label=pd_item["short_label"])
            continue

        kde   = scipy_stats.gaussian_kde(vals)
        kde_y = kde(x_range)

        ax.fill_between(x_range, kde_y, alpha=0.10, color=color, zorder=2)
        ax.plot(x_range, kde_y, color=color, linewidth=1.8,
                label=pd_item["short_label"], zorder=3)

        # mean + median per period (dashed/solid, same period color, thinner)
        ax.axvline(float(vals.mean()),      color=color, linewidth=0.9,
                   linestyle="--", alpha=0.75, zorder=4)
        ax.axvline(float(np.median(vals)), color=color, linewidth=0.9,
                   linestyle="-",  alpha=0.75, zorder=4)

        # real value for this period (amber, thick)
        if chunk is not None:
            real = _real_metric(chunk, col, initial_capital)
            if real and col in real and np.isfinite(real[col]):
                ax.axvline(real[col], color=_AMBER, linewidth=1.6,
                           linestyle=":", alpha=0.85, zorder=5)

    ax.axvline(0, color="white", linewidth=0.8, alpha=0.25)
    ax.legend(fontsize=7, labelcolor=_DIM, framealpha=0.0,
              loc="upper right", handlelength=1.2)
    ax.set_ylabel("Density", fontsize=8)


def _draw_evolution_subplot(ax, pd_item, ci_pct, period_idx):
    """One equity-fan subplot for a rolling period."""
    chunk  = pd_item["chunk"]
    curves = pd_item["eq_curves"]
    color  = pd_item["color"]
    label  = pd_item["label"]

    _style(ax)
    _grid(ax)
    ax.set_title(label, color=_TEXT, fontsize=8.5, pad=4)

    real_pnl = chunk.sort_values("Close time")["Profit/Loss"].dropna().values
    real_eq  = np.concatenate([[0], np.cumsum(real_pnl)])
    xs_real  = list(range(len(real_eq)))

    if curves:
        L = len(real_pnl)
        padded = []
        for c in curves:
            arr = np.array([0] + list(c), dtype=float)
            if len(arr) >= L + 1:
                padded.append(arr[:L + 1])
            else:
                padded.append(np.pad(arr, (0, L + 1 - len(arr)), "edge"))
        sim_mat = np.array(padded)
        xs_sim  = list(range(sim_mat.shape[1]))

        step = max(1, len(sim_mat) // 80)
        for row in sim_mat[::step]:
            ax.plot(xs_sim, row, color=color, linewidth=0.4, alpha=0.13)

        lo_p  = (100 - ci_pct) / 2
        hi_p  = 100 - lo_p
        p_lo  = np.percentile(sim_mat, lo_p, axis=0)
        p_hi  = np.percentile(sim_mat, hi_p, axis=0)
        ax.fill_between(xs_sim, p_lo, p_hi, color=color, alpha=0.22)

    ax.plot(xs_real, real_eq, color=_AMBER, linewidth=2.0, zorder=5)
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.3, linestyle="-")
    ax.set_xlabel("Trades", fontsize=8)
    ax.set_ylabel("PnL ($)", fontsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))


def _draw_kde_legend(fig, bottom=0.025, height=0.050):
    """Bottom legend bar for KDE mode."""
    ax = fig.add_axes([0.02, bottom, 0.96, height])
    ax.set_facecolor(_AX)
    for sp in ax.spines.values():
        sp.set_color(_BORDER)
    ax.set_axis_off()

    handles = [
        mpatches.Patch(color=CI_90, alpha=0.55, label="90% CI"),
        mpatches.Patch(color=CI_95, alpha=0.65, label="95% CI"),
        mpatches.Patch(color=CI_99, alpha=0.65, label="99% CI"),
        mlines.Line2D([], [], color=_YELLOW, linestyle="--",
                      linewidth=1.4, label="Mean"),
        mlines.Line2D([], [], color=_GREEN,  linestyle="-",
                      linewidth=1.4, label="Median"),
        mlines.Line2D([], [], color=_AMBER,  linestyle=":",
                      linewidth=1.8, label="Real (Evolution)"),
    ]
    # add period colours
    for i, c in enumerate(PERIOD_COLORS[:6]):
        handles.append(mlines.Line2D([], [], color=c, linewidth=1.6,
                                     label=f"P{i+1}"))

    ax.legend(handles=handles, loc="center", ncol=len(handles),
              framealpha=0.0, fontsize=8, labelcolor=_TEXT,
              handlelength=1.6, columnspacing=1.0)
    return ax


# ── Main panel ────────────────────────────────────────────────────────────────

def plot_mc_panel(
    all_strategies: dict,
    initial_key:    str,
    initial_capital: float = 10_000.0,
) -> None:

    all_keys = sorted(all_strategies.keys())

    def _df_for(key):
        return (all_strategies[key]
                .sort_values("Close time")
                .reset_index(drop=True))

    state = {
        "key_idx":     all_keys.index(initial_key) if initial_key in all_keys else 0,
        "test":        "reshuffle",
        "n_periods":   4,
        "nsims":       1_000,
        "time_block":  "6ME",
        "trade_block": 10,
        "best_removal": 5.0,
        "view":        "kde",    # "kde" | "evolution"
        "ci_band":     95,
        "result":      None,
        "running":     False,
    }

    fig = plt.figure(figsize=(18, 10), facecolor=_BG)
    fig.canvas.manager.set_window_title("AlphaForge — Monte Carlo")

    _plot_axes: list = []
    _ctrl_axes: list = []
    _btns:      list = []   # keep references — prevents garbage collection

    # ── Row positions ──────────────────────────────────────────────────────────
    NAV_Y   = 0.940   # navigation (◄ name ►)
    NAV_H   = 0.046
    TEST_Y  = 0.885
    TEST_H  = 0.046
    PER_Y   = 0.830
    PER_H   = 0.046
    PAR_Y   = 0.775
    PAR_H   = 0.046

    def _make_btn(x, y, w, h, label, cb, color=None):
        ax = fig.add_axes([x, y, w, h])
        ax.set_facecolor(color or _BORDER)
        for sp in ax.spines.values():
            sp.set_color(_DIM)
        btn = Button(ax, label, color=color or _BORDER, hovercolor="#1e3055")
        btn.label.set(**_BTN_TXT)
        btn.on_clicked(cb)
        _ctrl_axes.append(ax)
        _btns.append(btn)   # keep reference alive
        return btn, ax

    # ── Navigation row ─────────────────────────────────────────────────────────
    btn_prev, _ = _make_btn(0.005, NAV_Y, 0.030, NAV_H, "◄",
                             lambda _: _nav(-1))
    btn_next, _ = _make_btn(0.965, NAV_Y, 0.030, NAV_H, "►",
                             lambda _: _nav(+1))

    _btns.extend([btn_prev, btn_next])
    _title_txt = [fig.text(0.5, NAV_Y + NAV_H * 0.5, "",
                           ha="center", va="center", color=_TEXT,
                           fontsize=12, fontfamily="monospace",
                           fontweight="bold")]

    def _update_title():
        key    = all_keys[state["key_idx"]]
        df_    = _df_for(key)
        n      = len(df_["Profit/Loss"].dropna())
        _title_txt[0].set_text(f"{key}   ({n} trades)")
        fig.canvas.draw_idle()

    def _nav(direction):
        state["key_idx"] = (state["key_idx"] + direction) % len(all_keys)
        state["result"]  = None
        _update_title()
        _redraw()

    # ── Test buttons ───────────────────────────────────────────────────────────
    _test_btns: dict = {}   # key -> (btn, ax)
    tx = 0.005
    tw = (0.72 - 0.005) / len(_TESTS) - 0.004
    for key, lbl in _TESTS:
        btn_t, ax_t = _make_btn(tx, TEST_Y, tw, TEST_H, lbl,
                                lambda _, k=key: _set_test(k))
        _test_btns[key] = (btn_t, ax_t)
        tx += tw + 0.004

    # View buttons + RUN
    _view_btns: dict = {}   # key -> (btn, ax)
    btn_kde, ax_kde = _make_btn(0.728, TEST_Y, 0.080, TEST_H, "KDE",
                                lambda _: _set_view("kde"))
    btn_evo, ax_evo = _make_btn(0.812, TEST_Y, 0.080, TEST_H, "Evolution",
                                lambda _: _set_view("evolution"))
    _view_btns["kde"]       = (btn_kde, ax_kde)
    _view_btns["evolution"] = (btn_evo, ax_evo)

    btn_run, ax_run = _make_btn(0.900, TEST_Y, 0.095, TEST_H,
                                 "▶  RUN", lambda _: _on_run(),
                                 color="#1a4a1a")
    ax_run.set_facecolor("#1a4a1a")
    btn_run.color = "#1a4a1a"
    btn_run.label.set(color=_GREEN, fontsize=9, fontfamily="monospace",
                      fontweight="bold")

    # ── Periods row ────────────────────────────────────────────────────────────
    fig.text(0.010, PER_Y + PER_H * 0.55, "Periods:", color=_DIM,
             fontsize=8.5, fontfamily="monospace", va="center")
    _per_btns: dict = {}    # n -> (btn, ax)
    for i, n in enumerate([1, 2, 3, 4, 5, 6]):
        px = 0.075 + i * 0.045
        btn_p, ax_p = _make_btn(px, PER_Y, 0.038, PER_H, str(n),
                                lambda _, nn=n: _set_periods(nn))
        _per_btns[n] = (btn_p, ax_p)

    # CI Band buttons (used in Evolution view)
    fig.text(0.365, PER_Y + PER_H * 0.55, "CI Band:", color=_DIM,
             fontsize=8.5, fontfamily="monospace", va="center")
    _ci_btns: dict = {}     # ci -> (btn, ax)
    for i, ci in enumerate([80, 90, 95, 98, 99]):
        cx = 0.428 + i * 0.058
        btn_ci, ax_ci = _make_btn(cx, PER_Y, 0.050, PER_H, f"{ci}%",
                                  lambda _, c=ci: _set_ci(c))
        _ci_btns[ci] = (btn_ci, ax_ci)

    # Status text
    _status_txt = [fig.text(
        0.760, PER_Y + PER_H * 0.5, "",
        color=_AMBER, fontsize=8, fontfamily="monospace", va="center")]

    # ── Parameters row ─────────────────────────────────────────────────────────
    def _make_tb(x, w, initial, label_txt, unit_txt):
        """TextBox + surrounding labels."""
        fig.text(x, PAR_Y + PAR_H * 0.58, label_txt, color=_DIM,
                 fontsize=8, fontfamily="monospace", va="center")
        ax_tb = fig.add_axes([x + 0.048, PAR_Y, w, PAR_H])
        ax_tb.set_facecolor(_BORDER)
        for sp in ax_tb.spines.values():
            sp.set_color(_DIM)
        tb = TextBox(ax_tb, "", initial=str(initial),
                     color=_BORDER, hovercolor="#1e3055", label_pad=0.02)
        tb.text_disp.set_color(_TEXT)
        tb.text_disp.set_fontfamily("monospace")
        tb.text_disp.set_fontsize(8.5)
        _ctrl_axes.append(ax_tb)
        _btns.append(tb)    # keep TextBox reference alive
        if unit_txt:
            fig.text(x + 0.048 + w + 0.006, PAR_Y + PAR_H * 0.58,
                     unit_txt, color=_DIM, fontsize=8,
                     fontfamily="monospace", va="center")
        return tb

    _tb_sims    = _make_tb(0.005, 0.055, state["nsims"],       "Sims:",          "")
    _tb_tblock  = _make_tb(0.140, 0.045, state["time_block"],  "Time Block:",    "block period")
    _tb_trblock = _make_tb(0.320, 0.040, state["trade_block"], "Trade Block:",   "trades")
    _tb_removal = _make_tb(0.490, 0.040, state["best_removal"],"Best Removal:",  "% top trades")

    # ── Separator lines ────────────────────────────────────────────────────────
    for y_sep in [0.769, 0.082]:
        fig.add_artist(plt.Line2D([0.005, 0.995], [y_sep, y_sep],
                                  transform=fig.transFigure,
                                  color=_BORDER, linewidth=0.8))

    # ── Highlight helpers ──────────────────────────────────────────────────────
    def _set_btn_color(btn, ax, color):
        ax.set_facecolor(color)
        btn.color = color

    def _refresh_highlights():
        for k, (btn, ax) in _test_btns.items():
            _set_btn_color(btn, ax, _ACCENT if k == state["test"] else _BORDER)
        for n, (btn, ax) in _per_btns.items():
            _set_btn_color(btn, ax, _YELLOW if n == state["n_periods"] else _BORDER)
        for v, (btn, ax) in _view_btns.items():
            _set_btn_color(btn, ax, _ACCENT if v == state["view"] else _BORDER)
        for c, (btn, ax) in _ci_btns.items():
            _set_btn_color(btn, ax, _ACCENT if c == state["ci_band"] else _BORDER)

    def _set_test(k):
        state["test"] = k
        _refresh_highlights()
        fig.canvas.draw_idle()

    def _set_periods(n):
        state["n_periods"] = n
        _refresh_highlights()
        fig.canvas.draw_idle()

    def _set_view(v):
        state["view"] = v
        _refresh_highlights()
        _redraw()

    def _set_ci(c):
        state["ci_band"] = c
        _refresh_highlights()
        _redraw()

    # ── Run simulation ─────────────────────────────────────────────────────────
    def _on_run(_e=None):
        if state["running"]:
            return
        # read textboxes
        try:
            state["nsims"]        = max(100, int(_tb_sims.text.strip()))
        except ValueError:
            pass
        try:
            state["time_block"]   = _tb_tblock.text.strip() or "6ME"
        except Exception:
            pass
        try:
            state["trade_block"]  = max(2, int(_tb_trblock.text.strip()))
        except ValueError:
            pass
        try:
            state["best_removal"] = max(0.01, float(_tb_removal.text.strip()))
        except ValueError:
            pass

        state["running"] = True
        ax_run.set_facecolor(_AMBER)
        _status_txt[0].set_text("Running…")
        fig.canvas.draw_idle()

        def _worker():
            test      = state["test"]
            nsims     = state["nsims"]
            n_periods = state["n_periods"]
            skip      = 0.0 if test in _NO_SKIP else 0.05
            key       = all_keys[state["key_idx"]]
            df        = _df_for(key)

            # extra kwargs per test
            extra = {}
            if test == "time_block":
                extra["time_block_size"] = state["time_block"]
            elif test == "trade_block":
                extra["block_size"] = state["trade_block"]
            elif test == "best_trade_removal":
                extra["removal_pct"] = state["best_removal"] / 100.0

            # split into periods (equal calendar-time chunks)
            df_s  = df.sort_values("Close time").reset_index(drop=True)
            t_min = df_s["Close time"].min()
            t_max = df_s["Close time"].max()
            edges = pd.date_range(t_min, t_max, periods=n_periods + 1)

            period_data = []
            for i in range(n_periods):
                lo, hi = edges[i], edges[i + 1]
                if i < n_periods - 1:
                    mask = (df_s["Close time"] >= lo) & (df_s["Close time"] < hi)
                else:
                    mask = (df_s["Close time"] >= lo) & (df_s["Close time"] <= hi)
                chunk = df_s[mask].reset_index(drop=True)

                t0 = chunk["Close time"].min().strftime("%Y-%m") if len(chunk) else "?"
                t1 = chunk["Close time"].max().strftime("%Y-%m") if len(chunk) else "?"
                label = f"P{i+1}: {t0} → {t1}  ({len(chunk)} trades)"

                # per-period simulation (for KDE view)
                if len(chunk) >= 5:
                    fn     = _MC_FN[test]
                    sim_df = fn(chunk, n_simulations=nsims,
                                skip_trade_probability=skip,
                                initial_capital=initial_capital,
                                **extra)
                    # equity curves for evolution view
                    eq_curves = get_equity_curves(
                        chunk, test=test,
                        n_curves=min(100, nsims),
                        skip_trade_probability=0.0, seed=42)
                else:
                    sim_df    = None
                    eq_curves = []

                period_data.append({
                    "label":       label,
                    "short_label": f"P{i+1}",
                    "chunk":       chunk,
                    "sim_df":      sim_df,
                    "eq_curves":   eq_curves,
                    "color":       PERIOD_COLORS[i % len(PERIOD_COLORS)],
                })

            state["result"] = {
                "period_data": period_data,
                "test_lbl":    _TEST_DICT.get(test, test),
                "nsims":       nsims,
                "n_periods":   n_periods,
                "key":         key,
            }
            state["running"] = False
            ax_run.set_facecolor("#1a4a1a")
            _status_txt[0].set_text(
                f"Done  ({nsims:,} sims × {n_periods} periods)")
            _redraw()
            fig.canvas.draw_idle()

        threading.Thread(target=_worker, daemon=True).start()

    # ── Redraw ────────────────────────────────────────────────────────────────
    def _redraw():
        for ax in list(_plot_axes):
            try:
                ax.remove()
            except Exception:
                pass
        _plot_axes.clear()

        _refresh_highlights()
        result = state["result"]
        view   = state["view"]

        PLOT_TOP = 0.760
        PLOT_BOT = 0.095

        # ── no results yet ─────────────────────────────────────────────────────
        if result is None:
            ax_ph = fig.add_axes([0.05, PLOT_BOT, 0.90, PLOT_TOP - PLOT_BOT])
            ax_ph.set_facecolor(_AX)
            for sp in ax_ph.spines.values():
                sp.set_color(_BORDER)
            ax_ph.set_axis_off()
            ax_ph.text(0.5, 0.54, "No results yet",
                       ha="center", va="center", transform=ax_ph.transAxes,
                       color=_ACCENT, fontsize=20, fontfamily="monospace",
                       fontweight="bold")
            ax_ph.text(0.5, 0.42, "Configure options above and press  ▶ RUN",
                       ha="center", va="center", transform=ax_ph.transAxes,
                       color=_DIM, fontsize=11, fontfamily="monospace")
            _plot_axes.append(ax_ph)
            fig.canvas.draw_idle()
            return

        period_data = result["period_data"]
        n_periods   = result["n_periods"]
        test_lbl    = result["test_lbl"]

        # ── KDE view — always 2×2 (4 metrics regardless of period count) ───────
        if view == "kde":
            gs = gridspec.GridSpec(
                2, 2, figure=fig,
                left=0.06, right=0.99,
                top=PLOT_TOP, bottom=PLOT_BOT,
                hspace=0.42, wspace=0.22)
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for (r, c), (col, label) in zip(positions, _METRICS):
                ax = fig.add_subplot(gs[r, c])
                _plot_axes.append(ax)
                _draw_kde_subplot(ax, period_data, col, label, initial_capital)

            # bottom legend
            ax_leg = _draw_kde_legend(fig, bottom=0.022, height=0.052)
            _plot_axes.append(ax_leg)

        # ── Evolution view — grid sized to n_periods ───────────────────────────
        elif view == "evolution":
            ci_pct = state["ci_band"]

            if n_periods == 1:
                ncols, nrows = 1, 1
            elif n_periods == 2:
                ncols, nrows = 2, 1
            elif n_periods <= 4:
                ncols, nrows = 2, 2
            elif n_periods <= 6:
                ncols, nrows = 3, 2
            else:
                ncols, nrows = 3, 3

            gs = gridspec.GridSpec(
                nrows, ncols, figure=fig,
                left=0.06, right=0.99,
                top=PLOT_TOP, bottom=PLOT_BOT,
                hspace=0.42, wspace=0.22)

            # title above plots
            _plot_axes.append(fig.text(
                0.5, PLOT_TOP + 0.008,
                f"Equity Curves  |  {test_lbl}  |  "
                f"{result['key']}  |  {ci_pct}% CI band",
                ha="center", color=_TEXT, fontsize=9.5,
                fontfamily="monospace"))

            n_show = min(n_periods, len(period_data))
            positions_used = [(r, c) for r in range(nrows) for c in range(ncols)]

            for idx, (r, c) in enumerate(positions_used):
                ax = fig.add_subplot(gs[r, c])
                _plot_axes.append(ax)
                if idx < n_show:
                    _draw_evolution_subplot(ax, period_data[idx], ci_pct, idx)
                else:
                    ax.set_facecolor(_AX)
                    ax.set_axis_off()

        fig.canvas.draw_idle()

    # ── Initial setup ──────────────────────────────────────────────────────────
    _update_title()
    _refresh_highlights()
    _redraw()
    plt.show()
