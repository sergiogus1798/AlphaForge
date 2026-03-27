"""
dashboard_mc.py — Monte Carlo simulation dashboard for portfolio combinations.

One matplotlib window:
  Left area  : equity-curve fan chart with 90 / 95 / 98 % CI shading.
               Single mode  — full portfolio history.
               Rolling mode — N equal-time windows, one subplot each.
  Right panel: controls
                 · Combination navigator  (< Prev / Next >)
                 · Weighting-method ticks (CheckButtons, one per method)
                 · MC-method radio        (6 methods)
                 · N-runs text box        (default 10 000)
                 · Mode radio             (Single / Rolling)
                 · Windows slider         (1 – 8, rolling mode only)
                 · RUN button

MC methods
  trade_wr   Trade Bootstrap     (with replacement)
  trade_wor  Trade Reshuffle     (without replacement)
  daily_wr   Daily Bootstrap     (with replacement)
  daily_wor  Daily Reshuffle     (without replacement)
  block_wr   Block Bootstrap     (with replacement)
  block_wor  Block Reshuffle     (without replacement)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox, Slider

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.combo_result import (
    CombinationResult, METHODS, METHOD_COLORS, METHOD_LABELS,
)


# ── Theme ───────────────────────────────────────────────────────────────────────

BG     = "#0f1117"
AX_BG  = "#1a1a2e"
BORDER = "#2a2a4a"
CYAN   = "#00d4ff"
AMBER  = "#f0a500"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
TEXT   = "#e0e0e0"
DIM    = "#888888"
PANEL  = "#0c0f1e"


# ── MC method registry ──────────────────────────────────────────────────────────

MC_METHODS: dict[str, str] = {
    "trade_wr":  "Trade Bootstrap  (w/ repl.)",
    "trade_wor": "Trade Reshuffle  (w/o repl.)",
    "daily_wr":  "Daily Bootstrap  (w/ repl.)",
    "daily_wor": "Daily Reshuffle  (w/o repl.)",
    "block_wr":  "Block Bootstrap  (w/ repl.)",
    "block_wor": "Block Reshuffle  (w/o repl.)",
}
MC_KEYS = list(MC_METHODS.keys())

BLOCK_SIZE = 10   # trades per block — not exposed in UI


# ── Simulation core ─────────────────────────────────────────────────────────────

def _simulate_curves(
    portfolio_df: pd.DataFrame,
    mc_method: str,
    n_sims: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Run n_sims Monte Carlo simulations on a portfolio DataFrame.

    Returns:
        curves  : (n_sims, T) cumulative P&L paths, each starting at 0
        actual  : (T,) real historical cumulative P&L
        x_label : "Trade #" or "Day #"
    """
    rng  = np.random.default_rng(seed)
    df_s = portfolio_df.sort_values("Close time")

    if mc_method in ("trade_wr", "trade_wor"):
        pnl    = df_s["Profit/Loss"].dropna().values
        T      = len(pnl)
        repl   = mc_method == "trade_wr"
        curves = np.empty((n_sims, T), dtype=np.float64)
        for i in range(n_sims):
            curves[i] = np.cumsum(rng.choice(pnl, size=T, replace=repl))
        return curves, np.cumsum(pnl), "Trade #"

    if mc_method in ("daily_wr", "daily_wor"):
        daily  = df_s.groupby(df_s["Close time"].dt.date)["Profit/Loss"].sum().values
        T      = len(daily)
        repl   = mc_method == "daily_wr"
        curves = np.empty((n_sims, T), dtype=np.float64)
        for i in range(n_sims):
            curves[i] = np.cumsum(rng.choice(daily, size=T, replace=repl))
        return curves, np.cumsum(daily), "Day #"

    if mc_method in ("block_wr", "block_wor"):
        pnl    = df_s["Profit/Loss"].dropna().values
        T      = len(pnl)
        blocks = [pnl[i:i + BLOCK_SIZE] for i in range(0, T, BLOCK_SIZE)]
        n_blk  = len(blocks)
        repl   = mc_method == "block_wr"
        curves = np.empty((n_sims, T), dtype=np.float64)
        for i in range(n_sims):
            idx = rng.integers(0, n_blk, size=n_blk) if repl else rng.permutation(n_blk)
            seq = np.concatenate([blocks[j] for j in idx])[:T]
            curves[i] = np.cumsum(seq)
        return curves, np.cumsum(pnl), "Trade #"

    raise ValueError(f"Unknown mc_method: {mc_method!r}")


def _ci_bands(curves: np.ndarray) -> dict:
    """Extract CI percentile bands from a (n_sims, T) array."""
    p = np.percentile(curves, [1.0, 2.5, 5.0, 50.0, 95.0, 97.5, 99.0], axis=0)
    return {
        "p1":   p[0], "p99":   p[6],   # 98 % CI
        "p2_5": p[1], "p97_5": p[5],   # 95 % CI
        "p5":   p[2], "p95":   p[4],   # 90 % CI
        "p50":  p[3],                   # median
    }


# ── Axis helpers ────────────────────────────────────────────────────────────────

def _style(ax) -> None:
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color=BORDER, lw=0.4)


def _draw_fan(
    ax,
    x: np.ndarray,
    bands: dict,
    actual: np.ndarray,
    color: str,
    label: str,
    show_legend: bool = True,
) -> None:
    """Draw CI shading (90/95/98%), median, and actual historical curve."""
    ax.fill_between(x, bands["p1"],   bands["p99"],   alpha=0.10, color=color)
    ax.fill_between(x, bands["p2_5"], bands["p97_5"], alpha=0.17, color=color)
    ax.fill_between(x, bands["p5"],   bands["p95"],   alpha=0.28, color=color,
                    label=f"{label}  90/95/98% CI")
    ax.plot(x, bands["p50"], color=color, lw=1.8, alpha=0.9,
            label=f"{label}  Median")
    ax.plot(x, actual[:len(x)], color=color, lw=2.5, ls="--",
            label=f"{label}  Actual")
    ax.axhline(0, color=BORDER, lw=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    if show_legend:
        ax.legend(fontsize=7, facecolor=AX_BG, labelcolor=TEXT,
                  loc="upper left", framealpha=0.85, ncol=1)


# ── Plot area helpers ───────────────────────────────────────────────────────────

_PLOT_L = 0.03
_PLOT_B = 0.07
_PLOT_W = 0.70
_PLOT_H = 0.86


def _clear_plot_axes(fig, state: dict) -> None:
    for ax in state.get("_plot_axes", []):
        try:
            fig.delaxes(ax)
        except Exception:
            pass
    state["_plot_axes"] = []


def _split_df(portfolio_df: pd.DataFrame, n_windows: int) -> list[pd.DataFrame]:
    """Split portfolio into n_windows equal calendar-time chunks."""
    df_s  = portfolio_df.sort_values("Close time")
    t_min = df_s["Close time"].min()
    t_max = df_s["Close time"].max()
    edges = pd.date_range(t_min, t_max, periods=n_windows + 1)
    chunks = []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        mask = (df_s["Close time"] >= lo) & (
            df_s["Close time"] < hi if i < n_windows - 1
            else df_s["Close time"] <= hi
        )
        chunk = df_s[mask].reset_index(drop=True)
        chunks.append((chunk, edges[i], edges[i + 1]))
    return chunks


# ── Main draw/run orchestrator ──────────────────────────────────────────────────

def _run_and_draw(
    fig,
    combinations: list[CombinationResult],
    state: dict,
    config: PortfolioConfig,
) -> None:
    _clear_plot_axes(fig, state)

    cr        = combinations[state["idx"]]
    active_w  = [m for m in METHODS if m in state["active_weights"]]
    mc_method = state["mc_method"]
    n_sims    = state["n_sims"]
    mode      = state["mode"]
    n_windows = state["n_windows"]

    # Show computing indicator
    fig.suptitle("  Computing…", color=DIM, fontsize=10, y=0.99)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    if not active_w:
        fig.canvas.draw_idle()
        return

    # ── Single mode ────────────────────────────────────────────────────────────
    if mode == "single":
        ax = fig.add_axes([_PLOT_L, _PLOT_B, _PLOT_W, _PLOT_H])
        _style(ax)
        state["_plot_axes"] = [ax]

        x_label_final = "Trade #"
        for wm in active_w:
            vp = cr.portfolios.get(wm)
            if vp is None:
                continue
            curves, actual, x_label = _simulate_curves(
                vp.portfolio_df, mc_method, n_sims,
            )
            x_label_final = x_label
            x     = np.arange(len(actual))
            bands = _ci_bands(curves)
            _draw_fan(ax, x, bands, actual,
                      METHOD_COLORS[wm], METHOD_LABELS[wm], show_legend=True)

        ax.set_xlabel(x_label_final, color=TEXT, fontsize=9)
        ax.set_ylabel("Cumul. P&L ($)", color=TEXT, fontsize=9)
        ax.set_ylim(bottom=None)
        ax.margins(x=0.01)

    # ── Rolling mode ───────────────────────────────────────────────────────────
    else:
        # First valid weight portfolio to derive time edges
        first_vp = next(
            (cr.portfolios.get(wm) for wm in active_w if cr.portfolios.get(wm)),
            None,
        )
        if first_vp is None:
            fig.canvas.draw_idle()
            return

        n_cols = min(n_windows, 4)
        n_rows = math.ceil(n_windows / n_cols)
        w_each = _PLOT_W / n_cols
        h_each = _PLOT_H / n_rows

        # Build window time edges from the first valid portfolio
        chunks_ref = _split_df(first_vp.portfolio_df, n_windows)

        axes: list = []
        for win_i in range(n_windows):
            col = win_i % n_cols
            row = win_i // n_cols
            left = _PLOT_L + col * w_each
            bot  = _PLOT_B + (n_rows - 1 - row) * h_each
            ax   = fig.add_axes([left, bot, w_each * 0.96, h_each * 0.88])
            _style(ax)
            axes.append(ax)

        state["_plot_axes"] = axes

        x_label_final = "Trade #"
        for win_i, ax in enumerate(axes):
            _, t0, t1 = chunks_ref[win_i]
            ax.set_title(
                f"W{win_i+1}  {t0.strftime('%Y-%m')} → {t1.strftime('%Y-%m')}",
                color=TEXT, fontsize=8,
            )
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
            ax.tick_params(labelsize=7)

            for wm in active_w:
                vp = cr.portfolios.get(wm)
                if vp is None:
                    continue
                chunks_wm = _split_df(vp.portfolio_df, n_windows)
                chunk_df, _, _ = chunks_wm[win_i]
                if len(chunk_df) < 5:
                    continue
                curves, actual, x_label = _simulate_curves(
                    chunk_df, mc_method, n_sims,
                )
                x_label_final = x_label
                x     = np.arange(len(actual))
                bands = _ci_bands(curves)
                _draw_fan(ax, x, bands, actual,
                          METHOD_COLORS[wm], METHOD_LABELS[wm],
                          show_legend=(win_i == 0))

        # Shared x-label on a dedicated axes
        ax_xl = fig.add_axes([_PLOT_L, _PLOT_B - 0.04, _PLOT_W, 0.03])
        ax_xl.axis("off")
        ax_xl.text(0.5, 0.5, x_label_final, color=TEXT, fontsize=9,
                   ha="center", va="center", transform=ax_xl.transAxes)
        axes.append(ax_xl)

    fig.suptitle(
        f"Monte Carlo  ·  Combo #{state['idx']+1}/{len(combinations)}  ·  "
        f"{MC_METHODS[mc_method]}  ·  n={n_sims:,}  ·  "
        f"{'Single' if mode == 'single' else f'Rolling ×{n_windows}'}",
        color=CYAN, fontsize=10, y=0.995,
    )
    fig.canvas.draw_idle()


# ── Entry point ─────────────────────────────────────────────────────────────────

def run_mc_dashboard(
    combinations: list[CombinationResult],
    config: PortfolioConfig,
    _show: bool = True,
) -> None:
    """
    Open the Monte Carlo simulation dashboard.

    Args:
        combinations : ranked list from the pipeline.
        config       : PortfolioConfig (used for initial_capital reference).
    """
    if not combinations:
        print("  No combinations to display.")
        return

    n = len(combinations)

    state: dict = {
        "idx":            0,
        "mc_method":      "trade_wr",
        "active_weights": set(METHODS),
        "n_sims":         10_000,
        "mode":           "single",
        "n_windows":      4,
        "_plot_axes":     [],
    }

    fig = plt.figure(figsize=(24, 11), facecolor=BG)

    # ── Sidebar background ──────────────────────────────────────────────────────
    from matplotlib.patches import Rectangle as _Rect
    fig.add_artist(_Rect(
        (0.745, 0.01), 0.252, 0.98,
        transform=fig.transFigure,
        facecolor=PANEL, edgecolor=BORDER, linewidth=1.0, zorder=0,
    ))

    SBL = 0.758    # sidebar left edge (figure coords)
    SBW = 0.225    # sidebar width

    def _lbl(y: float, text: str) -> None:
        """Draw a section label inside the sidebar."""
        ax_ = fig.add_axes([SBL, y, SBW, 0.022])
        ax_.set_facecolor(PANEL)
        ax_.axis("off")
        ax_.text(0.04, 0.35, text, color=AMBER, fontsize=7.5,
                 fontweight="bold", transform=ax_.transAxes)

    # ── Combination navigation ──────────────────────────────────────────────────
    _lbl(0.935, "COMBINATION")
    ax_prev = fig.add_axes([SBL,               0.895, SBW * 0.38, 0.038])
    ax_cnav = fig.add_axes([SBL + SBW * 0.39,  0.895, SBW * 0.22, 0.038])
    ax_next = fig.add_axes([SBL + SBW * 0.62,  0.895, SBW * 0.38, 0.038])
    for _a in (ax_prev, ax_cnav, ax_next):
        _a.set_facecolor(PANEL); _a.axis("off")
    nav_txt = ax_cnav.text(0.5, 0.5, f"1 / {n}", transform=ax_cnav.transAxes,
                           color=TEXT, ha="center", va="center", fontsize=8.5)
    btn_prev = Button(ax_prev, "< Prev", color=AX_BG, hovercolor=BORDER)
    btn_next = Button(ax_next, "Next >",  color=AX_BG, hovercolor=BORDER)
    for b in (btn_prev, btn_next):
        b.label.set_color(TEXT); b.label.set_fontsize(8)

    # ── Weighting method checkboxes ─────────────────────────────────────────────
    _lbl(0.875, "WEIGHTING METHOD")
    ax_chk = fig.add_axes([SBL, 0.700, SBW, 0.173])
    ax_chk.set_facecolor(PANEL)
    for sp in ax_chk.spines.values():
        sp.set_edgecolor(BORDER)
    chk = CheckButtons(
        ax_chk,
        labels=[METHOD_LABELS[m] for m in METHODS],
        actives=[True] * len(METHODS),
    )
    for lbl, m in zip(chk.labels, METHODS):
        lbl.set_color(METHOD_COLORS[m])
        lbl.set_fontsize(8.5)
    chk.set_check_props({"color": CYAN, "linewidth": 1.5})
    chk.set_frame_props({"facecolor": PANEL, "edgecolor": BORDER})

    # ── MC method radio buttons ─────────────────────────────────────────────────
    _lbl(0.688, "MC METHOD")
    ax_radio = fig.add_axes([SBL, 0.445, SBW, 0.240])
    ax_radio.set_facecolor(PANEL)
    for sp in ax_radio.spines.values():
        sp.set_edgecolor(BORDER)
    radio_mc = RadioButtons(
        ax_radio,
        labels=[MC_METHODS[k] for k in MC_KEYS],
        activecolor=CYAN,
    )
    for lbl in radio_mc.labels:
        lbl.set_color(TEXT)
        lbl.set_fontsize(7.8)
    radio_mc.set_radio_props({"facecolor": PANEL, "edgecolor": BORDER})

    # ── N simulations text box ──────────────────────────────────────────────────
    _lbl(0.432, "N SIMULATIONS")
    ax_txt = fig.add_axes([SBL + 0.005, 0.390, SBW * 0.82, 0.038])
    ax_txt.set_facecolor(AX_BG)
    for sp in ax_txt.spines.values():
        sp.set_edgecolor(BORDER)
    txt_runs = TextBox(ax_txt, "", initial="10000",
                       color=AX_BG, hovercolor="#232342")
    txt_runs.label.set_color(TEXT)
    txt_runs.text_disp.set_color(CYAN)
    txt_runs.text_disp.set_fontsize(9)

    # ── Mode radio (Single / Rolling) ──────────────────────────────────────────
    _lbl(0.378, "MODE")
    ax_mode = fig.add_axes([SBL, 0.268, SBW, 0.108])
    ax_mode.set_facecolor(PANEL)
    for sp in ax_mode.spines.values():
        sp.set_edgecolor(BORDER)
    radio_mode = RadioButtons(
        ax_mode,
        labels=["Single time series", "Rolling windows"],
        activecolor=CYAN,
    )
    for lbl in radio_mode.labels:
        lbl.set_color(TEXT); lbl.set_fontsize(8)
    radio_mode.set_radio_props({"facecolor": PANEL, "edgecolor": BORDER})

    # ── Rolling windows slider ──────────────────────────────────────────────────
    _lbl(0.254, "ROLLING WINDOWS  (1 – 8)")
    ax_sld = fig.add_axes([SBL + 0.005, 0.212, SBW * 0.84, 0.033])
    ax_sld.set_facecolor(AX_BG)
    for sp in ax_sld.spines.values():
        sp.set_edgecolor(BORDER)
    slider_win = Slider(ax_sld, "", 1, 8, valinit=4, valstep=1, color=CYAN)
    slider_win.label.set_color(TEXT)
    slider_win.valtext.set_color(CYAN)
    slider_win.valtext.set_fontsize(9)

    # ── RUN button ──────────────────────────────────────────────────────────────
    ax_run = fig.add_axes([SBL + 0.01, 0.035, SBW - 0.02, 0.155])
    btn_run = Button(ax_run, "▶  RUN MONTE CARLO", color="#081808", hovercolor="#0e2e0e")
    btn_run.label.set_color(GREEN)
    btn_run.label.set_fontsize(11)
    btn_run.label.set_fontweight("bold")

    # ── Callbacks ───────────────────────────────────────────────────────────────

    def on_prev(_):
        state["idx"] = max(state["idx"] - 1, 0)
        nav_txt.set_text(f"{state['idx'] + 1} / {n}")
        fig.canvas.draw_idle()

    def on_next(_):
        state["idx"] = min(state["idx"] + 1, n - 1)
        nav_txt.set_text(f"{state['idx'] + 1} / {n}")
        fig.canvas.draw_idle()

    def on_chk(label: str):
        m = next(k for k, v in METHOD_LABELS.items() if v == label)
        if m in state["active_weights"]:
            state["active_weights"].discard(m)
        else:
            state["active_weights"].add(m)

    def on_mc(label: str):
        state["mc_method"] = next(k for k, v in MC_METHODS.items() if v == label)

    def on_mode(label: str):
        state["mode"] = "rolling" if "Rolling" in label else "single"

    def on_windows(val: float):
        state["n_windows"] = int(val)

    def on_run(_):
        raw = txt_runs.text.strip().replace(",", "").replace("_", "").replace(" ", "")
        try:
            state["n_sims"] = max(100, min(int(raw), 500_000))
        except ValueError:
            state["n_sims"] = 10_000
        _run_and_draw(fig, combinations, state, config)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    chk.on_clicked(on_chk)
    radio_mc.on_clicked(on_mc)
    radio_mode.on_clicked(on_mode)
    slider_win.on_changed(on_windows)
    btn_run.on_clicked(on_run)

    # Keep widget references alive (prevents GC killing callbacks)
    fig._refs = [
        btn_prev, btn_next, btn_run,
        chk, radio_mc, radio_mode, slider_win, txt_runs,
        nav_txt, state,
    ]

    # Initial title
    fig.suptitle(
        f"Monte Carlo  ·  {n} combination(s) loaded  ·  "
        "Select settings and click  ▶ RUN MONTE CARLO",
        color=CYAN, fontsize=11, y=0.995,
    )

    if _show:
        plt.show()
