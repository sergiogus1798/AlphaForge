"""
plot_regime_hmm.py — Interactive dashboard for returns-based HMM regime comparison.

Layout (dark theme, 3-row grid):

  Row 1 — Strategy Weekly Returns
      Bars coloured by strategy HMM state (Red=Bad, Amber=Neutral, Green=Good).
      Horizontal zero line.  State legend top-right.

  Row 2 — Market Weekly Returns
      Same format for the market HMM states.
      Shares x-axis with Row 1 so you can directly compare timelines.

  Row 3 (split 2 columns):
    Left  — Confusion Matrix heatmap
        Rows = market state, Cols = strategy state.
        Cell value = P(strategy state | market state).
        A near-uniform matrix means genuine independence.

    Right — Per-State Statistics
        Grouped bar chart: annualised mean return per state, strategy vs market.
        Shows how well-separated the states are for each series.

Controls (bottom strip):
  [▾ Strategy]  N States:[3]  N Seeds:[30]  [? Explain]  [▶ Run]

Usage:
    from alphaforge.IndividualAnalysis.RegimeHMM import plot_regime_hmm_dashboard

    plot_regime_hmm_dashboard(
        all_strategies = strategies,
        load_strategy  = load_fn,     # key -> (aligned, price_df, pair)
        initial_key    = "AUDJPY/Strategy 11.12.166",
    )
"""

from __future__ import annotations

import tkinter as tk
from textwrap import dedent

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.widgets import Button, TextBox

from .regime_hmm import RegimeHMM, RegimeHMMReport, print_regime_hmm_report, _STATE_LABELS

matplotlib.use("TkAgg")


# ── Colours ───────────────────────────────────────────────────────────────────

_BG       = "#0d1117"
_PANEL_BG = "#161b22"
_TEXT     = "#e6edf3"
_GRID     = "#21262d"
_GREEN    = "#3fb950"
_RED      = "#f85149"
_AMBER    = "#d29922"
_BLUE     = "#58a6ff"
_DIM      = "#4a5568"

# State colours: index 0=worst → red, 1=neutral → amber, 2=best → green
_STATE_COLORS_3 = [_RED, _AMBER, _GREEN]
_STATE_COLORS_2 = [_RED, _GREEN]
_STATE_COLORS_4 = [_RED, "#e05252", _AMBER, _GREEN]

def _state_colors(n: int) -> list[str]:
    if n == 2: return _STATE_COLORS_2
    if n == 4: return _STATE_COLORS_4
    return _STATE_COLORS_3  # default 3


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4, alpha=0.7)


def _style_fig(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        _style_ax(ax)


# ── Panel drawers ─────────────────────────────────────────────────────────────

def _draw_weekly_returns(
    ax:       plt.Axes,
    returns:  pd.Series,
    states:   np.ndarray,
    n_states: int,
    title:    str,
) -> None:
    ax.set_title(title, fontsize=9, pad=4)
    ax.axhline(0, color=_TEXT, linewidth=0.6, alpha=0.4)

    colors = _state_colors(n_states)
    labels = _STATE_LABELS.get(n_states, [str(i) for i in range(n_states)])

    bar_colors = [colors[s] for s in states]
    ax.bar(returns.index, returns.values * 100,
           color=bar_colors, alpha=0.8, width=5.0, zorder=3)

    ax.set_ylabel("Weekly Return (%)", fontsize=8)
    ax.tick_params(axis="x", labelrotation=20, labelsize=7.5)

    # Legend
    handles = [Patch(color=colors[i], alpha=0.85, label=f"State {i}: {labels[i]}")
               for i in range(n_states)]
    ax.legend(handles=handles, fontsize=7.5, facecolor=_PANEL_BG, labelcolor=_TEXT,
              framealpha=0.8, loc="upper right")


def _draw_confusion_matrix(
    ax:      plt.Axes,
    report:  RegimeHMMReport,
) -> None:
    cm_norm = report.confusion_matrix_norm
    n       = report.n_states
    labels  = _STATE_LABELS.get(n, [str(i) for i in range(n)])

    ax.set_title(
        f"Confusion Matrix  (rows=Market, cols=Strategy)\n"
        f"P(strategy state | market state)   Cramér's V = {report.cramers_v:.3f}",
        fontsize=8.5, pad=6,
    )

    # Custom colormap: dark → bright blue
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "dark_blue", [_PANEL_BG, "#1f4068", _BLUE], N=256
    )
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Cell annotations
    for i in range(n):
        for j in range(n):
            val  = cm_norm[i, j]
            cnt  = report.confusion_matrix[i, j]
            text_color = _TEXT if val < 0.6 else _BG
            ax.text(j, i, f"{val:.2f}\n({cnt})",
                    ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Strat: {lb}" for lb in labels], fontsize=8)
    ax.set_yticklabels([f"Mkt: {lb}"  for lb in labels], fontsize=8)
    ax.set_xlabel("Strategy State", fontsize=8)
    ax.set_ylabel("Market State",   fontsize=8)
    ax.grid(False)

    # Verdict badge
    vc = {
        "INDEPENDENT":        _GREEN,
        "WEAK ALIGNMENT":     "#79c0ff",
        "MODERATE ALIGNMENT": _AMBER,
        "STRONG ALIGNMENT":   _RED,
    }.get(report.verdict, _TEXT)
    ax.text(0.99, -0.22, report.verdict,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=vc, fontweight="bold")


def _draw_state_stats(
    ax:     plt.Axes,
    report: RegimeHMMReport,
) -> None:
    ax.set_title("Annualised Return by State  (Strategy vs Market)", fontsize=9, pad=4)
    ax.axhline(0, color=_TEXT, linewidth=0.6, alpha=0.4)

    n      = report.n_states
    labels = _STATE_LABELS.get(n, [str(i) for i in range(n)])
    colors = _state_colors(n)

    x       = np.arange(n)
    width   = 0.35
    s_means = [st.mean_ann * 100 for st in report.strategy_stats]
    m_means = [st.mean_ann * 100 for st in report.market_stats]

    bars_s = ax.bar(x - width / 2, s_means, width, color=colors,
                    alpha=0.85, label="Strategy", zorder=3)
    bars_m = ax.bar(x + width / 2, m_means, width, color=colors,
                    alpha=0.45, hatch="///", edgecolor=_TEXT,
                    linewidth=0.5, label="Market", zorder=3)

    # Value labels on bars
    for bar, val in zip(list(bars_s) + list(bars_m), s_means + m_means):
        ypos = val + (0.3 if val >= 0 else -0.3)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.1f}%", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7, color=_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Annualised Return (%)", fontsize=8)

    # Custom legend: solid = strategy, hatched = market
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#58a6ff", alpha=0.85,   label="Strategy"),
        Patch(facecolor="#58a6ff", alpha=0.45, hatch="///",
              edgecolor=_TEXT, linewidth=0.5, label="Market"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, facecolor=_PANEL_BG,
              labelcolor=_TEXT, framealpha=0.8)

    # % time annotations below x-axis labels
    for i, st in enumerate(report.strategy_stats):
        ax.text(i - width / 2, ax.get_ylim()[0] * 0.97,
                f"{st.pct_time:.0f}%", ha="center", va="top",
                fontsize=7, color=colors[i], alpha=0.8)


# ── Explain text ──────────────────────────────────────────────────────────────

_EXPLAIN = dedent("""\
    REGIME HMM — RETURNS-BASED COMPARISON
    ══════════════════════════════════════════════════════════════════

    WHAT THIS MODULE DOES
    ──────────────────────────────────────────────────────────────────
    Standard regime analysis detects market regimes from price data
    and then measures how your strategy performs inside them.

    This module does something fundamentally different:
    it fits two INDEPENDENT Hidden Markov Models — one on your
    strategy's weekly returns, one on the market's weekly returns —
    and then compares whether the two state sequences are aligned.

    The core question:
      "Are my strategy's good/bad periods driven by the market,
       or do they follow their own independent rhythm?"

    If INDEPENDENT → your strategy has genuine alpha. It finds
                     its own edge regardless of what the market does.

    If STRONGLY ALIGNED → your strategy is essentially a beta
                          vehicle. Its regime structure mirrors the
                          market's. The "alpha" is disguised beta.

    WHY WEEKLY RETURNS?
    ──────────────────────────────────────────────────────────────────
    Daily returns are too noisy — the HMM ends up fitting random
    fluctuations rather than genuine regime structure. Weekly
    aggregation smooths enough that state transitions become
    meaningful (regime changes at the month-to-quarter timescale,
    not day-to-day noise).

    WHY 3 STATES?
    ──────────────────────────────────────────────────────────────────
    2 states (good/bad) misses the most strategically important
    regime — the choppy "treading water" middle state where you
    should be reducing size, not stopping, but not pressing either.

    3 states maps to real trading experience:
      Bad     — negative mean, often high variance. Strategy is
                working against the current market environment.
      Neutral — near-zero mean, moderate variance. No clear edge.
                The market isn't giving clean signals.
      Good    — positive mean, lower variance. Your edge is
                expressing itself cleanly. Size up here.

    CONTROLS
    ──────────────────────────────────────────────────────────────────
    N States — number of HMM states (2–4). Default 3.
    N Seeds  — how many random initialisations to try. The best
               log-likelihood wins. More seeds = more stable fit
               but slower. Default 30 is usually enough.

    HOW TO READ THE PANELS
    ──────────────────────────────────────────────────────────────────
    Top Panel — Strategy Weekly Returns
      Each bar is the strategy's weekly return. Colour = the HMM
      state assigned to that week (Red=Bad, Amber=Neutral, Green=Good).
      Look for: are green stretches long and persistent? Do red
      stretches cluster together (high persistence = real regimes)?

    Middle Panel — Market Weekly Returns
      Same format for the market. Compare the colour patterns
      between the two panels by eye. If they look similar,
      the strategy is market-driven. If different, it's independent.

    Bottom Left — Confusion Matrix
      The key diagnostic. Rows = market state, Cols = strategy state.
      Each cell = P(strategy state | market state).

      A UNIFORM matrix (all cells ≈ 1/N) means:
        → The strategy's state is equally likely regardless of
          what state the market is in. INDEPENDENCE. GENUINE ALPHA.

      A DIAGONAL matrix (large values on the diagonal) means:
        → When the market is Bad, your strategy is Bad too.
          When the market is Good, you're Good. PURE BETA.

      The number in parentheses is the raw count of weeks.
      Cramér's V summarises the association: 0=independent, 1=correlated.

    Bottom Right — Annualised Return by State
      Solid bars = strategy. Hatched bars = market.
      Shows how well-separated the states are.
      Good separation (big gap between Bad and Good bars) means
      the HMM has found meaningful structure.
      The small % number below each bar = % of time in that state.

    VERDICT THRESHOLDS
    ──────────────────────────────────────────────────────────────────
    INDEPENDENT         V < 0.10  Genuine alpha, no market alignment
    WEAK ALIGNMENT      V < 0.25  Mostly independent, some sensitivity
    MODERATE ALIGNMENT  V < 0.45  Partial beta exposure
    STRONG ALIGNMENT    V ≥ 0.45  Primarily market-driven (beta)
""")


# ── Main dashboard ────────────────────────────────────────────────────────────

def plot_regime_hmm_dashboard(
    all_strategies: dict,
    load_strategy,          # callable: key -> (aligned, price_df, pair)
    initial_key:    str,
) -> None:
    """
    Open the interactive Regime HMM dashboard.

    Parameters
    ----------
    all_strategies : dict
        {key: trades_df} from load_folder().
    load_strategy : callable
        (key: str) -> (aligned: DataFrame, price_df: DataFrame, pair: str)
    initial_key : str
        Strategy to load on startup.
    """

    state = {"key": initial_key}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12), facecolor=_BG)
    fig.canvas.manager.set_window_title("AlphaForge — Regime HMM")

    gs = gridspec.GridSpec(
        3, 2,
        left=0.07, right=0.97,
        top=0.92, bottom=0.14,
        hspace=0.52, wspace=0.28,
        height_ratios=[1, 1, 1.1],
    )

    ax_strat = fig.add_subplot(gs[0, :])           # strategy returns (full width)
    ax_mkt   = fig.add_subplot(gs[1, :],
                               sharex=ax_strat)    # market returns (shared x)
    ax_cm    = fig.add_subplot(gs[2, 0])           # confusion matrix
    ax_stats = fig.add_subplot(gs[2, 1])           # state stats

    all_axes = [ax_strat, ax_mkt, ax_cm, ax_stats]
    _style_fig(fig, all_axes)

    # ── Controls ──────────────────────────────────────────────────────────────
    C  = _PANEL_BG
    y0 = 0.055
    h  = 0.045

    ax_sb = fig.add_axes([0.07, y0, 0.18, h])
    btn_strat = Button(ax_sb, "▾  Select Strategy", color=C, hovercolor="#1f2937")
    btn_strat.label.set_color(_TEXT); btn_strat.label.set_fontsize(9)

    _params = [("N States", "3", 0.29), ("N Seeds",  "30", 0.39)]
    textboxes = {}
    for label, default, x in _params:
        fig.text(x, y0 + h + 0.013, label, color=_TEXT, fontsize=8, ha="left")
        ax_tb = fig.add_axes([x, y0, 0.082, h])
        tb = TextBox(ax_tb, "", initial=default, color=_PANEL_BG, hovercolor="#1f2937")
        tb.text_disp.set_color(_TEXT); tb.text_disp.set_fontsize(9)
        textboxes[label] = tb

    ax_exp = fig.add_axes([0.510, y0, 0.10, h])
    btn_exp = Button(ax_exp, "? Explain", color=C, hovercolor="#1f2937")
    btn_exp.label.set_color(_TEXT); btn_exp.label.set_fontsize(9)

    ax_run = fig.add_axes([0.625, y0, 0.345, h])
    btn_run = Button(ax_run, "▶  Run", color="#1a4731", hovercolor="#22c55e")
    btn_run.label.set_color(_GREEN); btn_run.label.set_fontsize(9)

    strat_label = fig.text(
        0.07, 0.965, f"Strategy: {initial_key}",
        color=_BLUE, fontsize=9, ha="left", va="top",
    )

    # ── Strategy picker ────────────────────────────────────────────────────────
    def _open_picker(_event):
        popup = tk.Toplevel()
        popup.title("Select Strategy")
        popup.configure(bg=_BG)
        popup.geometry("420x520")

        tk.Label(popup, text="Filter:", bg=_BG, fg=_TEXT,
                 font=("Consolas", 10)).pack(anchor="w", padx=8, pady=(8, 0))
        fv = tk.StringVar()
        tk.Entry(popup, textvariable=fv, bg=_PANEL_BG, fg=_TEXT,
                 insertbackground=_TEXT, font=("Consolas", 10)).pack(fill="x", padx=8, pady=4)

        frame = tk.Frame(popup, bg=_BG); frame.pack(fill="both", expand=True, padx=8, pady=4)
        sb2 = tk.Scrollbar(frame); sb2.pack(side="right", fill="y")
        lb = tk.Listbox(frame, yscrollcommand=sb2.set, bg=_PANEL_BG, fg=_TEXT,
                        selectbackground="#1f4068", font=("Consolas", 10), activestyle="none")
        lb.pack(fill="both", expand=True)
        sb2.config(command=lb.yview)

        all_keys = sorted(all_strategies.keys())

        def _refresh(*_):
            q = fv.get().lower(); lb.delete(0, "end")
            for k in all_keys:
                if q in k.lower(): lb.insert("end", k)

        fv.trace_add("write", _refresh); _refresh()

        def _select(_e=None):
            sel = lb.curselection()
            if not sel: return
            chosen = lb.get(sel[0])
            state["key"] = chosen
            strat_label.set_text(f"Strategy: {chosen}")
            fig.canvas.draw_idle(); popup.destroy()

        lb.bind("<Double-Button-1>", _select)
        tk.Button(popup, text="Select", command=_select,
                  bg="#1a4731", fg=_GREEN, font=("Consolas", 10)).pack(pady=6)

    btn_strat.on_clicked(_open_picker)

    # ── Explain popup ──────────────────────────────────────────────────────────
    def _open_explain(_event):
        popup = tk.Toplevel()
        popup.title("Regime HMM — Explanation")
        popup.configure(bg=_BG)
        popup.geometry("700x680")
        frame = tk.Frame(popup, bg=_BG); frame.pack(fill="both", expand=True, padx=10, pady=10)
        sb2 = tk.Scrollbar(frame); sb2.pack(side="right", fill="y")
        txt = tk.Text(frame, yscrollcommand=sb2.set, bg=_PANEL_BG, fg=_TEXT,
                      font=("Consolas", 10), wrap="word", relief="flat", padx=8, pady=8)
        txt.pack(fill="both", expand=True)
        sb2.config(command=txt.yview)
        txt.insert("1.0", _EXPLAIN); txt.config(state="disabled")

    btn_exp.on_clicked(_open_explain)

    # ── Redraw ─────────────────────────────────────────────────────────────────
    def _redraw(report: RegimeHMMReport) -> None:
        for ax in all_axes:
            ax.cla()
        _style_fig(fig, all_axes)

        _draw_weekly_returns(
            ax_strat,
            report.strategy_weekly,
            report.strategy_states,
            report.n_states,
            "Strategy Weekly Returns  (coloured by HMM state)",
        )
        _draw_weekly_returns(
            ax_mkt,
            report.market_weekly,
            report.market_states,
            report.n_states,
            "Market Weekly Returns  (coloured by HMM state)",
        )
        _draw_confusion_matrix(ax_cm, report)
        _draw_state_stats(ax_stats, report)

        fig.suptitle(
            f"Regime HMM  —  {report.strategy_name}"
            f"   [{report.date_start}  →  {report.date_end}]"
            f"   {report.n_weeks} weeks   {report.n_states} states",
            color=_TEXT, fontsize=9.5, y=0.98,
        )
        strat_label.set_text(f"Strategy: {report.strategy_name}")
        fig.canvas.draw_idle()

    # ── Run callback ───────────────────────────────────────────────────────────
    def _get_int(tb: TextBox, default: int, lo: int = 1, hi: int = 999) -> int:
        try:
            return max(lo, min(hi, int(tb.text.strip())))
        except ValueError:
            return default

    def _on_run(_event) -> None:
        key      = state["key"]
        n_states = _get_int(textboxes["N States"], 3, lo=2, hi=6)
        n_seeds  = _get_int(textboxes["N Seeds"],  30, lo=1, hi=200)

        print(f"\n  Regime HMM: {key}  |  states={n_states}  seeds={n_seeds}\n")

        try:
            aligned, price_df, pair = load_strategy(key)
        except Exception as exc:
            print(f"[RegimeHMM] Error loading '{key}': {exc}"); return

        try:
            model  = RegimeHMM(aligned, strategy_name=key, pair=pair)
            report = model.run(n_states=n_states, n_seeds=n_seeds)
        except Exception as exc:
            print(f"[RegimeHMM] Error fitting HMM: {exc}"); return

        print_regime_hmm_report(report)
        _redraw(report)

    btn_run.on_clicked(_on_run)

    # ── Initial run ────────────────────────────────────────────────────────────
    _on_run(None)
    plt.show()
