"""
plot_factor.py — Interactive dashboard for Test 9: Multi-Factor Exposure.

Layout (dark theme, 2×2 grid):

  Panel TL — Cumulative Factor Returns
      Cumulative return of each factor (MOM, CARRY, VOL_SELL) over the full period.
      Shows whether each factor was profitable as a standalone systematic strategy.

  Panel TR — Factor Loadings (static, full-period)
      Horizontal bar chart: beta of each factor in the multi-factor OLS.
      Error bars = ±1.96·SE.  Colour = green (sig positive) / red (sig negative) / grey (not sig).

  Panel BL — Rolling Factor Betas (time series)
      How each factor beta evolves over time using a rolling OLS window.
      Dashed zero line. Shaded bands at ±2·SE approx.

  Panel BR — Rolling Alpha (multi-factor vs naive)
      Annualised rolling alpha from the multi-factor model vs the naive (1-factor) model.
      Alpha bars coloured by significance (bright when |t|≥2, dim otherwise).

Controls (bottom strip):
  [▾ Strategy]  MOM(d):[252]  CARRY(d):[63]  VOL(d):[21]  Roll Win(d):[252]  Step(d):[22]
  [? Explain]  [▶ Run]

Usage:
    from alphaforge.IndividualAnalysis.FactorExposure import plot_factor_dashboard

    plot_factor_dashboard(
        all_strategies = strategies,
        load_strategy  = load_fn,   # key -> (aligned, price_df, pair)
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
from matplotlib.widgets import Button, TextBox

from .factor_tests import (
    FactorExposure,
    FactorReport,
    print_factor_report,
    compute_rolling_factor_betas,
    _compute_factors,
    _MIN_OBS,
)

matplotlib.use("TkAgg")


# ── Colours ───────────────────────────────────────────────────────────────────

_BG        = "#0d1117"
_PANEL_BG  = "#161b22"
_TEXT      = "#e6edf3"
_GRID      = "#21262d"
_GREEN     = "#3fb950"
_RED       = "#f85149"
_AMBER     = "#d29922"
_BLUE      = "#58a6ff"
_CYAN      = "#79c0ff"
_ORANGE    = "#ffa657"
_MAGENTA   = "#d2a8ff"
_DIM       = "#4a5568"

_FACTOR_COLORS = {"MOM": _CYAN, "VOL_SELL": _MAGENTA}
_FACTOR_LABELS = {"MOM": "Momentum (TSMOM)", "VOL_SELL": "Vol-Selling"}


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4, alpha=0.8)


def _style_fig(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        _style_ax(ax)


# ── Main entry point ──────────────────────────────────────────────────────────

def plot_factor_dashboard(
    all_strategies: dict,
    load_strategy,          # callable: key -> (aligned, price_df, pair)
    initial_key:    str,
) -> None:
    """
    Open the interactive Factor Exposure dashboard (Test 9).

    Parameters
    ----------
    all_strategies : dict
        {key: trades_df} mapping from load_folder().
    load_strategy : callable
        Function (key: str) -> (aligned: DataFrame, price_df: DataFrame, pair: str).
    initial_key : str
        Strategy key to display on first load.
    """

    # ── State ─────────────────────────────────────────────────────────────────
    state = {"key": initial_key}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor=_BG)
    fig.canvas.manager.set_window_title("AlphaForge — Factor Exposure (Test 9)")

    # 2×2 panel grid + controls strip
    gs = gridspec.GridSpec(
        2, 2,
        left=0.07, right=0.97,
        top=0.92, bottom=0.16,
        hspace=0.42, wspace=0.28,
    )
    ax_tl = fig.add_subplot(gs[0, 0])   # cumulative factor returns
    ax_tr = fig.add_subplot(gs[0, 1])   # factor loadings (static)
    ax_bl = fig.add_subplot(gs[1, 0])   # rolling factor betas
    ax_br = fig.add_subplot(gs[1, 1])   # rolling alpha
    all_axes = [ax_tl, ax_tr, ax_bl, ax_br]
    _style_fig(fig, all_axes)

    # ── Controls ──────────────────────────────────────────────────────────────
    C = _PANEL_BG
    y0, h = 0.055, 0.045

    # Strategy button
    ax_sb = fig.add_axes([0.07, y0, 0.15, h])
    btn_strat = Button(ax_sb, "▾  Select Strategy", color=C, hovercolor="#1f2937")
    btn_strat.label.set_color(_TEXT); btn_strat.label.set_fontsize(9)

    # Parameter text boxes with labels above
    _params = [
        ("MOM (d)",      "252", 0.255),
        ("VOL (d)",      "21",  0.355),
        ("Roll Win (d)", "252", 0.455),
        ("Step (d)",     "22",  0.555),
    ]
    textboxes = {}
    for label, default, x in _params:
        fig.text(x, y0 + h + 0.013, label, color=_TEXT, fontsize=7.5,
                 ha="left", va="bottom")
        ax_tb = fig.add_axes([x, y0, 0.073, h])
        tb = TextBox(ax_tb, "", initial=default,
                     color=_PANEL_BG, hovercolor="#1f2937")
        tb.text_disp.set_color(_TEXT)
        tb.text_disp.set_fontsize(9)
        textboxes[label] = tb

    # Explain button
    ax_exp = fig.add_axes([0.660, y0, 0.10, h])
    btn_exp = Button(ax_exp, "? Explain", color=C, hovercolor="#1f2937")
    btn_exp.label.set_color(_TEXT); btn_exp.label.set_fontsize(9)

    # Run button
    ax_run = fig.add_axes([0.775, y0, 0.195, h])
    btn_run = Button(ax_run, "▶  Run", color="#1a4731", hovercolor="#22c55e")
    btn_run.label.set_color(_GREEN); btn_run.label.set_fontsize(9)

    # Strategy name label
    strat_label = fig.text(
        0.07, 0.965, f"Strategy: {initial_key}",
        color=_BLUE, fontsize=9, ha="left", va="top",
    )

    # ── Strategy picker ────────────────────────────────────────────────────────
    def _open_strategy_picker(_event):
        popup = tk.Toplevel()
        popup.title("Select Strategy")
        popup.configure(bg=_BG)
        popup.geometry("420x520")

        tk.Label(popup, text="Filter:", bg=_BG, fg=_TEXT,
                 font=("Consolas", 10)).pack(anchor="w", padx=8, pady=(8, 0))
        filter_var = tk.StringVar()
        entry = tk.Entry(popup, textvariable=filter_var, bg=_PANEL_BG,
                         fg=_TEXT, insertbackground=_TEXT, font=("Consolas", 10))
        entry.pack(fill="x", padx=8, pady=4)

        frame = tk.Frame(popup, bg=_BG)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        lb = tk.Listbox(frame, yscrollcommand=sb.set,
                        bg=_PANEL_BG, fg=_TEXT, selectbackground="#1f4068",
                        font=("Consolas", 10), activestyle="none")
        lb.pack(fill="both", expand=True)
        sb.config(command=lb.yview)

        all_keys = sorted(all_strategies.keys())

        def _refresh(*_):
            q = filter_var.get().lower()
            lb.delete(0, "end")
            for k in all_keys:
                if q in k.lower():
                    lb.insert("end", k)

        filter_var.trace_add("write", _refresh)
        _refresh()

        def _select(_event=None):
            sel = lb.curselection()
            if not sel:
                return
            chosen = lb.get(sel[0])
            state["key"] = chosen
            strat_label.set_text(f"Strategy: {chosen}")
            fig.canvas.draw_idle()
            popup.destroy()

        lb.bind("<Double-Button-1>", _select)
        tk.Button(popup, text="Select", command=_select,
                  bg="#1a4731", fg=_GREEN, font=("Consolas", 10)).pack(pady=6)

    btn_strat.on_clicked(_open_strategy_picker)

    # ── Explain popup ──────────────────────────────────────────────────────────
    _EXPLAIN = dedent("""\
        TEST 9 — MULTI-FACTOR EXPOSURE
        ══════════════════════════════════════════════════════════════

        WHAT THIS TEST DOES
        ────────────────────────────────────────────────────────────
        A strategy can look like it generates alpha in a simple OLS vs the asset.
        But that apparent alpha might actually be compensation for systematic risk
        exposures that a 1-factor model misses entirely.

        This test checks 2 well-known systematic factors (built from your price
        data — no external data needed):

          FACTOR 1 — Momentum (TSMOM)
          ─────────────────────────────────────────────────────────
          Signal  = sign of cumulative return from t-MOM(d) to t-22d
          Return  = signal × daily asset return
          Meaning : does the strategy tend to be long when the asset has
                    been trending up for the past year (minus last month)?
          Parameter: MOM(d) — the lookback for the momentum signal.
                     Default 252 (≈ 1 year).  Reduce to get shorter-term
                     momentum, increase for longer-term.

          FACTOR 2 — Vol-Selling
          ─────────────────────────────────────────────────────────
          Return  = -(change in VOL(d)-day realised vol) / long-run vol
          Meaning : does the strategy profit when volatility falls and
                    lose when volatility spikes? This is a short-vol /
                    short-gamma payoff profile.
          Parameter: VOL(d) — the realised vol estimation window.
                     Default 21 (≈ 1 month).

        ROLLING PARAMETERS
        ────────────────────────────────────────────────────────────
          Roll Win(d) — lookback window for the rolling OLS.
                        How many days of data each rolling point uses.
                        Larger = smoother but slower to detect regime changes.
                        Default 252 (≈ 1 year).

          Step(d)     — how many days to advance between rolling windows.
                        Smaller = more data points but slower to compute.
                        Default 22 (≈ 1 month).

        HOW TO READ THE PANELS
        ────────────────────────────────────────────────────────────
          Top-Left: Cumulative Factor Returns
            Shows how each systematic factor would have performed as a
            standalone strategy. A rising line = the risk premium paid off.
            Compares MOM (cyan), CARRY (orange), VOL_SELL (magenta).

          Top-Right: Factor Loadings (full-period)
            Horizontal bars show each factor's beta in the multi-factor OLS.
            Error bars = 95% CI (±1.96·SE).
            Green = significant positive loading.
            Red   = significant negative loading.
            Grey  = not significant.
            The number on each bar is the t-statistic.

          Bottom-Left: Rolling Factor Betas
            Same betas, but computed on a rolling window.
            Shows whether exposures are stable or shifting over time.
            A beta that drifts toward zero = the strategy is decoupling
            from that factor over time (improving quality).

          Bottom-Right: Rolling Alpha
            Solid line = multi-factor residual alpha (annualised %).
            Dashed line = naive alpha (vs asset only).
            Bright colour = significant window (|t| ≥ 2).
            Dim colour = not significant.
            If the two lines track closely → factors explain little.
            If the solid line is much lower → factors were eating alpha.

        VERDICT DEFINITIONS
        ────────────────────────────────────────────────────────────
          ALPHA SURVIVES  — alpha significant after all 3 factors.
                            The strategy has genuine edge.
          ALPHA REDUCED   — alpha reduced but still significant.
                            Some edge, with partial factor exposure.
          FACTOR BETA     — >50% of alpha explained by factors and
                            alpha no longer significant. The strategy
                            is largely systematic risk premium, not edge.
          INCONCLUSIVE    — insufficient evidence either way.
    """)

    def _open_explain(_event):
        popup = tk.Toplevel()
        popup.title("Factor Exposure — Explanation")
        popup.configure(bg=_BG)
        popup.geometry("700x640")
        frame = tk.Frame(popup, bg=_BG)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        sb = tk.Scrollbar(frame); sb.pack(side="right", fill="y")
        txt = tk.Text(frame, yscrollcommand=sb.set,
                      bg=_PANEL_BG, fg=_TEXT, font=("Consolas", 10),
                      wrap="word", relief="flat", padx=8, pady=8)
        txt.pack(fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.insert("1.0", _EXPLAIN)
        txt.config(state="disabled")

    btn_exp.on_clicked(_open_explain)

    # ── Panel drawing functions ────────────────────────────────────────────────

    def _draw_cumulative_factors(ax: plt.Axes, report: FactorReport) -> None:
        ax.set_title("Cumulative Factor Returns", fontsize=9, pad=4)
        ax.axhline(0, color=_GRID, linewidth=0.8)
        for fname, color in _FACTOR_COLORS.items():
            cum = (1 + report.factor_returns[fname].fillna(0)).cumprod() - 1
            ax.plot(cum.index, cum * 100, color=color, linewidth=1.2,
                    label=_FACTOR_LABELS[fname], alpha=0.9)
        ax.set_ylabel("Cumulative Return (%)", fontsize=8)
        ax.legend(fontsize=7.5, facecolor=_PANEL_BG, labelcolor=_TEXT,
                  framealpha=0.8, loc="upper left")
        ax.tick_params(axis="x", labelrotation=20, labelsize=7.5)

    def _draw_factor_loadings(ax: plt.Axes, report: FactorReport) -> None:
        ax.set_title("Factor Loadings  (full period, β ± 95% CI)", fontsize=9, pad=4)
        ax.axvline(0, color=_TEXT, linewidth=0.6, alpha=0.4)
        mf     = report.multi_factor
        fnames = list(mf.betas.keys())
        betas  = [mf.betas[f]    for f in fnames]
        errs   = [1.96 * mf.beta_ses[f] for f in fnames]
        colors = [
            (_GREEN if b > 0 else _RED) if abs(mf.beta_tstats[f]) >= 2.0 else _DIM
            for b, f in zip(betas, fnames)
        ]
        y_pos = list(range(len(fnames)))
        ax.barh(y_pos, betas, xerr=errs, color=colors, alpha=0.85, height=0.5,
                error_kw={"ecolor": _TEXT, "capsize": 4, "linewidth": 1.1}, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_FACTOR_LABELS[f] for f in fnames], fontsize=8)
        ax.set_xlabel("β coefficient", fontsize=8)
        for i, (f, b) in enumerate(zip(fnames, betas)):
            t = mf.beta_tstats[f]
            offset = max(abs(b) * 0.03, 0.0005)
            ax.text(b + (offset if b >= 0 else -offset), i,
                    f"t={t:+.2f}",
                    color=_TEXT, fontsize=7.5, va="center",
                    ha="left" if b >= 0 else "right")
        # Verdict badge
        vc = {
            "ALPHA SURVIVES": _GREEN, "ALPHA REDUCED": _AMBER,
            "FACTOR BETA": _RED, "INCONCLUSIVE": _DIM,
        }.get(report.verdict, _TEXT)
        ax.text(0.98, 0.97, report.verdict, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=vc, fontweight="bold")

    def _draw_rolling_betas(ax: plt.Axes, roll_df: pd.DataFrame) -> None:
        ax.set_title("Rolling Factor Betas", fontsize=9, pad=4)
        ax.axhline(0, color=_TEXT, linewidth=0.6, alpha=0.4)
        if roll_df.empty:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=_DIM, fontsize=9)
            return
        for fname, color in _FACTOR_COLORS.items():
            col = f"b_{fname}"
            if col in roll_df.columns:
                ax.plot(roll_df.index, roll_df[col], color=color,
                        linewidth=1.2, label=_FACTOR_LABELS[fname], alpha=0.9)
        ax.set_ylabel("Beta", fontsize=8)
        ax.legend(fontsize=7.5, facecolor=_PANEL_BG, labelcolor=_TEXT,
                  framealpha=0.8, loc="upper left")
        ax.tick_params(axis="x", labelrotation=20, labelsize=7.5)

    def _draw_rolling_alpha(ax: plt.Axes, roll_df: pd.DataFrame) -> None:
        ax.set_title("Rolling Alpha — Multi-Factor vs Naive (annualised %)", fontsize=9, pad=4)
        ax.axhline(0, color=_TEXT, linewidth=0.6, alpha=0.4)
        if roll_df.empty:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=_DIM, fontsize=9)
            return

        # Naive alpha — dashed
        ax.plot(roll_df.index, roll_df["naive_alpha_ann"] * 100,
                color=_TEXT, linewidth=0.9, linestyle="--", alpha=0.45,
                label="Naive α (vs asset)", zorder=2)

        # Multi-factor alpha — coloured by significance
        dates  = roll_df.index
        alphas = roll_df["alpha_ann"] * 100
        tstats = roll_df["alpha_tstat"]

        # Fill area
        ax.fill_between(dates, 0, alphas,
                        where=alphas >= 0, color=_GREEN, alpha=0.08, zorder=1)
        ax.fill_between(dates, 0, alphas,
                        where=alphas < 0, color=_RED, alpha=0.08, zorder=1)

        # Segment plot by significance
        sig   = tstats.abs() >= 2.0
        for is_sig, color, lw in [(True, _GREEN, 1.5), (False, _DIM, 1.0)]:
            mask = sig == is_sig
            if mask.any():
                # Plot individual segments to handle gaps
                idx_list = dates[mask]
                # Use nan-masking trick to break lines cleanly
                y_masked = alphas.copy().astype(float)
                y_masked[~mask] = np.nan
                ax.plot(dates, y_masked, color=color, linewidth=lw,
                        alpha=0.95 if is_sig else 0.5, zorder=3)

        # Proxy for legend
        from matplotlib.lines import Line2D
        leg_handles = [
            Line2D([0], [0], color=_GREEN, linewidth=1.5, label="Multi-factor α (sig)"),
            Line2D([0], [0], color=_DIM,   linewidth=1.0, label="Multi-factor α (not sig)"),
            Line2D([0], [0], color=_TEXT,  linewidth=0.9, linestyle="--", alpha=0.5,
                   label="Naive α (vs asset)"),
        ]
        ax.legend(handles=leg_handles, fontsize=7.5, facecolor=_PANEL_BG,
                  labelcolor=_TEXT, framealpha=0.8, loc="upper left")
        ax.set_ylabel("Annualised Alpha (%)", fontsize=8)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7.5)

    # ── Full redraw ────────────────────────────────────────────────────────────

    def _redraw(report: FactorReport, roll_df: pd.DataFrame) -> None:
        for ax in all_axes:
            ax.cla()
        _style_fig(fig, all_axes)

        _draw_cumulative_factors(ax_tl, report)
        _draw_factor_loadings(ax_tr, report)
        _draw_rolling_betas(ax_bl, roll_df)
        _draw_rolling_alpha(ax_br, roll_df)

        fig.suptitle(
            f"Factor Exposure (Test 9)  —  {report.strategy_name}"
            f"   [{report.date_start}  →  {report.date_end}]"
            f"   n = {report.n_obs}",
            color=_TEXT, fontsize=9.5, y=0.98,
        )
        strat_label.set_text(f"Strategy: {report.strategy_name}")
        fig.canvas.draw_idle()

    # ── Run callback ───────────────────────────────────────────────────────────

    def _get_int(tb: TextBox, default: int) -> int:
        try:
            return max(1, int(tb.text.strip()))
        except ValueError:
            return default

    def _on_run(_event) -> None:
        key = state["key"]

        # Read parameters
        mom_long  = _get_int(textboxes["MOM (d)"],      252)
        vol_window = _get_int(textboxes["VOL (d)"],      21)
        roll_win  = _get_int(textboxes["Roll Win (d)"], 252)
        step      = _get_int(textboxes["Step (d)"],     22)

        print(f"\n  Running factor exposure: {key}")
        print(f"  MOM={mom_long}d  VOL={vol_window}d"
              f"  RollWin={roll_win}d  Step={step}d\n")

        try:
            aligned, price_df, pair = load_strategy(key)
        except Exception as exc:
            print(f"[FactorDashboard] Error loading '{key}': {exc}")
            return

        try:
            tester = FactorExposure(
                aligned, price_df,
                strategy_name = key,
                pair          = pair,
                mom_long      = mom_long,
                vol_window    = vol_window,
            )
            report = tester.run()
        except Exception as exc:
            print(f"[FactorDashboard] Error computing factors: {exc}")
            return

        # Build merged df for rolling computation
        try:
            factors = _compute_factors(
                price_df,
                mom_long   = mom_long,
                vol_window = vol_window,
            )
            factors.index = pd.to_datetime(factors.index).normalize()
            aligned.index = pd.to_datetime(aligned.index).normalize()
            df_merged = aligned[["strategy", "asset"]].join(factors, how="inner").dropna()
            roll_df = compute_rolling_factor_betas(df_merged, window=roll_win, step=step)
        except Exception as exc:
            print(f"[FactorDashboard] Error computing rolling betas: {exc}")
            roll_df = pd.DataFrame()

        print_factor_report(report)
        _redraw(report, roll_df)

    btn_run.on_clicked(_on_run)

    # ── Initial run ────────────────────────────────────────────────────────────
    _on_run(None)
    plt.show()
