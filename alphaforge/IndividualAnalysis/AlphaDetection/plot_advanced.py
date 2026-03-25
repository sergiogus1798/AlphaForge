"""
plot_advanced.py — Visualization for the advanced alpha-validity tests.

Always opens TWO figures simultaneously:

  Static figure (4×2 = 7 subplots) — Tests 1, 2, 3, 4, 6:
    [0,0] ACF bar chart        — residual autocorrelation lags 1-20
    [0,1] Return histogram     — distribution shape, active days only
    [1,0] Nonlinear beta       — scatter with linear + quadratic fits
    [1,1] Asymmetric beta      — scatter split by up/down market days
    [2,0] Beta comparison bars — β_up, β_overall, β_down
    [3,0] Placebo: individual  — null distribution (random shuffle)
    [3,1] Placebo: block       — null distribution (block shuffle)

  Rolling figure (interactive 3×2 + 1 wide) — live control panel at the bottom:
    [0,0] Rolling lag-1 ACF             — Test 1
    [0,1] Rolling skewness              — Test 2a
    [1,0] Rolling excess kurtosis       — Test 2b
    [1,1] Rolling β₂                    — Test 3
    [2,0] Rolling β_up / β_down         — Test 4a
    [2,1] Rolling asymmetry ratio       — Test 4b
    [3, full] Rolling alpha vs null     — Test 6 (rolling placebo)
    [bottom] TextBox (window size) + Run button + Explain button

Usage:
    from alphaforge.IndividualAnalysis.AlphaDetection.plot_advanced import plot_advanced_report
    from alphaforge.IndividualAnalysis.AlphaDetection import AlphaDetector
    from alphaforge.IndividualAnalysis.AlphaDetection.advanced_tests import AdvancedTests

    detector = AlphaDetector(trades_df, price_df, strategy_name="AUDJPY7", pair="AUDJPY")
    aligned  = detector.build_returns()
    core     = detector.run_core_regression(aligned)

    tester = AdvancedTests(aligned, core)
    plot_advanced_report(tester, detector.strategy_name, detector.pair)
"""

import warnings

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates
from matplotlib.widgets import Button, TextBox
import numpy as np
import pandas as pd
import scipy.stats as stats

from .advanced_tests import (
    AdvancedTests,
    AdvancedReport,
    AutocorrelationResult,
    DistributionResult,
    NonlinearBetaResult,
    AsymmetricBetaResult,
    PlaceboResult,
)

# ── Colour palette ────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
DIM    = "#8b949e"
AMBER  = "#f0c060"
GREEN  = "#3fb950"
RED    = "#f85149"
BLUE   = "#58a6ff"
PURPLE = "#bc8cff"
ORANGE = "#ffa657"

VERDICT_COLOR = {
    "CLEAN":           GREEN,
    "AUTOCORRELATED":  RED,
    "NORMAL":          GREEN,
    "FAT_TAILS":       AMBER,
    "SKEWED":          AMBER,
    "DANGEROUS":       RED,
    "LINEAR":          GREEN,
    "CONVEX":          BLUE,
    "CONCAVE":         RED,
    "SYMMETRIC":       GREEN,
    "CRASH_SENSITIVE": RED,
    "REAL":            GREEN,
    "BORDERLINE":      AMBER,
    "SPURIOUS":        RED,
    "INCONCLUSIVE":    AMBER,
}


# ── Public entry points ───────────────────────────────────────────────────────

def plot_advanced_report(
    tester:                AdvancedTests,
    strategy_name:         str  = "Strategy",
    pair:                  str  = "",
    initial_rolling_window: int = 252,
    n_permutations:        int  = 1000,
    all_strategies:        dict | None = None,
    reload_fn              = None,   # callable(key) -> (AdvancedTests, name, pair)
) -> None:
    """
    Run Tests 1-4 (no placebo) and open the static 2×2 figure + rolling figure.

    Args:
        tester:                 AdvancedTests instance (aligned + core loaded).
        strategy_name:          Label shown in the figure title.
        pair:                   Instrument name shown in the figure title.
        initial_rolling_window: Starting window size shown in the rolling TextBox.
        n_permutations:         Kept for API compatibility; placebo is no longer shown.
        all_strategies:         If provided, enables the strategy selector.
        reload_fn:              callable(key) → (AdvancedTests, name, pair).
    """
    print(f"Running advanced tests for '{strategy_name} / {pair}'...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = tester.run_all(n_permutations=0)   # placebo skipped

    # ── Static figure (2×2) ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    title_obj = fig.suptitle(
        f"Advanced Alpha Tests  |  {strategy_name}  |  {pair}",
        color=TEXT, fontsize=12, fontweight="bold", y=0.985,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        top=0.910, bottom=0.10, left=0.07, right=0.97,
        hspace=0.55, wspace=0.32,
    )

    ax_acf    = fig.add_subplot(gs[0, 0])
    ax_hist   = fig.add_subplot(gs[0, 1])
    ax_nonlin = fig.add_subplot(gs[1, 0])
    ax_asym   = fig.add_subplot(gs[1, 1])

    for ax in [ax_acf, ax_hist, ax_nonlin, ax_asym]:
        _style_ax(ax)

    def _draw_static(rpt, tstr):
        for ax in [ax_acf, ax_hist, ax_nonlin, ax_asym]:
            ax.cla(); _style_ax(ax)
        _plot_acf(ax_acf,       rpt.autocorrelation)
        _plot_hist(ax_hist,     rpt.distribution, tstr._get_stream(), tstr.aligned)
        _plot_nonlin(ax_nonlin, rpt.nonlinear_beta, tstr.aligned)
        _plot_asym(ax_asym,     rpt.asymmetric_beta, tstr.aligned)
        fig.canvas.draw_idle()

    _draw_static(report, tester)
    _add_explain_button(fig)

    # ── Optional strategy selector ────────────────────────────────────────────
    if reload_fn is not None and all_strategies is not None:
        import tkinter as tk
        import tkinter.ttk as ttk

        _strat_keys = sorted(all_strategies.keys())
        _state_s = {"key": strategy_name}
        _last_run_s = {"key": strategy_name}

        fig.text(0.07, 0.064, "Strategy:", color=DIM,
                 fontsize=8.5, ha="left", va="center")

        _strat_lbl = fig.text(
            0.185, 0.064, strategy_name,
            color=TEXT, fontsize=8, ha="left", va="center",
            fontfamily="monospace",
        )
        _status_lbl = fig.text(
            0.55, 0.048, "",
            color=DIM, fontsize=7.5, ha="left", va="center",
            fontfamily="monospace",
        )

        ax_sel2 = fig.add_axes([0.07, 0.028, 0.20, 0.032])
        ax_sel2.set_facecolor(PANEL)
        for sp in ax_sel2.spines.values(): sp.set_edgecolor(BORDER)
        btn_sel2 = Button(ax_sel2, "\u25be  Select Strategy",
                          color=PANEL, hovercolor="#21262d")
        btn_sel2.label.set_color(BLUE); btn_sel2.label.set_fontsize(8.5)

        ax_run2 = fig.add_axes([0.78, 0.028, 0.10, 0.032])
        ax_run2.set_facecolor(PANEL)
        for sp in ax_run2.spines.values(): sp.set_edgecolor(AMBER)
        btn_run2 = Button(ax_run2, "\u25b6  Re-run",
                          color=PANEL, hovercolor="#21262d")
        btn_run2.label.set_color(AMBER); btn_run2.label.set_fontsize(9)

        def _open_sel2(_event):
            popup = tk.Toplevel()
            popup.title("Select Strategy")
            popup.configure(bg="#0d1117"); popup.geometry("420x480")
            tk.Label(popup, text="Select a strategy:", bg="#0d1117",
                     fg="#8b949e", font=("Consolas", 10)).pack(pady=(10, 4))
            fv = tk.StringVar()
            ttk.Entry(popup, textvariable=fv, font=("Consolas", 10)).pack(
                fill=tk.X, padx=10)
            lb = tk.Listbox(popup, bg="#161b22", fg="#e6edf3",
                            selectbackground="#1f6feb", font=("Consolas", 10),
                            relief=tk.FLAT, activestyle="none", height=20)
            sb2 = tk.Scrollbar(popup, command=lb.yview)
            lb.config(yscrollcommand=sb2.set)
            sb2.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4))
            lb.pack(fill=tk.BOTH, expand=True, padx=(10, 0), pady=4)

            def _pop(flt=""):
                lb.delete(0, tk.END)
                for k in _strat_keys:
                    if flt.lower() in k.lower():
                        lb.insert(tk.END, k)
            fv.trace_add("write", lambda *_: _pop(fv.get())); _pop()

            def _confirm():
                sel = lb.curselection()
                if sel:
                    _state_s["key"] = lb.get(sel[0])
                    _strat_lbl.set_text(_state_s["key"])
                    fig.canvas.draw_idle()
                popup.destroy()

            lb.bind("<Double-Button-1>", lambda _: _confirm())
            lb.bind("<Return>", lambda _: _confirm())
            tk.Button(popup, text="Select", command=_confirm,
                      bg="#1f6feb", fg="white", font=("Consolas", 10, "bold"),
                      relief=tk.FLAT).pack(pady=(4, 10))
            popup.lift(); popup.grab_set()

        def _on_rerun2(_event=None):
            key = _state_s["key"]
            if key == _last_run_s["key"]:
                return
            _last_run_s["key"] = key
            _status_lbl.set_text("Running tests…"); _status_lbl.set_color(AMBER)
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            try:
                new_tester, new_name, new_pair = reload_fn(key)
            except Exception as exc:
                _status_lbl.set_text(f"Error: {exc}"); _status_lbl.set_color(RED)
                fig.canvas.draw_idle(); return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_report = new_tester.run_all(n_permutations=0)
            _draw_static(new_report, new_tester)
            title_obj.set_text(
                f"Advanced Alpha Tests  |  {new_name}  |  {new_pair}")
            _status_lbl.set_text(f"Done: {new_name}"); _status_lbl.set_color(GREEN)
            fig.canvas.draw_idle()

        btn_sel2.on_clicked(_open_sel2)
        btn_run2.on_clicked(_on_rerun2)
        fig._static_widgets = (btn_sel2, btn_run2, _state_s, _last_run_s,
                                _strat_lbl, _status_lbl)

    # ── Interactive rolling figure — always opened ─────────────────────────────
    plot_rolling_interactive(tester, strategy_name, pair,
                             initial_window=initial_rolling_window)

    plt.show(block=True)


def plot_rolling_report(
    rolling_df:    pd.DataFrame,
    window:        int,
    strategy_name: str = "Strategy",
    pair:          str = "",
) -> None:
    """
    Open a separate 2×2 figure showing how test metrics evolved over time.

    Args:
        rolling_df:    Output of AdvancedTests.rolling_metrics(window).
        window:        The rolling window used (for the title).
        strategy_name: Label for the title.
        pair:          Instrument label for the title.
    """
    if rolling_df.empty:
        print("[plot_rolling_report] Not enough data to plot rolling metrics.")
        return

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle(
        f"Rolling Metrics  ({window}-day window)  |  {strategy_name}  |  {pair}",
        color=TEXT, fontsize=12, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        top=0.92, bottom=0.08, left=0.07, right=0.97,
        hspace=0.45, wspace=0.30,
    )

    ax_ac  = fig.add_subplot(gs[0, 0])
    ax_sk  = fig.add_subplot(gs[0, 1])
    ax_b2  = fig.add_subplot(gs[1, 0])
    ax_ra  = fig.add_subplot(gs[1, 1])

    for ax in fig.axes:
        _style_ax(ax)

    _plot_rolling_lag1(ax_ac, rolling_df, window)
    _plot_rolling_skew(ax_sk, rolling_df, window)
    _plot_rolling_beta2(ax_b2, rolling_df, window)
    _plot_rolling_asym(ax_ra, rolling_df, window)


def plot_rolling_interactive(
    tester:         AdvancedTests,
    strategy_name:  str = "Strategy",
    pair:           str = "",
    initial_window: int = 252,
) -> None:
    """
    Open an interactive rolling-metrics figure with a live control panel.

    The control row at the bottom has:
      - A status bar showing progress / results
      - A TextBox where the user types the window size (in trading days)
      - A Run button that recomputes and redraws all 6 charts
      - An Explain button (top-right) that opens a scrollable guide window

    Pressing Enter in the TextBox is equivalent to clicking Run.
    The figure auto-runs with `initial_window` on open.
    """
    fig = plt.figure(figsize=(16, 13), facecolor=BG)
    fig.suptitle(
        f"Rolling Metrics — Interactive  |  {strategy_name}  |  {pair}",
        color=TEXT, fontsize=12, fontweight="bold", y=0.98,
    )

    # ── 3×2 rolling metrics grid ─────────────────────────────────────────────
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        top=0.910, bottom=0.16, left=0.07, right=0.97,
        hspace=0.50, wspace=0.30,
    )
    ax_ac  = fig.add_subplot(gs[0, 0])   # lag-1 ACF
    ax_sk  = fig.add_subplot(gs[0, 1])   # skewness
    ax_krt = fig.add_subplot(gs[1, 0])   # excess kurtosis
    ax_b2  = fig.add_subplot(gs[1, 1])   # beta2 (nonlinear)
    ax_sb  = fig.add_subplot(gs[2, 0])   # split betas (up/down)
    ax_ra  = fig.add_subplot(gs[2, 1])   # asymmetry ratio

    for ax in [ax_ac, ax_sk, ax_krt, ax_b2, ax_sb, ax_ra]:
        _style_ax(ax)

    # ── Control row ───────────────────────────────────────────────────────────
    #  [  status bar .............................  ] [ Window (days): ] [ 252 ] [ ▶ Run ]

    ax_status = fig.add_axes([0.07, 0.04, 0.52, 0.055])
    ax_status.set_facecolor(PANEL)
    for sp in ax_status.spines.values():
        sp.set_edgecolor(BORDER)
    ax_status.set_xticks([]); ax_status.set_yticks([])
    status_txt = ax_status.text(
        0.015, 0.5,
        "Enter a window size and click Run  (or press Enter).",
        transform=ax_status.transAxes, va="center", ha="left",
        color=DIM, fontfamily="monospace", fontsize=8.5,
    )

    ax_lbl = fig.add_axes([0.61, 0.04, 0.13, 0.055])
    ax_lbl.set_facecolor(PANEL)
    for sp in ax_lbl.spines.values():
        sp.set_edgecolor(BORDER)
    ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
    ax_lbl.text(0.5, 0.5, "Window (days):",
                transform=ax_lbl.transAxes, va="center", ha="center",
                color=DIM, fontsize=8.5)

    ax_tb = fig.add_axes([0.75, 0.04, 0.08, 0.055])
    ax_tb.set_facecolor(PANEL)
    for sp in ax_tb.spines.values():
        sp.set_edgecolor(AMBER)
    textbox = TextBox(ax_tb, "", initial=str(initial_window),
                      color=PANEL, hovercolor="#21262d", label_pad=0)
    textbox.label.set_color(DIM)
    textbox.text_disp.set_color(TEXT)
    textbox.text_disp.set_fontfamily("monospace")
    textbox.text_disp.set_fontsize(10)

    ax_run = fig.add_axes([0.84, 0.04, 0.12, 0.055])
    ax_run.set_facecolor(PANEL)
    for sp in ax_run.spines.values():
        sp.set_edgecolor(AMBER)
    btn_run = Button(ax_run, "Run", color=PANEL, hovercolor="#21262d")
    btn_run.label.set_color(AMBER)
    btn_run.label.set_fontsize(10)
    btn_run.label.set_fontweight("bold")

    state = {"computing": False}

    def _refresh(rolling_df: pd.DataFrame, window: int) -> None:
        """Clear and redraw all 6 rolling subplots."""
        for ax in [ax_ac, ax_sk, ax_krt, ax_b2, ax_sb, ax_ra]:
            ax.cla()
            _style_ax(ax)
        _plot_rolling_lag1(ax_ac, rolling_df, window)
        _plot_rolling_skew(ax_sk, rolling_df, window)
        _plot_rolling_kurtosis(ax_krt, rolling_df, window)
        _plot_rolling_beta2(ax_b2, rolling_df, window)
        _plot_rolling_split_betas(ax_sb, rolling_df, window)
        _plot_rolling_asym(ax_ra, rolling_df, window)
        fig.canvas.draw_idle()

    def _run(_event=None) -> None:
        if state["computing"]:
            return

        # ── Validate input ─────────────────────────────────────────────────
        try:
            window = int(textbox.text.strip())
        except (ValueError, AttributeError):
            status_txt.set_text("Invalid input — enter an integer, e.g. 252.")
            status_txt.set_color(RED)
            fig.canvas.draw_idle()
            return

        n_data = len(tester.aligned.dropna())
        if window < 30:
            status_txt.set_text("Window too small — minimum is 30 days.")
            status_txt.set_color(RED)
            fig.canvas.draw_idle()
            return
        if window >= n_data:
            status_txt.set_text(
                f"Window ({window}) exceeds data length ({n_data}). "
                "Choose a smaller value."
            )
            status_txt.set_color(RED)
            fig.canvas.draw_idle()
            return

        state["computing"] = True
        btn_run.label.set_color(DIM)

        # ── Compute rolling metrics ────────────────────────────────────────
        status_txt.set_text(f"Computing {window}-day rolling metrics...")
        status_txt.set_color(AMBER)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rolling_df = tester.rolling_metrics(window=window)

        n_points = len(rolling_df)
        _refresh(rolling_df, window)

        status_txt.set_text(
            f"Done.  Window = {window} days  |  "
            f"{n_points} rolling observations  |  "
            f"data from {rolling_df.index[0].strftime('%Y-%m-%d')} "
            f"to {rolling_df.index[-1].strftime('%Y-%m-%d')}."
            if n_points > 0 else "No rolling observations — window may be too large."
        )
        status_txt.set_color(GREEN if n_points > 0 else RED)
        btn_run.label.set_color(AMBER)
        state["computing"] = False
        fig.canvas.draw_idle()

    textbox.on_submit(lambda _: _run())
    btn_run.on_clicked(_run)

    # Keep widgets alive against GC
    fig._textbox = textbox
    fig._btn_run = btn_run

    _add_rolling_explain_button(fig)

    # Auto-run with the initial window
    _run()


# ── Static plot functions ─────────────────────────────────────────────────────

def _plot_acf(ax, result: AutocorrelationResult) -> None:
    """ACF bar chart with ±2/√n significance bands."""
    ax.set_title("Test 1 · Residual Autocorrelation", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax, r"ACF$(\tau) = \mathrm{Corr}(\varepsilon_t,\,\varepsilon_{t-\tau})$"
              r"   |   band $= \pm 2/\sqrt{n}$  (95% CI)")
    ax.set_xlabel("Lag (days)", color=DIM, fontsize=8)
    ax.set_ylabel("Autocorrelation", color=DIM, fontsize=8)

    if result is None or len(result.acf_values) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    band = result.significance_band
    lags = result.lags
    vals = result.acf_values

    colors = [RED if abs(v) > band else DIM for v in vals]
    ax.bar(lags, vals, color=colors, width=0.7, alpha=0.80)
    ax.axhline(+band, color=AMBER, lw=1.0, linestyle="--", alpha=0.8)
    ax.axhline(-band, color=AMBER, lw=1.0, linestyle="--", alpha=0.8)
    ax.axhline(0, color=BORDER, lw=0.8)

    verdict_col = VERDICT_COLOR.get(result.verdict, DIM)
    ax.text(
        0.02, 0.97,
        f"Ljung-Box p={result.ljung_box_pval:.3f}\n"
        f"Sig. lags: {result.n_significant}/{len(lags)}\n"
        f"NW t(α)={result.newey_west_tstat:+.2f}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=7.5, color=TEXT, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                  edgecolor=verdict_col, lw=1),
    )
    ax.text(0.97, 0.97, f"[{result.verdict}]",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, color=verdict_col, fontweight="bold")


def _plot_hist(ax, result: DistributionResult, stream: pd.Series,
               aligned: pd.DataFrame = None) -> None:
    """Residual histogram with fitted normal overlay (active trading days only)."""
    ax.set_title("Test 2 · Return Distribution  (active days)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"Skew $= \mu_3/\sigma^3$   |   "
              r"Kurt $= \mu_4/\sigma^4 - 3$   |   "
              r"zero-return days excluded")
    ax.set_xlabel("Daily return (active days)", color=DIM, fontsize=8)
    ax.set_ylabel("Density", color=DIM, fontsize=8)

    if result is None or stream.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    # Use active trading days only (same filter as run_distribution)
    if aligned is not None:
        active_mask = aligned["strategy"].reindex(stream.index).abs() > 1e-10
        active_stream = stream[active_mask]
        arr = active_stream.values if len(active_stream) >= 30 else stream.values
    else:
        arr = stream.values
    ax.hist(arr, bins=60, density=True, color=BLUE, alpha=0.45,
            edgecolor="none", label="Residuals")

    mu, sigma = arr.mean(), arr.std()
    x_fit     = np.linspace(arr.min(), arr.max(), 300)
    y_fit     = stats.norm.pdf(x_fit, mu, sigma)
    ax.plot(x_fit, y_fit, color=AMBER, lw=1.5, label="Normal fit")

    if not np.isnan(result.pct_05):
        ax.axvline(result.pct_05, color=RED, lw=1.2, linestyle="--", alpha=0.8,
                   label=f"5th pct = {result.pct_05*100:.2f}%")
    if not np.isnan(result.pct_95):
        ax.axvline(result.pct_95, color=GREEN, lw=1.2, linestyle="--", alpha=0.8,
                   label=f"95th pct = {result.pct_95*100:.2f}%")

    ax.legend(fontsize=6.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    verdict_col = VERDICT_COLOR.get(result.verdict, DIM)
    ax.text(
        0.02, 0.97,
        f"Skew   = {result.skewness:+.3f}\n"
        f"Kurt   = {result.excess_kurtosis:+.3f}\n"
        f"SW p   = {result.shapiro_pval:.3f}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=7.5, color=TEXT, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                  edgecolor=verdict_col, lw=1),
    )
    ax.text(0.97, 0.97, f"[{result.verdict}]",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, color=verdict_col, fontweight="bold")


def _plot_qq(ax, stream: pd.Series) -> None:
    """Q-Q plot of residuals vs theoretical normal quantiles."""
    ax.set_title("Test 2 · Q-Q Plot  (Normal)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"Quantiles of residuals vs $\mathcal{N}(0,1)$  — "
              r"dots on line = Gaussian  |  tails off = fat tails")
    ax.set_xlabel("Theoretical quantiles", color=DIM, fontsize=8)
    ax.set_ylabel("Sample quantiles", color=DIM, fontsize=8)

    if stream.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    arr = stream.dropna().values
    (osm, osr), (slope, intercept, _) = stats.probplot(arr)

    ax.scatter(osm, osr, s=3, alpha=0.30, color=BLUE, linewidths=0)
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, color=AMBER, lw=1.5)

    fitted = slope * osm + intercept
    ax.fill_between(osm, osr, fitted,
                    where=(osm > 1.5) | (osm < -1.5),
                    alpha=0.25, color=RED, label="Tail deviation")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)


def _plot_nonlin(ax, result: NonlinearBetaResult, aligned: pd.DataFrame) -> None:
    """Scatter with linear + quadratic regression fits."""
    ax.set_title("Test 3 · Nonlinear Beta", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"$R_{strat} = \alpha + \beta_1 R_a + \beta_2 R_a^2 + \varepsilon$"
              r"   |   $\beta_2 < 0$ = concave (short vol)")
    ax.set_xlabel("Asset return", color=DIM, fontsize=8)
    ax.set_ylabel("Strategy return", color=DIM, fontsize=8)

    data = aligned.dropna()
    if data.empty or result is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    x = data["asset"].values
    y = data["strategy"].values
    ax.scatter(x, y, s=4, alpha=0.25, color=BLUE, linewidths=0)

    x_s = np.linspace(x.min(), x.max(), 300)
    if not np.isnan(result.beta_linear):
        y_lin = result.alpha_nonlin + result.beta_linear * x_s
        ax.plot(x_s, y_lin, color=DIM, lw=1.2, linestyle="--",
                label=f"Linear  β₁={result.beta_linear:+.3f}")

    if not np.isnan(result.beta_quadratic):
        y_q = (result.alpha_nonlin
               + result.beta_linear * x_s
               + result.beta_quadratic * x_s ** 2)
        quad_col = RED if result.verdict == "CONCAVE" else (
            BLUE if result.verdict == "CONVEX" else AMBER
        )
        ax.plot(x_s, y_q, color=quad_col, lw=1.8,
                label=f"Quadratic  β₂={result.beta_quadratic:+.4f}")

    ax.axhline(0, color=BORDER, lw=0.7)
    ax.axvline(0, color=BORDER, lw=0.7)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    verdict_col = VERDICT_COLOR.get(result.verdict, DIM)
    ax.text(
        0.02, 0.97,
        f"β₂={result.beta_quadratic:+.5f}\n"
        f"t(β₂)={result.beta2_tstat:+.2f}\n"
        f"ΔR²={result.r2_gain*100:+.2f}pp",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=7.5, color=TEXT, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                  edgecolor=verdict_col, lw=1),
    )
    ax.text(0.97, 0.97, f"[{result.verdict}]",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, color=verdict_col, fontweight="bold")


def _plot_asym(ax, result: AsymmetricBetaResult, aligned: pd.DataFrame) -> None:
    """Scatter coloured by market direction with separate regression lines."""
    ax.set_title("Test 4 · Asymmetric Beta", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"Blue = up-market ($R_a>0$)   Red = down-market   separate OLS per group")
    ax.set_xlabel("Asset return", color=DIM, fontsize=8)
    ax.set_ylabel("Strategy return", color=DIM, fontsize=8)

    data = aligned.dropna()
    if data.empty or result is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    mask_up   = data["asset"] > 0
    ax.scatter(data["asset"][mask_up],   data["strategy"][mask_up],
               s=4, alpha=0.25, color=BLUE, linewidths=0, label="Up market")
    ax.scatter(data["asset"][~mask_up],  data["strategy"][~mask_up],
               s=4, alpha=0.25, color=RED, linewidths=0, label="Down market")

    def _line(sub, a, b, color):
        if len(sub) < 10 or np.isnan(b):
            return
        xr = np.linspace(sub["asset"].min(), sub["asset"].max(), 200)
        ax.plot(xr, a + b * xr, color=color, lw=1.6)

    if result is not None:
        _line(data[mask_up],  result.alpha_up,   result.beta_up,   BLUE)
        _line(data[~mask_up], result.alpha_down, result.beta_down, RED)

    ax.axhline(0, color=BORDER, lw=0.7)
    ax.axvline(0, color=BORDER, lw=0.7)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    if result is not None:
        verdict_col = VERDICT_COLOR.get(result.verdict, DIM)
        ratio_txt   = (
            f"{result.asymmetry_ratio:.2f}"
            if not np.isnan(result.asymmetry_ratio) else "n/a"
        )
        ax.text(
            0.02, 0.97,
            f"β_up  = {result.beta_up:+.3f}  (n={result.n_up})\n"
            f"β_dn  = {result.beta_down:+.3f}  (n={result.n_down})\n"
            f"ratio = {ratio_txt}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, color=TEXT, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                      edgecolor=verdict_col, lw=1),
        )
        ax.text(0.97, 0.97, f"[{result.verdict}]",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=8, color=verdict_col, fontweight="bold")


def _plot_betas(ax, result: AsymmetricBetaResult, core) -> None:
    """Bar chart comparing β_up, β_overall, β_down."""
    ax.set_title("Test 4 · Beta Comparison", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"$\beta_{up}$ vs $\beta_{overall}$ vs $\beta_{down}$  — "
              r"$|\beta_{down}| \gg |\beta_{up}|$ = crash risk")
    ax.set_ylabel("β value", color=DIM, fontsize=8)

    if result is None or core is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    labels = ["β_up", "β_overall", "β_down"]
    values = [result.beta_up, core.beta, result.beta_down]
    colors = [BLUE, AMBER, RED]

    vl, vv, vc = [], [], []
    for l, v, c in zip(labels, values, colors):
        if not np.isnan(v):
            vl.append(l); vv.append(v); vc.append(c)

    if not vv:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    x    = np.arange(len(vl))
    bars = ax.bar(x, vv, color=vc, alpha=0.80, width=0.5)

    for bar, val in zip(bars, vv):
        yp = bar.get_height()
        va = "bottom" if yp >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, yp,
                f"{val:+.3f}", ha="center", va=va, fontsize=8, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(vl, color=TEXT, fontsize=9)
    ax.axhline(0, color=BORDER, lw=0.8)

    verdict_col = VERDICT_COLOR.get(result.verdict, DIM)
    ax.text(0.97, 0.97, f"[{result.verdict}]",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, color=verdict_col, fontweight="bold")


def _plot_placebo(ax, result: PlaceboResult, variant: str) -> None:
    """
    Null distribution histogram from the permutation test.

    variant: "individual" or "block"
    """
    is_block = variant == "block"
    label    = "Block shuffle" if is_block else "Individual shuffle"
    ax.set_title(f"Test 6 · Placebo  [{label}]", color=TEXT, fontsize=9, pad=18)

    if is_block:
        _subtitle(ax,
                  r"Blocks $\approx$ 20 days shuffled (preserves serial structure)  "
                  r"|  conservative null")
    else:
        _subtitle(ax,
                  r"Completely random permutation of strategy returns  "
                  r"|  liberal null")

    ax.set_xlabel("Null alpha (shuffled)", color=DIM, fontsize=8)
    ax.set_ylabel("Count", color=DIM, fontsize=8)

    if result is None or len(result.null_alphas_individual) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    null   = result.null_alphas_block if is_block else result.null_alphas_individual
    pval   = result.pval_block        if is_block else result.pval_individual
    obs    = result.observed_alpha

    # Histogram of null distribution
    ax.hist(null, bins=50, color=DIM, alpha=0.60, edgecolor="none")

    # Observed alpha vertical line
    verdict_col = VERDICT_COLOR.get(result.verdict, DIM)
    ax.axvline(obs, color=verdict_col, lw=2.0,
               label=f"Observed α = {obs*100:+.4f}%")

    # Shade the rejection region (null alphas ≥ observed)
    reject_mask = null >= obs
    if reject_mask.any():
        reject_heights, reject_edges = np.histogram(
            null[reject_mask], bins=30, range=(null.min(), null.max())
        )
        ax.bar(
            reject_edges[:-1], reject_heights,
            width=np.diff(reject_edges),
            color=RED, alpha=0.40, align="edge",
        )

    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    ax.text(
        0.02, 0.97,
        f"p-value  = {pval:.3f}\n"
        f"n perm   = {result.n_permutations:,}\n"
        f"block sz = {result.block_size if is_block else 'n/a'}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=7.5, color=TEXT, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                  edgecolor=verdict_col, lw=1),
    )
    ax.text(0.97, 0.97, f"[{result.verdict}]",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, color=verdict_col, fontweight="bold")


# ── Rolling plot functions ────────────────────────────────────────────────────

def _plot_rolling_lag1(ax, df: pd.DataFrame, window: int) -> None:
    ax.set_title(f"Rolling Lag-1 Autocorrelation  ({window}d)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax, r"AC(1) of alpha stream on trailing window  |  red = suspicious (|AC| > 0.10)")
    ax.set_ylabel("AC(1)", color=DIM, fontsize=8)

    col = "lag1_acf"
    if col not in df.columns or df[col].dropna().empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    vals  = df[col]
    dates = df.index
    ax.set_ylim(-0.25, 0.25)
    ax.plot(dates, vals, color=AMBER, lw=1.2)
    ax.fill_between(dates, 0, vals,
                    where=(vals.abs() > 0.10), alpha=0.30, color=RED)
    ax.fill_between(dates, 0, vals,
                    where=(vals.abs() <= 0.10), alpha=0.15, color=GREEN)
    ax.axhline(0,     color=BORDER, lw=0.8, linestyle="--")
    ax.axhline(+0.10, color=RED,    lw=0.7, linestyle=":", alpha=0.7)
    ax.axhline(-0.10, color=RED,    lw=0.7, linestyle=":", alpha=0.7)
    _fmt_date_axis(ax, dates)


def _plot_rolling_skew(ax, df: pd.DataFrame, window: int) -> None:
    ax.set_title(f"Rolling Skewness  ({window}d)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"Skewness of alpha stream on trailing window  |  "
              r"red = left-skewed (<−1)  |  amber = right-skewed (>1)")
    ax.set_ylabel("Skewness", color=DIM, fontsize=8)

    col = "skewness"
    if col not in df.columns or df[col].dropna().empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    vals  = df[col]
    dates = df.index
    # Cap at ±3 so one extreme window doesn't flatten the rest
    ax.set_ylim(-3.0, 3.0)
    vals_clipped = vals.clip(-3.0, 3.0)
    ax.plot(dates, vals_clipped, color=AMBER, lw=1.2)
    ax.fill_between(dates, 0, vals_clipped,
                    where=(vals < -1.0), alpha=0.30, color=RED,
                    label="Neg. skew < −1 (left tail)")
    ax.fill_between(dates, 0, vals_clipped,
                    where=((vals >= -1.0) & (vals <= 1.0)), alpha=0.12, color=GREEN)
    ax.fill_between(dates, 0, vals_clipped,
                    where=(vals > 1.0), alpha=0.20, color=AMBER)
    ax.axhline(0,    color=BORDER, lw=0.8, linestyle="--")
    ax.axhline(-1.0, color=RED,   lw=0.7, linestyle=":", alpha=0.7)
    ax.axhline(+1.0, color=AMBER, lw=0.7, linestyle=":", alpha=0.7)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    _fmt_date_axis(ax, dates)


def _plot_rolling_kurtosis(ax, df: pd.DataFrame, window: int) -> None:
    ax.set_title(f"Rolling Excess Kurtosis  ({window}d)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"$\mu_4/\sigma^4 - 3$ on trailing window  |  "
              r"red > 3 = fat tails  |  ideal near 0 (Gaussian)")
    ax.set_ylabel("Excess Kurtosis", color=DIM, fontsize=8)

    col = "excess_kurtosis"
    if col not in df.columns or df[col].dropna().empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    vals  = df[col]
    dates = df.index
    # Cap at ±6: fat tails at >3, but no need to show values of 100+
    ax.set_ylim(-6.0, 6.0)
    vals_clipped = vals.clip(-6.0, 6.0)
    ax.plot(dates, vals_clipped, color=AMBER, lw=1.2)
    ax.fill_between(dates, 0, vals_clipped,
                    where=(vals > 3.0), alpha=0.30, color=RED,
                    label="Fat tails (kurt > 3)")
    ax.fill_between(dates, 0, vals_clipped,
                    where=((vals >= 0) & (vals <= 3.0)), alpha=0.12, color=GREEN)
    ax.fill_between(dates, 0, vals_clipped,
                    where=(vals < 0), alpha=0.12, color=BLUE,
                    label="Thin tails (kurt < 0)")
    ax.axhline(0,   color=BORDER, lw=0.8, linestyle="--")
    ax.axhline(3.0, color=RED,    lw=0.7, linestyle=":", alpha=0.7)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    _fmt_date_axis(ax, dates)


def _plot_rolling_split_betas(ax, df: pd.DataFrame, window: int) -> None:
    ax.set_title(f"Rolling β_up / β_down  ({window}d)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"OLS β on up-market days (blue) vs down-market days (red) — trailing window")
    ax.set_ylabel("β", color=DIM, fontsize=8)

    col_up = "beta_up"
    col_dn = "beta_down"
    if col_up not in df.columns or df[col_up].dropna().empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    dates   = df.index
    beta_up = df[col_up]
    beta_dn = df[col_dn] if col_dn in df.columns else pd.Series(np.nan, index=dates)

    ax.plot(dates, beta_up, color=BLUE, lw=1.2, alpha=0.85, label="β_up (up-market)")
    ax.plot(dates, beta_dn, color=RED,  lw=1.2, alpha=0.85, label="β_down (down-market)")
    ax.fill_between(dates, beta_up, beta_dn,
                    where=(beta_dn.abs() > beta_up.abs()),
                    alpha=0.18, color=RED, label="Crash-bias region")
    ax.axhline(0, color=BORDER, lw=0.8, linestyle="--")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    _fmt_date_axis(ax, dates)


def _plot_rolling_beta2(ax, df: pd.DataFrame, window: int) -> None:
    ax.set_title(f"Rolling β₂ (Nonlinear)  ({window}d)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"Quadratic coefficient on trailing window  |  "
              r"red = concave ($\beta_2 < 0$)  |  blue = convex ($\beta_2 > 0$)")
    ax.set_ylabel("β₂", color=DIM, fontsize=8)

    col = "beta2"
    if col not in df.columns or df[col].dropna().empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    vals  = df[col]
    dates = df.index
    ax.plot(dates, vals, color=AMBER, lw=1.2)
    ax.fill_between(dates, 0, vals, where=(vals < 0), alpha=0.30, color=RED,
                    label="Concave (β₂<0)")
    ax.fill_between(dates, 0, vals, where=(vals >= 0), alpha=0.25, color=BLUE,
                    label="Convex (β₂>0)")
    ax.axhline(0, color=BORDER, lw=0.8, linestyle="--")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    _fmt_date_axis(ax, dates)


def _plot_rolling_asym(ax, df: pd.DataFrame, window: int) -> None:
    ax.set_title(f"Rolling Asymmetry Ratio  ({window}d)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax,
              r"$|\beta_{down}| / |\beta_{up}|$ on trailing window  |  "
              r"red > 1.5 = crash sensitive")
    ax.set_ylabel(r"$|\beta_{down}| / |\beta_{up}|$", color=DIM, fontsize=8)

    col = "asymmetry_ratio"
    if col not in df.columns or df[col].dropna().empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    vals  = df[col]
    dates = df.index

    # Hard cap at [0, 3]: 1 = symmetric, 1.5 = threshold, >3 is extreme
    # anything beyond 3 is already clearly crash-sensitive
    cap = 3.0
    ax.set_ylim(0.0, cap)
    vals_clipped = vals.replace([np.inf, -np.inf], np.nan).clip(upper=cap)

    ax.plot(dates, vals_clipped, color=AMBER, lw=1.2)
    ax.fill_between(dates, 1.0, vals_clipped,
                    where=(vals > 1.5), alpha=0.30, color=RED,
                    label="Crash-sensitive (>1.5)")
    ax.fill_between(dates, 1.0, vals_clipped,
                    where=((vals > 0) & (vals <= 1.5)), alpha=0.12, color=GREEN)
    ax.axhline(1.0, color=BORDER, lw=0.8, linestyle="--", label="ratio = 1 (symmetric)")
    ax.axhline(1.5, color=RED,   lw=0.7, linestyle=":", alpha=0.7)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    _fmt_date_axis(ax, dates)


# ── Explain window ────────────────────────────────────────────────────────────

_ADVANCED_GUIDES = [
    {
        "number": "01",
        "title":  "Residual Autocorrelation  (Test 1)",
        "position": "top-left",
        "what": (
            "Each bar shows how correlated today's alpha-stream return is with "
            "the return at that lag. Red bars exceed the 95% white-noise band — "
            "those lags are statistically significant. Ljung-Box tests them jointly. "
            "NW t(α) re-estimates alpha significance after correcting for autocorrelation."
        ),
        "formula": "ACF(τ)  =  Corr(ε_t, ε_{t−τ})   for τ = 1 … 20",
        "variables": [
            ("Band (amber)",  "±2/√n — 95% CI for white noise."),
            ("Red bar",       "ACF at that lag exceeds the band — statistically significant."),
            ("Ljung-Box p",   "Global test: p < 0.05 = autocorrelation is present."),
            ("NW t(α)",       "Alpha t-stat corrected for autocorrelation (Newey-West HAC)."),
        ],
        "want": [
            "All bars inside the amber bands  (white-noise residuals)",
            "Ljung-Box p > 0.05",
            "NW t(α) close to the naive t(α)",
            "Verdict: CLEAN",
        ],
        "avoid": [
            "Several red bars  (systematic structure in residuals)",
            "Ljung-Box p < 0.05",
            "NW t(α) << naive t(α)  (significance was inflated by autocorrelation)",
            "Large lag-1 bar  (momentum / mean-reversion in the alpha stream itself)",
        ],
    },
    {
        "number": "02",
        "title":  "Return Distribution  (Test 2a — Histogram)",
        "position": "top-right",
        "what": (
            "Shape of the alpha stream's daily return distribution. "
            "The amber line is the best-fit Gaussian. "
            "Negative skew + high kurtosis = short-volatility profile in disguise — "
            "small steady gains but extreme losses during market stress."
        ),
        "formula": (
            "Skew = μ₃/σ³   (0 = symmetric)\n"
            "                 Kurt = μ₄/σ⁴ − 3  (0 = normal,  >0 = fatter tails)"
        ),
        "variables": [
            ("Skew",       "< 0 = left tail.  > 0 = right tail."),
            ("Kurt",       "Excess kurtosis. 0 = Gaussian.  > 3 = fat tails."),
            ("SW p-value", "Shapiro-Wilk: p < 0.05 rejects normality."),
            ("5th pct",    "Worst 5% of days."),
        ],
        "want": [
            "Histogram matches the amber normal overlay",
            "Skew between −0.5 and +0.5",
            "Excess kurtosis < 2.0",
            "Verdict: NORMAL",
        ],
        "avoid": [
            "Skew < −1.0 AND kurtosis > 3.0  → DANGEROUS (short-vol profile)",
            "Very fat tails far beyond the normal overlay",
            "Shapiro-Wilk p < 0.01",
        ],
    },
    {
        "number": "04",
        "title":  "Nonlinear Beta  (Test 3)",
        "position": "middle-right",
        "what": (
            "Adds squared market return R_asset² as a regressor. "
            "Detects whether the payoff is CURVED with respect to the market. "
            "β₂ < 0 = concave = short gamma = vulnerable to large moves in either direction."
        ),
        "formula": "R_strategy  =  α  +  β₁·R_asset  +  β₂·R_asset²  +  ε",
        "variables": [
            ("Dashed line",   "Linear fit (simple one-factor model)."),
            ("Coloured line", "Quadratic fit — shows the curvature."),
            ("β₂",           "Curvature.  < 0 = concave.  > 0 = convex."),
            ("ΔR²",          "Extra variance explained by β₂ (significant if > 2pp)."),
        ],
        "want": [
            "β₂ close to 0 and t(β₂) < 1.5  (no curvature)",
            "Linear and quadratic lines nearly identical",
            "Verdict: LINEAR  (or CONVEX is also acceptable)",
        ],
        "avoid": [
            "β₂ significantly negative  (t < −2)  →  CONCAVE = short vol in disguise",
            "Quadratic line bowed downward",
            "ΔR² > 5pp  (curvature is material)",
        ],
    },
    {
        "number": "05",
        "title":  "Asymmetric Beta — Scatter  (Test 4a)",
        "position": "bottom-left",
        "what": (
            "Splits days into up-market (blue) and down-market (red). "
            "Runs separate OLS on each group. "
            "If the red slope is steeper, the strategy absorbs more downside than upside — "
            "classic crash sensitivity seen in carry and short-premium strategies."
        ),
        "formula": (
            "β_up  = OLS slope ONLY on days where R_asset > 0\n"
            "                 β_down = OLS slope ONLY on days where R_asset ≤ 0"
        ),
        "variables": [
            ("Blue dots",  "Up-market days."),
            ("Red dots",   "Down-market days."),
            ("β_up",       "Market sensitivity on rising days."),
            ("β_down",     "Market sensitivity on falling days."),
            ("Ratio",      "|β_down| / |β_up|.  > 1.5 = crash-sensitive."),
        ],
        "want": [
            "Both slopes similar in magnitude  (symmetric)",
            "Ratio < 1.2",
            "Verdict: SYMMETRIC",
        ],
        "avoid": [
            "Red slope much steeper than blue",
            "Ratio > 1.5  →  CRASH_SENSITIVE",
            "α_down significantly negative while α_up is positive",
        ],
    },
    {
        "number": "06",
        "title":  "Beta Comparison  (Test 4b)",
        "position": "bottom-right (static figure, row 2)",
        "what": (
            "Bar chart comparing β_up (blue), β_overall (amber), and β_down (red). "
            "Ideal: all bars near zero and similar height. "
            "If |β_down| >> |β_up|, the strategy is crash-sensitive."
        ),
        "formula": "β_overall = Cov(R_strat, R_asset) / Var(R_asset)  on all days",
        "variables": [
            ("β_up (blue)",      "Beta on rising-market days."),
            ("β_overall (amber)", "Beta from the full-period core regression."),
            ("β_down (red)",     "Beta on falling-market days."),
        ],
        "want": [
            "All three bars near zero and similar height",
            "|β_down| ≈ |β_up|  (asymmetry ratio ≈ 1.0)",
        ],
        "avoid": [
            "|β_down| >> |β_up|  (amplifies losses in crashes)",
            "β_overall low while β_down is large  (overall masks crash risk)",
        ],
    },
    {
        "number": "07",
        "title":  "Placebo — Individual Shuffle  (Test 6a)",
        "position": "bottom-left (static figure, row 3)",
        "what": (
            "Randomly shuffles the strategy's daily returns 1000 times — "
            "each shuffle destroys any real relationship with the market. "
            "For each shuffle, OLS alpha is re-estimated. The histogram shows "
            "what alpha looks like under the null hypothesis of no skill. "
            "The coloured vertical line is the actual observed alpha. "
            "Red shading = the fraction of null alphas ≥ observed (empirical p-value)."
        ),
        "formula": (
            "p_individual  =  #{shuffled alphas ≥ α_observed}  /  N_permutations\n"
            "                 Individual: daily returns shuffled completely at random"
        ),
        "variables": [
            ("Histogram",    "1000 alphas from randomly shuffled strategy returns."),
            ("Vertical line","Actual observed alpha."),
            ("Red shading",  "Fraction of null alphas that are at least as large (p-value)."),
            ("p-value",      "Probability of seeing this alpha by luck alone."),
        ],
        "want": [
            "Observed alpha (vertical line) far to the right of the null distribution",
            "p-value < 0.05  (alpha is statistically significant)",
            "Verdict: REAL",
        ],
        "avoid": [
            "Observed alpha buried inside the null distribution",
            "p-value > 0.15  (alpha could easily have appeared by chance)",
            "Verdict: SPURIOUS",
        ],
    },
    {
        "number": "08",
        "title":  "Placebo — Block Shuffle  (Test 6b)",
        "position": "bottom-right (static figure, row 3)",
        "what": (
            "Same as the individual placebo but shuffles in ~20-day contiguous blocks. "
            "This preserves short-range autocorrelation within each block — "
            "giving a harder, more realistic null hypothesis. "
            "A strategy that passes the block test but not the individual test "
            "has autocorrelation in its returns that may be inflating the naive p-value. "
            "The block p-value is the primary result to use for inference."
        ),
        "formula": (
            "p_block  =  #{block-shuffled alphas ≥ α_observed}  /  N_permutations\n"
            "                 Blocks of ~20 days shuffled — serial structure preserved within blocks"
        ),
        "variables": [
            ("Block size",      "~20 days — each block keeps its internal return structure."),
            ("p_block",         "More conservative p-value — harder to reject the null."),
            ("Gap p_ind−p_blk", "Large gap → returns are autocorrelated (matters for inference)."),
        ],
        "want": [
            "Block p-value < 0.05  (alpha survives the conservative test)",
            "Block p ≈ individual p  (no meaningful autocorrelation in the strategy)",
            "Verdict: REAL",
        ],
        "avoid": [
            "Block p > 0.10 while individual p < 0.05  (autocorrelation is inflating the naive result)",
            "Both p-values > 0.15  (alpha is not statistically distinguishable from noise)",
            "Verdict: SPURIOUS — rethink the strategy entirely",
        ],
    },
]


def _open_advanced_explain_window() -> None:
    """Open a separate scrollable Tk window with advanced test explanations."""
    import tkinter as tk

    win = tk.Toplevel()
    win.title("Advanced Alpha Tests — How to Read the Plots")
    win.configure(bg="#0d1117")
    win.geometry("860x720")
    win.resizable(True, True)

    frame = tk.Frame(win, bg="#0d1117")
    frame.pack(fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame, bg="#21262d", troughcolor="#0d1117",
                             activebackground="#30363d")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    txt = tk.Text(
        frame,
        bg="#0d1117", fg="#e6edf3",
        font=("Consolas", 11),
        yscrollcommand=scrollbar.set,
        wrap=tk.WORD, padx=28, pady=20,
        borderwidth=0, highlightthickness=0,
        cursor="arrow", state=tk.NORMAL,
    )
    txt.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=txt.yview)

    txt.tag_configure("header",
        font=("Consolas", 14, "bold"), foreground="#f0c060",
        spacing1=18, spacing3=6)
    txt.tag_configure("header_sub",
        font=("Consolas", 9), foreground="#8b949e", spacing3=10)
    txt.tag_configure("section",
        font=("Consolas", 10, "bold"), foreground="#8b949e",
        spacing1=10, spacing3=2)
    txt.tag_configure("what",
        font=("Consolas", 10), foreground="#c9d1d9",
        spacing3=6, lmargin1=16, lmargin2=16)
    txt.tag_configure("formula_box",
        font=("Consolas", 11, "bold"), foreground="#58a6ff",
        background="#161b22",
        spacing1=6, spacing3=6, lmargin1=16, lmargin2=16)
    txt.tag_configure("var_name",
        font=("Consolas", 10, "bold"), foreground="#bc8cff")
    txt.tag_configure("var_desc",
        font=("Consolas", 10), foreground="#8b949e")
    txt.tag_configure("want_header",
        font=("Consolas", 10, "bold"), foreground="#3fb950",
        spacing1=8, spacing3=2)
    txt.tag_configure("want_item",
        font=("Consolas", 10), foreground="#3fb950",
        lmargin1=28, lmargin2=38, spacing3=1)
    txt.tag_configure("avoid_header",
        font=("Consolas", 10, "bold"), foreground="#f85149",
        spacing1=8, spacing3=2)
    txt.tag_configure("avoid_item",
        font=("Consolas", 10), foreground="#f85149",
        lmargin1=28, lmargin2=38, spacing3=1)
    txt.tag_configure("divider",
        font=("Consolas", 7), foreground="#21262d",
        spacing1=14, spacing3=14)

    txt.insert(tk.END, "Advanced Alpha Validity Tests — How to Read the Plots\n", "header")
    txt.insert(tk.END,
        "Tests 1, 2, 3, 4 (static figure) + Test 6 (placebo/permutation)\n"
        "Rolling figure: shows how each metric evolved over time.\n",
        "header_sub")
    txt.insert(tk.END, "─" * 80 + "\n", "divider")

    for guide in _ADVANCED_GUIDES:
        txt.insert(tk.END,
            f"  PLOT {guide['number']}  ·  {guide['title']}"
            f"              [{guide['position']}]\n", "header")
        txt.insert(tk.END, "What it shows\n", "section")
        txt.insert(tk.END, guide["what"] + "\n", "what")
        txt.insert(tk.END, "Formula\n", "section")
        txt.insert(tk.END, "  " + guide["formula"] + "\n", "formula_box")
        txt.insert(tk.END, "Variables\n", "section")
        for var, desc in guide["variables"]:
            txt.insert(tk.END, f"  {var:<18}", "var_name")
            txt.insert(tk.END, f"  {desc}\n", "var_desc")
        txt.insert(tk.END, "What you WANT to see\n", "want_header")
        for item in guide["want"]:
            txt.insert(tk.END, f"  •  {item}\n", "want_item")
        txt.insert(tk.END, "What to AVOID\n", "avoid_header")
        for item in guide["avoid"]:
            txt.insert(tk.END, f"  •  {item}\n", "avoid_item")
        txt.insert(tk.END, "─" * 80 + "\n", "divider")

    txt.config(state=tk.DISABLED)
    txt.see("1.0")


def _add_explain_button(fig: plt.Figure) -> None:
    btn_ax = fig.add_axes([0.865, 0.970, 0.10, 0.022])
    btn_ax.set_facecolor(PANEL)
    for sp in btn_ax.spines.values():
        sp.set_edgecolor(BORDER)
    btn = Button(btn_ax, "?  Explain", color=PANEL, hovercolor="#21262d")
    btn.label.set_color(DIM)
    btn.label.set_fontsize(8.5)

    def _open(_event):
        btn.label.set_color(AMBER)
        fig.canvas.draw_idle()
        _open_advanced_explain_window()
        btn.label.set_color(DIM)
        fig.canvas.draw_idle()

    btn.on_clicked(_open)
    fig._explain_btn = btn


_PERIOD_COLORS = [BLUE, GREEN, AMBER, PURPLE, ORANGE, RED,
                  "#79c0ff", "#56d364", "#e3b341", "#d2a8ff", "#ffa198"]


def _plot_rolling_placebo(
    ax_hist, ax_bar, periods: list | None, window: int
) -> None:
    """
    Two-panel dedicated placebo figure.

    ax_hist — overlapping null-distribution histograms, one colour per period.
              Vertical line = observed alpha.  Triangle marker = p-value colour.
    ax_bar  — bar chart of observed alpha per period, coloured by p-value.
              Dashed horizontal = overall observed alpha (full-period placebo).
    """
    # ── Top: overlapping histograms ───────────────────────────────────────────
    n_perms = len(periods[0]["null_alphas"]) if periods else 0
    n_periods = len(periods) if periods else 0
    ax_hist.set_title(
        f"Null distributions — {window}-day periods  "
        f"({n_periods} periods  \u00d7  {n_perms:,} permutations)",
        color=TEXT, fontsize=10, pad=4,
    )
    ax_hist.text(
        0.5, -0.08,
        r"Each colour = one period  |  histogram = shuffled $\alpha$  |  "
        r"vertical line = observed $\alpha$  |  $\blacktriangledown$ = significance",
        transform=ax_hist.transAxes, ha="center", va="top",
        fontsize=7.5, color=DIM, style="italic",
    )
    ax_hist.set_xlabel("Alpha per day (%)", color=DIM, fontsize=9)
    ax_hist.set_ylabel("Density", color=DIM, fontsize=9)

    # ── Bottom: observed alpha bar chart ──────────────────────────────────────
    ax_bar.set_title("Observed alpha per period", color=TEXT, fontsize=10, pad=14)
    ax_bar.set_ylabel("Alpha per day (%)", color=DIM, fontsize=9)

    if not periods:
        for ax in [ax_hist, ax_bar]:
            ax.text(0.5, 0.5,
                    "No data — window may be larger than the strategy's active history",
                    transform=ax.transAxes, ha="center", va="center", color=DIM)
        return

    bar_labels, bar_vals, bar_cols = [], [], []

    for i, period in enumerate(periods):
        color      = _PERIOD_COLORS[i % len(_PERIOD_COLORS)]
        null       = period["null_alphas"] * 100
        obs        = period["observed_alpha"] * 100
        pval       = period["pval"]
        pval_col   = GREEN if pval < 0.05 else (AMBER if pval < 0.15 else RED)
        date_label = (f"{period['start'].strftime('%Y-%m')}\n"
                      f"{period['end'].strftime('%Y-%m')}")

        # Histogram
        ax_hist.hist(null, bins=40, density=True,
                     color=color, alpha=0.25, edgecolor="none")
        ax_hist.axvline(obs, color=color, lw=2.0, alpha=0.92,
                        label=f"{period['start'].strftime('%b %Y')} – "
                              f"{period['end'].strftime('%b %Y')}   "
                              f"α={obs:+.4f}%   p={pval:.2f}")

        bar_labels.append(date_label)
        bar_vals.append(obs)
        bar_cols.append(pval_col)

    # Cap y-axis: base it only on non-degenerate distributions.
    # Periods with very few trades produce null alphas ≈ 0 (shuffling zeros
    # gives zeros), creating a single massive density spike.  Exclude those
    # from the cap calculation so the real distributions are legible.
    max_heights = []
    for period in periods:
        null = period["null_alphas"] * 100
        if np.std(null) < 1e-4:          # degenerate — skip for cap
            continue
        h, _ = np.histogram(null, bins=40, density=True)
        max_heights.append(float(np.max(h)))
    if max_heights:
        y_cap = float(np.median(max_heights)) * 1.4
        ax_hist.set_ylim(0, y_cap)

    # Place triangle markers after ylim is fixed
    ymax = ax_hist.get_ylim()[1]
    for i, period in enumerate(periods):
        color    = _PERIOD_COLORS[i % len(_PERIOD_COLORS)]
        pval     = period["pval"]
        obs      = period["observed_alpha"] * 100
        pval_col = GREEN if pval < 0.05 else (AMBER if pval < 0.15 else RED)
        ax_hist.plot(obs, ymax * 0.97, marker="v", color=pval_col,
                     markersize=7, zorder=7, clip_on=False)

    ax_hist.axvline(0, color=BORDER, lw=1.0, linestyle="--", alpha=0.5)
    ax_hist.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER,
                   labelcolor=TEXT, loc="upper right",
                   ncol=max(1, len(periods) // 6))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    x = np.arange(len(bar_labels))
    bars = ax_bar.bar(x, bar_vals, color=bar_cols, alpha=0.80, width=0.6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels, color=DIM, fontsize=6.5)
    ax_bar.axhline(0, color=BORDER, lw=0.8, linestyle="--")

    for bar, val in zip(bars, bar_vals):
        ypos = bar.get_height()
        va   = "bottom" if ypos >= 0 else "top"
        ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.4f}%", ha="center", va=va,
                    fontsize=6.5, color=TEXT)

    # Legend for p-value colour coding
    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[
            Patch(facecolor=GREEN, label="p < 0.05  (significant)"),
            Patch(facecolor=AMBER, label="p < 0.15  (borderline)"),
            Patch(facecolor=RED,   label="p ≥ 0.15  (spurious)"),
        ],
        fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
        loc="upper right",
    )


# ── Rolling explain window ────────────────────────────────────────────────────

_ROLLING_GUIDES = [
    {
        "number": "R1",
        "title":  "Rolling Lag-1 Autocorrelation  (Test 1)",
        "what": (
            "Shows how the lag-1 autocorrelation of the alpha stream evolves over time. "
            "Each point is computed on a trailing window of the chosen size. "
            "Positive values mean today's return predicts tomorrow's direction — momentum. "
            "Negative values mean returns mean-revert — yesterday's gain predicts today's loss. "
            "Either extreme can be exploitable but also inflates naive performance metrics."
        ),
        "formula": "AC(1)_t  =  Corr(ε_{t−k}, ε_{t−k+1})  for k in the trailing window",
        "variables": [
            ("Amber line",  "Rolling lag-1 autocorrelation."),
            ("Red fill",    "|AC(1)| > 0.10 — suspicious level."),
            ("Green fill",  "|AC(1)| ≤ 0.10 — acceptable range."),
            ("±0.10 band",  "Rough threshold for a practically meaningful autocorrelation."),
        ],
        "want": [
            "Line near zero throughout the history",
            "Mostly green fill — consistently within the ±0.10 band",
            "No sustained positive or negative bias",
        ],
        "avoid": [
            "Prolonged positive streak (momentum in residuals — NW t-stat will be lower)",
            "Prolonged negative streak (mean-reversion — shuffled alpha may look better)",
            "Sharp regime changes coinciding with strategy modifications",
        ],
    },
    {
        "number": "R2",
        "title":  "Rolling Skewness  (Test 2a)",
        "what": (
            "Third standardised moment of the alpha stream on a trailing window. "
            "Negative skew = a long left tail = occasional large losses. "
            "This is the distribution fingerprint of short-premium strategies that collect "
            "small credits and occasionally blow up. "
            "Skewness can shift over market regimes — calm periods look fine; "
            "stress periods reveal the true shape."
        ),
        "formula": "Skew_t  =  E[(ε − μ)³] / σ³   on the trailing window",
        "variables": [
            ("Amber line",    "Rolling skewness."),
            ("Red fill",      "Skew < −1.0 — dangerous left tail."),
            ("Green fill",    "−1 ≤ Skew ≤ +1 — acceptable range."),
            ("Amber fill",    "Skew > +1.0 — right-skewed (usually good)."),
        ],
        "want": [
            "Skewness consistently between −0.5 and +0.5",
            "No persistent excursions below −1.0",
            "Stable across market regimes",
        ],
        "avoid": [
            "Persistent negative skew below −1.0  (hidden left tail risk)",
            "Skew dropping sharply during market stress periods",
            "Large negative skew + high kurtosis simultaneously  (DANGEROUS profile)",
        ],
    },
    {
        "number": "R3",
        "title":  "Rolling Excess Kurtosis  (Test 2b)",
        "what": (
            "Fourth standardised moment minus 3. "
            "Zero = Gaussian tails. Positive = fatter tails than normal = more extreme events. "
            "Fat tails combined with negative skew is the classic short-volatility signature. "
            "High kurtosis means your Value-at-Risk models will systematically under-estimate "
            "the true loss potential."
        ),
        "formula": "Kurt_t  =  E[(ε − μ)⁴] / σ⁴ − 3   on the trailing window",
        "variables": [
            ("Amber line",  "Rolling excess kurtosis."),
            ("Red fill",    "Kurt > 3 — significantly fatter tails than Gaussian."),
            ("Green fill",  "0 ≤ Kurt ≤ 3 — mild fat tails, acceptable."),
            ("Blue fill",   "Kurt < 0 — thinner tails than Gaussian (platykurtic)."),
            ("3.0 line",    "Threshold above which tail risk is practically significant."),
        ],
        "want": [
            "Excess kurtosis below 3.0 throughout history",
            "Stable, not spiking during market stress events",
            "Verdict: NORMAL or FAT_TAILS at worst",
        ],
        "avoid": [
            "Kurtosis > 5 persistently  (extreme tail risk)",
            "Kurtosis spikes during stress + negative skew simultaneously",
            "Kurtosis rising over time  (risk profile worsening)",
        ],
    },
    {
        "number": "R4",
        "title":  "Rolling β₂ — Nonlinear Beta  (Test 3)",
        "what": (
            "The quadratic coefficient from regressing strategy returns on both the market "
            "return and its square on a trailing window. "
            "β₂ < 0 means the payoff is concave: the strategy loses more in big market moves "
            "in either direction. This is the signature of selling optionality — "
            "collecting premium but being short gamma. "
            "Watch whether this property is stable or varies across regimes."
        ),
        "formula": "R_strat = α + β₁·R_asset + β₂·R_asset²  →  β₂ shown here",
        "variables": [
            ("Amber line",  "Rolling β₂ coefficient."),
            ("Red fill",    "β₂ < 0 — concave payoff (short gamma)."),
            ("Blue fill",   "β₂ > 0 — convex payoff (long gamma, good)."),
            ("0 line",      "Linear payoff boundary."),
        ],
        "want": [
            "β₂ near zero  (linear payoff profile)",
            "β₂ > 0 (convex — benefits from large market moves)",
            "Stable over time",
        ],
        "avoid": [
            "β₂ consistently negative  (short gamma)",
            "β₂ becoming more negative during high-volatility periods",
            "Large swings in β₂  (payoff structure changes with regime)",
        ],
    },
    {
        "number": "R5",
        "title":  "Rolling β_up / β_down  (Test 4a)",
        "what": (
            "Shows both the up-market beta (blue) and down-market beta (red) computed on "
            "a trailing window. "
            "Ideally both lines track each other closely. "
            "When the red line (down-market beta) diverges above the blue line in absolute "
            "terms, the strategy is crash-sensitive in that period. "
            "The shaded region highlights periods where |β_down| > |β_up|."
        ),
        "formula": (
            "β_up_t  = OLS(R_strat, R_asset) on up days in trailing window\n"
            "            β_down_t = OLS(R_strat, R_asset) on down days in trailing window"
        ),
        "variables": [
            ("Blue line",       "β_up — market sensitivity on rising days."),
            ("Red line",        "β_down — market sensitivity on falling days."),
            ("Red shading",     "|β_down| > |β_up| — crash-bias present."),
        ],
        "want": [
            "Both lines track closely and near zero",
            "No persistent red shading (crash-bias absent)",
            "Stable relationship across market regimes",
        ],
        "avoid": [
            "Red line consistently higher in magnitude than blue",
            "β_down spiking negative during market stress periods",
            "Growing divergence over time  (crash sensitivity worsening)",
        ],
    },
    {
        "number": "R6",
        "title":  "Rolling Asymmetry Ratio  (Test 4b)",
        "what": (
            "|β_down| / |β_up| over a trailing window. "
            "A ratio of 1.0 means perfectly symmetric market exposure. "
            "A ratio above 1.5 means the strategy loses more than proportionally "
            "on down days relative to what it gains on up days — crash sensitivity. "
            "This is the single most actionable risk metric for carry and short-vol strategies."
        ),
        "formula": "Ratio_t  =  |β_down_t|  /  |β_up_t|   on the trailing window",
        "variables": [
            ("Amber line",  "Rolling asymmetry ratio."),
            ("Red fill",    "Ratio > 1.5 — crash-sensitive period."),
            ("Green fill",  "0 < Ratio ≤ 1.5 — acceptable asymmetry."),
            ("1.0 line",    "Perfect symmetry boundary."),
            ("1.5 line",    "Warning threshold."),
        ],
        "want": [
            "Ratio consistently near 1.0  (symmetric)",
            "No sustained periods above 1.5",
            "Verdict: SYMMETRIC",
        ],
        "avoid": [
            "Ratio > 1.5 for extended periods  (CRASH_SENSITIVE)",
            "Ratio rising during market stress  (leverage effect)",
            "Ratio > 2.0 at any point  (severe crash sensitivity)",
        ],
    },
    {
        "number": "R7",
        "title":  "Rolling Alpha vs Permutation Null  (Test 6 — rolling placebo)",
        "what": (
            "Shows the rolling OLS intercept (alpha) on each trailing window as a white line. "
            "The shaded grey bands are the 5th–95th and 25th–75th percentiles of the "
            "full-period null distribution from 500 individual permutations. "
            "When the rolling alpha is above the p95 band (green fill), the strategy "
            "is generating alpha that is unlikely to be noise in that period. "
            "When it dips below the null median (red fill), the signal has degraded to "
            "noise level in that window. "
            "This is the most powerful rolling view: you can see exactly when the strategy "
            "stopped working."
        ),
        "formula": (
            "Rolling alpha_t  = OLS intercept on trailing window\n"
            "            Null bands        = percentiles of 500-permutation shuffle distribution"
        ),
        "variables": [
            ("White line",      "Rolling OLS alpha on the trailing window."),
            ("Grey bands",      "p5–p95 and p25–p75 of the null (no-skill) distribution."),
            ("Dashed line",     "Null median — the expected alpha under the null."),
            ("Green fill",      "Rolling alpha above null p95 — statistically meaningful."),
            ("Red fill",        "Rolling alpha below null median — noise territory."),
            ("p95 dotted",      "5% false positive threshold — clear this consistently."),
        ],
        "want": [
            "White line consistently above the grey bands (real persistent alpha)",
            "Green fill for most of the history",
            "Rolling alpha stable or rising over time",
        ],
        "avoid": [
            "White line buried inside the grey bands  (strategy returns are noise)",
            "Red fill for extended periods  (alpha has decayed)",
            "Sustained decline in rolling alpha toward zero or below",
            "Alpha only appearing in isolated windows  (data-mining artefact)",
        ],
    },
]


def _open_rolling_explain_window() -> None:
    """Open a separate scrollable Tk window with rolling-metrics explanations."""
    import tkinter as tk

    win = tk.Toplevel()
    win.title("Rolling Metrics — How to Read the Plots")
    win.configure(bg="#0d1117")
    win.geometry("860x720")
    win.resizable(True, True)

    frame = tk.Frame(win, bg="#0d1117")
    frame.pack(fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame, bg="#21262d", troughcolor="#0d1117",
                             activebackground="#30363d")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    txt = tk.Text(
        frame,
        bg="#0d1117", fg="#e6edf3",
        font=("Consolas", 11),
        yscrollcommand=scrollbar.set,
        wrap=tk.WORD, padx=28, pady=20,
        borderwidth=0, highlightthickness=0,
        cursor="arrow", state=tk.NORMAL,
    )
    txt.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=txt.yview)

    txt.tag_configure("header",
        font=("Consolas", 14, "bold"), foreground="#f0c060",
        spacing1=18, spacing3=6)
    txt.tag_configure("header_sub",
        font=("Consolas", 9), foreground="#8b949e", spacing3=10)
    txt.tag_configure("section",
        font=("Consolas", 10, "bold"), foreground="#8b949e",
        spacing1=10, spacing3=2)
    txt.tag_configure("what",
        font=("Consolas", 10), foreground="#c9d1d9",
        spacing3=6, lmargin1=16, lmargin2=16)
    txt.tag_configure("formula_box",
        font=("Consolas", 11, "bold"), foreground="#58a6ff",
        background="#161b22",
        spacing1=6, spacing3=6, lmargin1=16, lmargin2=16)
    txt.tag_configure("var_name",
        font=("Consolas", 10, "bold"), foreground="#bc8cff")
    txt.tag_configure("var_desc",
        font=("Consolas", 10), foreground="#8b949e")
    txt.tag_configure("want_header",
        font=("Consolas", 10, "bold"), foreground="#3fb950",
        spacing1=8, spacing3=2)
    txt.tag_configure("want_item",
        font=("Consolas", 10), foreground="#3fb950",
        lmargin1=28, lmargin2=38, spacing3=1)
    txt.tag_configure("avoid_header",
        font=("Consolas", 10, "bold"), foreground="#f85149",
        spacing1=8, spacing3=2)
    txt.tag_configure("avoid_item",
        font=("Consolas", 10), foreground="#f85149",
        lmargin1=28, lmargin2=38, spacing3=1)
    txt.tag_configure("divider",
        font=("Consolas", 7), foreground="#21262d",
        spacing1=14, spacing3=14)

    txt.insert(tk.END, "Rolling Metrics — How to Read the 7 Charts\n", "header")
    txt.insert(tk.END,
        "Each metric is computed on a trailing window of your chosen size.\n"
        "Use the TextBox to change the window and click Run to recompute.\n"
        "The wide bottom chart (R7) shows rolling alpha vs the permutation null.\n",
        "header_sub")
    txt.insert(tk.END, "─" * 80 + "\n", "divider")

    for guide in _ROLLING_GUIDES:
        txt.insert(tk.END,
            f"  PLOT {guide['number']}  ·  {guide['title']}\n", "header")
        txt.insert(tk.END, "What it shows\n", "section")
        txt.insert(tk.END, guide["what"] + "\n", "what")
        txt.insert(tk.END, "Formula\n", "section")
        txt.insert(tk.END, "  " + guide["formula"] + "\n", "formula_box")
        txt.insert(tk.END, "Variables\n", "section")
        for var, desc in guide["variables"]:
            txt.insert(tk.END, f"  {var:<18}", "var_name")
            txt.insert(tk.END, f"  {desc}\n", "var_desc")
        txt.insert(tk.END, "What you WANT to see\n", "want_header")
        for item in guide["want"]:
            txt.insert(tk.END, f"  •  {item}\n", "want_item")
        txt.insert(tk.END, "What to AVOID\n", "avoid_header")
        for item in guide["avoid"]:
            txt.insert(tk.END, f"  •  {item}\n", "avoid_item")
        txt.insert(tk.END, "─" * 80 + "\n", "divider")

    txt.config(state=tk.DISABLED)
    txt.see("1.0")


def _add_rolling_explain_button(fig: plt.Figure) -> None:
    btn_ax = fig.add_axes([0.865, 0.960, 0.10, 0.020])
    btn_ax.set_facecolor(PANEL)
    for sp in btn_ax.spines.values():
        sp.set_edgecolor(BORDER)
    btn = Button(btn_ax, "?  Explain", color=PANEL, hovercolor="#21262d")
    btn.label.set_color(DIM)
    btn.label.set_fontsize(8.5)

    def _open(_event):
        btn.label.set_color(AMBER)
        fig.canvas.draw_idle()
        _open_rolling_explain_window()
        btn.label.set_color(DIM)
        fig.canvas.draw_idle()

    btn.on_clicked(_open)
    fig._rolling_explain_btn = btn


# ── Shared helpers ────────────────────────────────────────────────────────────

def _subtitle(ax, formula: str) -> None:
    ax.text(
        0.5, 1.01, formula,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.5, color=DIM, style="italic",
    )


def _style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=7.5)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.5, linestyle="--", alpha=0.5)


def _fmt_date_axis(ax, dates) -> None:
    """Apply year-based date formatter and rotate labels."""
    ax.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%Y") if len(dates) > 500
        else matplotlib.dates.DateFormatter("%Y-%m")
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
