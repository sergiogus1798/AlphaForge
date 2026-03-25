"""
plot_alpha.py — Matplotlib visualization for AlphaDetector results.

Six subplots in a single dark-themed figure:
  1. Regression scatter      — daily strategy vs asset returns with OLS line
  2. Alpha stream equity     — cumulative return of beta-neutral residuals
  3. Rolling alpha           — 60-day rolling alpha with significance shading
  4. Rolling beta            — 60-day rolling beta with reference lines
  5. Overnight vs Intraday   — side-by-side alpha/beta/R² comparison
  6. Stability & stress      — alpha bar chart across all sub-periods

Usage:
    from alphaforge.IndividualAnalysis.AlphaDetection.plot_alpha import plot_alpha_report
    from alphaforge.IndividualAnalysis.AlphaDetection import AlphaDetector

    detector = AlphaDetector(trades_df, price_df, strategy_name="AUDJPY7", pair="AUDJPY")
    plot_alpha_report(detector)
"""

import warnings

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
import scipy.stats as stats

from .alpha_detection import AlphaDetector, AlphaReport, RegressionResult

# ── Colour palette (matches plot_mc.py style) ─────────────────────────────────
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

VERDICT_COLOR = {"ALPHA": GREEN, "BETA": RED, "INCONCLUSIVE": AMBER}

# ── Public entry point ─────────────────────────────────────────────────────────

def plot_alpha_report(
    detector:       AlphaDetector,
    rolling_window: int  = 60,
    all_strategies: dict | None = None,
    reload_fn            = None,   # callable(key) -> AlphaDetector
) -> None:
    """
    Run the full alpha detection pipeline and open a visualization window.

    Args:
        detector:       An AlphaDetector instance (trades + price data loaded).
        rolling_window: Days per rolling regression window (default 60).
        all_strategies: Full strategy dict for the selector dropdown (optional).
        reload_fn:      callable(key) -> AlphaDetector — builds a new detector
                        for the chosen strategy so the plots can be refreshed.
    """
    # ── Run pipeline ──────────────────────────────────────────────────────────
    print(f"Running alpha detection for '{detector.strategy_name}'...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report  = detector.run_all()
        aligned = detector.build_returns()

    if report.core is None:
        print("Not enough data to build a report.")
        return

    print("Computing rolling alpha/beta...")
    rolling = detector.rolling_alpha_beta(aligned, window=rolling_window)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    title_obj = fig.suptitle(
        f"Alpha Detection  |  {detector.strategy_name}  |  {detector.pair}",
        color=TEXT,
        fontsize=12, fontweight="bold", y=0.98,
    )

    has_selector = all_strategies is not None and reload_fn is not None
    bot = 0.12 if has_selector else 0.07

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        top=0.93, bottom=bot, left=0.07, right=0.97,
        hspace=0.62, wspace=0.32,
    )

    ax_scatter  = fig.add_subplot(gs[0, 0])
    ax_equity   = fig.add_subplot(gs[0, 1])
    ax_r_alpha  = fig.add_subplot(gs[1, 0])
    ax_r_beta   = fig.add_subplot(gs[1, 1])
    ax_decomp   = fig.add_subplot(gs[2, 0])
    ax_bars     = fig.add_subplot(gs[2, 1])

    _plot_axes = (ax_scatter, ax_equity, ax_r_alpha, ax_r_beta, ax_decomp, ax_bars)
    for ax in _plot_axes:
        _style_ax(ax)

    def _draw_all(det, rpt, rol):
        for ax in _plot_axes:
            ax.cla()
            _style_ax(ax)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aln = det.build_returns()
        _plot_scatter(ax_scatter, aln, rpt.core)
        _plot_equity(ax_equity, rpt.core)
        _plot_rolling_alpha(ax_r_alpha, rol, window=rolling_window)
        _plot_rolling_beta(ax_r_beta, rol, window=rolling_window)
        _plot_decomp(ax_decomp, rpt)
        _plot_period_bars(ax_bars, rpt)
        title_obj.set_text(
            f"Alpha Detection  |  {det.strategy_name}  |  {det.pair}"
        )
        fig.canvas.draw_idle()

    # Initial draw
    _draw_all(detector, report, rolling)

    # ── Explain button ────────────────────────────────────────────────────────
    _add_explain_button(fig)

    # ── Strategy selector + Re-run (only when reload_fn supplied) ─────────────
    if has_selector:
        strategy_keys = list(all_strategies.keys())
        _state = {"key": detector.strategy_name}

        # Strategy label
        strat_label = fig.text(
            0.30, 0.055, detector.strategy_name,
            color=TEXT, fontsize=8, ha="left", va="center", clip_on=True,
        )
        status_text = fig.text(
            0.30, 0.025, "",
            color=DIM, fontsize=7.5, ha="left", va="center",
        )

        # Selector button
        ax_sel = fig.add_axes([0.01, 0.038, 0.21, 0.034])
        ax_sel.set_facecolor(PANEL)
        for sp in ax_sel.spines.values():
            sp.set_edgecolor(BORDER)
        btn_sel = Button(ax_sel, "▾  Select Strategy", color=PANEL, hovercolor="#21262d")
        btn_sel.label.set_color(BLUE)
        btn_sel.label.set_fontsize(8.5)

        def _open_selector(_event):
            import tkinter as tk
            from tkinter import ttk
            popup = tk.Toplevel()
            popup.title("Select Strategy")
            popup.configure(bg="#0d1117")
            popup.geometry("420x480")
            popup.resizable(False, True)
            tk.Label(popup, text="Select a strategy:", bg="#0d1117",
                     fg="#8b949e", font=("Consolas", 10)).pack(pady=(10, 4))
            filter_var = tk.StringVar()
            filter_entry = ttk.Entry(popup, textvariable=filter_var,
                                     font=("Consolas", 10))
            filter_entry.pack(fill=tk.X, padx=10)
            filter_entry.focus()
            listbox = tk.Listbox(
                popup, bg="#161b22", fg="#e6edf3",
                selectbackground="#1f6feb", selectforeground="#ffffff",
                font=("Consolas", 10), relief=tk.FLAT,
                activestyle="none", height=20,
            )
            sb = tk.Scrollbar(popup, command=listbox.yview,
                              bg="#21262d", troughcolor="#0d1117")
            listbox.config(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4))
            listbox.pack(fill=tk.BOTH, expand=True, padx=(10, 0), pady=4)

            def _populate(filter_text=""):
                listbox.delete(0, tk.END)
                for k in strategy_keys:
                    if filter_text.lower() in k.lower():
                        listbox.insert(tk.END, k)
                for i in range(listbox.size()):
                    if listbox.get(i) == _state["key"]:
                        listbox.selection_set(i)
                        listbox.see(i)
                        break

            filter_var.trace_add("write", lambda *_: _populate(filter_var.get()))
            _populate()

            def _confirm():
                sel = listbox.curselection()
                if sel:
                    chosen = listbox.get(sel[0])
                    _state["key"] = chosen
                    strat_label.set_text(chosen)
                    fig.canvas.draw_idle()
                popup.destroy()

            listbox.bind("<Double-Button-1>", lambda _: _confirm())
            listbox.bind("<Return>", lambda _: _confirm())
            btn_frame = tk.Frame(popup, bg="#0d1117")
            btn_frame.pack(fill=tk.X, padx=10, pady=(4, 10))
            tk.Button(btn_frame, text="Select", command=_confirm,
                      bg="#1f6feb", fg="white", font=("Consolas", 10, "bold"),
                      relief=tk.FLAT, padx=10).pack(side=tk.RIGHT)
            tk.Button(btn_frame, text="Cancel", command=popup.destroy,
                      bg="#21262d", fg="#8b949e", font=("Consolas", 10),
                      relief=tk.FLAT, padx=10).pack(side=tk.RIGHT, padx=(0, 6))
            popup.lift()
            popup.grab_set()

        btn_sel.on_clicked(_open_selector)

        # Re-run button
        ax_run = fig.add_axes([0.78, 0.038, 0.10, 0.034])
        ax_run.set_facecolor(PANEL)
        for sp in ax_run.spines.values():
            sp.set_edgecolor(BORDER)
        btn_run = Button(ax_run, "▶  Re-run", color=PANEL, hovercolor="#21262d")
        btn_run.label.set_color(GREEN)
        btn_run.label.set_fontsize(9)

        def _on_rerun(_event=None):
            key = _state["key"]
            status_text.set_text("Computing…")
            fig.canvas.draw_idle()
            try:
                new_det = reload_fn(key)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new_rpt = new_det.run_all()
                new_rol = new_det.rolling_alpha_beta(
                    new_det.build_returns(), window=rolling_window
                )
                _draw_all(new_det, new_rpt, new_rol)
                status_text.set_text("")
            except Exception as exc:
                status_text.set_text(f"Error: {exc}")
            fig.canvas.draw_idle()

        btn_run.on_clicked(_on_rerun)

        # Keep references alive
        fig._alpha_widgets = (btn_sel, btn_run, strat_label, status_text, _state)

    plt.show(block=True)


# ── Individual plot functions ─────────────────────────────────────────────────

def _plot_scatter(ax, aligned: pd.DataFrame, core: RegressionResult) -> None:
    """Scatter of daily returns (in %) with OLS regression line."""
    ax.set_title("Core Regression", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax, r"$R_{strat}$ (%) $= \alpha + \beta \cdot R_{asset}$ (%) $+ \varepsilon$")
    ax.set_xlabel("Asset return (%)", color=DIM, fontsize=8)
    ax.set_ylabel("Strategy return (%)", color=DIM, fontsize=8)

    # Convert to percentage
    x = aligned["asset"].values   * 100
    y = aligned["strategy"].values * 100

    ax.scatter(x, y, s=4, alpha=0.35, color=BLUE, linewidths=0)

    # Regression line (fit is linear so scaling both axes keeps α/β in % terms)
    if not np.isnan(core.alpha):
        alpha_pct = core.alpha * 100          # daily alpha in %
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = alpha_pct + core.beta * x_line
        ax.plot(x_line, y_line, color=AMBER, lw=1.5)

        # Confidence band (95%)
        n    = len(x)
        resid = y - alpha_pct - core.beta * x
        se   = np.sqrt(np.sum(resid ** 2) / (n - 2))
        x_m  = x.mean()
        band = 1.96 * se * np.sqrt(1/n + (x_line - x_m)**2 / np.sum((x - x_m)**2))
        ax.fill_between(x_line, y_line - band, y_line + band, alpha=0.15, color=AMBER)

    ax.axhline(0, color=BORDER, lw=0.7)
    ax.axvline(0, color=BORDER, lw=0.7)

    # Stats annotation (α shown in % per day)
    verdict_col = VERDICT_COLOR.get(core.verdict, DIM)
    ax.text(
        0.04, 0.97,
        f"α = {core.alpha*100:+.4f}%/day\nβ = {core.beta:+.3f}\n"
        f"t(α) = {core.alpha_tstat:+.2f}\nR² = {core.r_squared:.3f}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=7.5, color=TEXT, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=verdict_col, lw=1),
    )
    ax.text(
        0.97, 0.97, f"[{core.verdict}]",
        transform=ax.transAxes, va="top", ha="right",
        fontsize=8, color=verdict_col, fontweight="bold",
    )


def _plot_equity(ax, core: RegressionResult) -> None:
    """Cumulative return of the beta-neutral alpha stream (residuals)."""
    ax.set_title("Alpha Stream  (beta-neutral residuals)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax, r"stream $= R_{strat} - \beta \cdot R_{asset}$  (market exposure stripped)")
    ax.set_ylabel("Cumulative return", color=DIM, fontsize=8)

    if core is None or core.residuals.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        return

    resid   = core.residuals.dropna()
    equity  = (1 + resid).cumprod()
    dates   = equity.index

    ax.plot(dates, equity, color=AMBER, lw=1.4)
    ax.fill_between(dates, 1.0, equity,
                    where=(equity >= 1.0), alpha=0.20, color=GREEN)
    ax.fill_between(dates, 1.0, equity,
                    where=(equity < 1.0),  alpha=0.20, color=RED)
    ax.axhline(1.0, color=BORDER, lw=0.8, linestyle="--")

    # Annotate final return
    final = float(equity.iloc[-1])
    col   = GREEN if final >= 1.0 else RED
    ax.text(0.97, 0.05, f"{(final-1)*100:+.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=col, fontweight="bold")

    ax.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%Y") if len(dates) > 500
        else matplotlib.dates.DateFormatter("%Y-%m")
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _plot_rolling_alpha(ax, rolling: pd.DataFrame, window: int) -> None:
    """Rolling alpha over time; shaded green where t-stat > 2."""
    ax.set_title(f"Rolling Alpha  ({window}-day window)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax, r"$\alpha(t)$ = OLS intercept on trailing " + str(window) + r" days  |  green = $|t(\alpha)| > 2$")
    ax.set_ylabel("α", color=DIM, fontsize=8)

    dates  = rolling.index
    alpha  = rolling["alpha"].values
    tstat  = rolling["alpha_tstat"].values

    # Shade significant positive alpha
    sig_mask = (tstat > 2) & (alpha > 0)
    ax.fill_between(dates, 0, alpha, where=sig_mask,
                    alpha=0.25, color=GREEN, label="α sig (t>2)")
    ax.fill_between(dates, 0, alpha, where=~sig_mask,
                    alpha=0.15, color=DIM, label="α not sig")

    ax.plot(dates, alpha, color=GREEN, lw=1.2)
    ax.axhline(0, color=BORDER, lw=0.8, linestyle="--")

    # Fraction of time significant
    frac = sig_mask.mean() * 100
    ax.text(0.02, 0.97, f"Significant {frac:.0f}% of time",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, color=DIM, fontfamily="monospace")

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _plot_rolling_beta(ax, rolling: pd.DataFrame, window: int) -> None:
    """Rolling beta over time with reference lines at 0 and 1."""
    ax.set_title(f"Rolling Beta  ({window}-day window)", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax, r"$\beta(t) = \mathrm{Cov}(R_s,\,R_a)\,/\,\mathrm{Var}(R_a)$  on trailing " + str(window) + r" days")
    ax.set_ylabel("β", color=DIM, fontsize=8)

    dates = rolling.index
    beta  = rolling["beta"].values

    ax.plot(dates, beta, color=BLUE, lw=1.2)
    ax.fill_between(dates, 0, beta,
                    where=(beta > 0), alpha=0.15, color=BLUE)
    ax.fill_between(dates, 0, beta,
                    where=(beta < 0), alpha=0.15, color=RED)

    ax.axhline(0,   color=BORDER,  lw=0.8, linestyle="--")
    ax.axhline(1.0, color=ORANGE,  lw=0.7, linestyle=":", label="β=1 (full exposure)")
    ax.axhline(-1.0,color=ORANGE,  lw=0.7, linestyle=":")

    ax.text(0.97, 0.97, f"avg β = {beta.mean():+.3f}",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=7.5, color=DIM, fontfamily="monospace")

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _plot_decomp(ax, report: AlphaReport) -> None:
    """Side-by-side bar chart: overnight vs intraday alpha, beta, R²."""
    ax.set_title("Overnight vs Intraday Decomposition", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax, r"$R_{ON} = Open_t / Close_{t-1} - 1$  |  $R_{ID} = Close_t / Open_t - 1$")

    on = report.overnight
    id_ = report.intraday

    metrics = ["α", "β", "R²"]
    on_vals  = [
        on.alpha       if on  and not np.isnan(on.alpha)       else np.nan,
        on.beta        if on  and not np.isnan(on.beta)        else np.nan,
        on.r_squared   if on  and not np.isnan(on.r_squared)   else np.nan,
    ]
    id_vals = [
        id_.alpha      if id_ and not np.isnan(id_.alpha)      else np.nan,
        id_.beta       if id_ and not np.isnan(id_.beta)       else np.nan,
        id_.r_squared  if id_ and not np.isnan(id_.r_squared)  else np.nan,
    ]

    x     = np.arange(len(metrics))
    width = 0.35

    bars_on = ax.bar(x - width/2, on_vals,  width, label="Overnight", color=PURPLE, alpha=0.80)
    bars_id = ax.bar(x + width/2, id_vals, width, label="Intraday",  color=ORANGE,  alpha=0.80)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=TEXT, fontsize=9)
    ax.axhline(0, color=BORDER, lw=0.8)
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    # Annotate verdicts
    for reg, label, xpos in [(on, "ON", -0.18), (id_, "ID", 0.18)]:
        if reg is not None and reg.verdict != "INCONCLUSIVE":
            col = VERDICT_COLOR.get(reg.verdict, DIM)
            ax.text(xpos, 0.97, f"[{reg.verdict}]",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color=col, fontweight="bold")


def _plot_period_bars(ax, report: AlphaReport) -> None:
    """Alpha bar chart across stability halves and stress regimes."""
    ax.set_title("Alpha Across Periods", color=TEXT, fontsize=9, pad=18)
    _subtitle(ax, r"Same OLS run independently on each sub-period  |  bar = $\alpha$,  label = $t(\alpha)$")
    ax.set_ylabel("α (daily)", color=DIM, fontsize=8)

    entries = [
        ("Overall",    report.core),
        ("Early half", report.stability_early),
        ("Late half",  report.stability_late),
    ]
    for name, r in report.stress_results.items():
        entries.append((name, r))

    labels = [e[0] for e in entries]
    alphas = [
        e[1].alpha if (e[1] is not None and not np.isnan(e[1].alpha)) else 0.0
        for e in entries
    ]
    colors = [
        VERDICT_COLOR.get(e[1].verdict, DIM)
        if e[1] is not None else DIM
        for e in entries
    ]
    tstats = [
        e[1].alpha_tstat if (e[1] is not None and not np.isnan(e[1].alpha_tstat)) else 0.0
        for e in entries
    ]

    x    = np.arange(len(labels))
    bars = ax.bar(x, alphas, color=colors, alpha=0.80, width=0.6)

    # Annotate t-stats above/below bars
    for i, (bar, t) in enumerate(zip(bars, tstats)):
        if t == 0.0:
            continue
        ypos  = bar.get_height() if bar.get_height() >= 0 else bar.get_height()
        va    = "bottom" if bar.get_height() >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"t={t:+.1f}", ha="center", va=va,
                fontsize=6.5, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=7.5, rotation=25, ha="right")
    ax.axhline(0, color=BORDER, lw=0.8)

    # Legend for colours
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=GREEN,  label="ALPHA"),
        Patch(facecolor=RED,    label="BETA"),
        Patch(facecolor=AMBER,  label="INCONCLUSIVE"),
        Patch(facecolor=DIM,    label="No data"),
    ]
    ax.legend(handles=legend_els, fontsize=7, facecolor=PANEL,
              edgecolor=BORDER, labelcolor=TEXT, loc="upper right")


# ── Explain window (separate scrollable Tk window) ────────────────────────────

# Structured content for each plot guide
_PLOT_GUIDES = [
    {
        "number": "01",
        "title":  "Core Regression",
        "position": "top-left",
        "what": (
            "Each dot is one trading day. The X-axis shows what the asset "
            "(e.g. AUDJPY) did that day; the Y-axis shows what the strategy made. "
            "The amber line is the best-fit regression — it summarises the average "
            "relationship between market moves and strategy returns."
        ),
        "formula": "R_strategy  =  α  +  β · R_asset  +  ε",
        "variables": [
            ("α  (alpha)",  "Return earned when the market is flat — the pure edge."),
            ("β  (beta)",   "Market sensitivity. β = 1 means moves 1-for-1 with the market."),
            ("R²",          "Fraction of strategy returns explained by the market (0 = none, 1 = all)."),
            ("t(α)",        "Statistical confidence in alpha (Student t-test). Needs to exceed 2."),
        ],
        "want": [
            "α > 0                  Positive daily alpha",
            "|t(α)| > 2.0           Alpha is statistically significant (not luck)",
            "|β| < 0.20             Low market dependency",
            "R² < 0.10              Market explains less than 10% of returns",
            "Dots scattered randomly around the line (low R²)",
        ],
        "avoid": [
            "α < 0                  Strategy destroys value after hedging",
            "|t(α)| < 2.0           Alpha is not statistically real",
            "R² > 0.30              Returns mostly explained by the market",
            "|β| > 0.5              Too much market exposure",
            "All dots on a tight line (pure beta — no independent edge)",
        ],
    },
    {
        "number": "02",
        "title":  "Alpha Stream  (Beta-Neutral Equity Curve)",
        "position": "top-right",
        "what": (
            "This is the cumulative equity of the beta-neutral strategy — what "
            "is left after stripping all market exposure every day. Think of it as: "
            "'if we perfectly hedged the market every day, how would the remaining "
            "returns compound over time?' A growing curve means the edge is real "
            "and independent of the market."
        ),
        "formula": "stream(t)  =  Π  (1 + R_strategy(i) − β · R_asset(i))   for i = 1 … t",
        "variables": [
            ("β · R_asset",  "The market component, removed from each daily return."),
            ("Residual",     "What is left — the strategy's daily alpha contribution."),
            ("Cumulative",   "All residuals compounded over time, starting from 1.0."),
        ],
        "want": [
            "Steadily rising curve over the full period",
            "Annualised Sharpe of residuals > 0.5",
            "Maximum drawdown of the curve < 25%",
            "Final value clearly above 1.0  (positive total alpha return)",
        ],
        "avoid": [
            "Flat curve after early gains (edge has decayed)",
            "Curve that mimics the market (beta is leaking through)",
            "Deep drawdowns > 40%  (fragile — will not survive live trading)",
            "Declining in recent years  (edge may already be dead)",
        ],
    },
    {
        "number": "03",
        "title":  "Rolling Alpha",
        "position": "middle-left",
        "what": (
            "Instead of a single regression over the whole period, this runs a "
            "fresh OLS every day using only the past 60 trading days. The resulting "
            "alpha line shows how the edge evolved over time. "
            "Green shading = periods where t(α) > 2 (significant edge). "
            "Grey shading = alpha not statistically significant in that window."
        ),
        "formula": "α(t)  =  OLS intercept  fitted on  [ t − 60, t ]",
        "variables": [
            ("60-day window",   "Each point uses only the 60 most recent days."),
            ("Green zone",      "t(α) > 2 in that window — edge is statistically present."),
            ("Grey zone",       "t(α) ≤ 2 — edge is not significant in that window."),
            ("% significant",   "Annotation showing fraction of time in the green zone."),
        ],
        "want": [
            "α(t) > 0 for most of the chart",
            "Green shading covers > 50% of the timeline",
            "Smooth positive line without long collapses",
            "Recent portion (right side) is green and positive",
        ],
        "avoid": [
            "Long grey stretches (edge disappears for months at a time)",
            "α(t) < 0 in recent periods  (strategy may be dying right now)",
            "Very spiky α with no persistence  (no real edge, just noise)",
            "Green only in the first half, grey in the second  (edge has decayed)",
        ],
    },
    {
        "number": "04",
        "title":  "Rolling Beta",
        "position": "middle-right",
        "what": (
            "Same 60-day rolling window but tracking beta instead of alpha. "
            "Shows how much market exposure the strategy carried at each point in "
            "time. The dashed orange lines mark ±1 (full market exposure). "
            "A strategy with genuine alpha should sit quietly near zero throughout."
        ),
        "formula": "β(t)  =  Cov(R_strategy, R_asset)  /  Var(R_asset)   on trailing 60 days",
        "variables": [
            ("β(t) = 0",   "Strategy is market-neutral at that point in time."),
            ("β(t) = +1",  "Strategy moves in lockstep with the market (full long beta)."),
            ("β(t) = −1",  "Strategy is effectively short the market."),
        ],
        "want": [
            "|β(t)| < 0.20 on average across the full period",
            "Flat, stable line close to zero",
            "Low variance — not swinging between positive and negative",
            "No sudden large spikes  (consistent risk profile)",
        ],
        "avoid": [
            "|β(t)| > 0.5  (strategy is riding the market, not generating edge)",
            "Sudden spikes toward ±1  (hidden regime-based market exposure)",
            "Consistently negative β  (short-market bet — fragile in bull markets)",
            "β drifting higher over time  (strategy becoming more market-dependent)",
        ],
    },
    {
        "number": "05",
        "title":  "Overnight vs Intraday Decomposition",
        "position": "bottom-left",
        "what": (
            "Splits both the asset returns and the strategy's trades into two "
            "components: the overnight gap (from previous close to open) and "
            "the intraday move (from open to close). Runs the regression "
            "separately for each component. This tells you WHERE in the day "
            "the strategy's edge actually lives."
        ),
        "formula": (
            "R_overnight  =  Open(t) / Close(t−1)  −  1\n"
            "                     R_intraday   =  Close(t) / Open(t)    −  1"
        ),
        "variables": [
            ("Overnight (purple)",  "Trades held across the session close — captures gaps, news, macro moves."),
            ("Intraday  (orange)",  "Trades opened and closed within the same day — execution, flow."),
            ("α bar",              "Alpha in that component — how much edge exists there."),
            ("β bar",              "Market sensitivity in that component."),
            ("R² bar",             "How much of returns the market explains in that component."),
        ],
        "want": [
            "α > 0 with t(α) > 2 in at least one component (ON or ID)",
            "Overnight α dominant = macro/news edge  (very robust, hard to erode)",
            "Intraday α dominant = execution edge  (valid but more fragile over time)",
            "β close to zero in both components",
        ],
        "avoid": [
            "Both α bars near zero  (edge vanishes when decomposed — suspicious)",
            "Very high β in overnight  (just riding gap risk, not skill)",
            "Very high R² in either component  (market is doing all the work)",
            "All return coming from one regime that may not persist",
        ],
    },
    {
        "number": "06",
        "title":  "Alpha Across Periods  (Stability & Stress Tests)",
        "position": "bottom-right",
        "what": (
            "Runs the same core regression independently on multiple sub-periods: "
            "the full backtest, the early half, the late half, and three historical "
            "crisis regimes. Bar height = alpha. Colour = verdict. "
            "This is your stability and stress-test dashboard at a glance."
        ),
        "formula": "Same  R_strategy = α + β · R_asset  run on each sub-window independently",
        "variables": [
            ("Overall",          "Full backtest period — the headline number."),
            ("Early / Late",     "Chronological split test — alpha must persist in both halves."),
            ("GFC 2008",         "Global Financial Crisis — extreme volatility, liquidity crisis."),
            ("COVID 2020",       "Pandemic shock — V-shaped crash and recovery."),
            ("Rate hike 2022",   "Aggressive central bank tightening — trend reversal across all assets."),
            ("GREEN bar",        "ALPHA verdict: significant positive alpha in that period."),
            ("RED bar",          "BETA verdict: returns explained by market, not edge."),
            ("AMBER bar",        "INCONCLUSIVE: not enough data or t(α) below threshold."),
        ],
        "want": [
            "GREEN in Overall, Early, AND Late  (edge is stable — not a one-period fluke)",
            "t(α) > 2 annotated on at least Overall + one of the halves",
            "Positive bars (even if amber) in the stress regimes",
            "Consistent bar heights across periods  (uniform edge across time)",
        ],
        "avoid": [
            "GREEN only in Early, grey/red in Late  (edge has already decayed — do not trade)",
            "GREEN only in Late  (may be overfitted to recent data — unproven edge)",
            "RED in Overall  (strategy is beta in disguise — no real edge at all)",
            "All bars near zero or negative in stress periods  (collapses exactly when needed most)",
            "Large swings in bar height across periods  (unstable, regime-dependent edge)",
        ],
    },
]


def _open_explain_window() -> None:
    """Open a separate scrollable Tk window with plot-by-plot explanations."""
    import tkinter as tk

    win = tk.Toplevel()
    win.title("Alpha Detection — How to Read the Plots")
    win.configure(bg="#0d1117")
    win.geometry("860x720")
    win.resizable(True, True)

    # ── Scrollable text area ──────────────────────────────────────────────────
    frame = tk.Frame(win, bg="#0d1117")
    frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    scrollbar = tk.Scrollbar(frame, bg="#21262d", troughcolor="#0d1117",
                             activebackground="#30363d")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    txt = tk.Text(
        frame,
        bg="#0d1117", fg="#e6edf3",
        font=("Consolas", 11),
        yscrollcommand=scrollbar.set,
        wrap=tk.WORD,
        padx=28, pady=20,
        borderwidth=0, highlightthickness=0,
        cursor="arrow",
        state=tk.NORMAL,
    )
    txt.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=txt.yview)

    # ── Text tags (styles) ────────────────────────────────────────────────────
    txt.tag_configure("header",
        font=("Consolas", 14, "bold"), foreground="#f0c060",
        spacing1=18, spacing3=6)
    txt.tag_configure("header_sub",
        font=("Consolas", 9), foreground="#8b949e",
        spacing3=10)
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

    # ── Page header ───────────────────────────────────────────────────────────
    txt.insert(tk.END, "How to Read the Alpha Detection Plots\n", "header")
    txt.insert(tk.END,
        "A practical guide — formula, variable definitions, and decision thresholds "
        "for each of the six diagnostic charts.\n", "header_sub")
    txt.insert(tk.END, "─" * 80 + "\n", "divider")

    # ── Insert each plot guide ────────────────────────────────────────────────
    for guide in _PLOT_GUIDES:
        # Plot title
        txt.insert(tk.END,
            f"  PLOT {guide['number']}  ·  {guide['title']}"
            f"                [{guide['position']}]\n", "header")

        # What is this plot
        txt.insert(tk.END, "What it shows\n", "section")
        txt.insert(tk.END, guide["what"] + "\n", "what")

        # Formula
        txt.insert(tk.END, "Formula\n", "section")
        txt.insert(tk.END, "  " + guide["formula"] + "\n", "formula_box")

        # Variables
        txt.insert(tk.END, "Variables\n", "section")
        for var, desc in guide["variables"]:
            txt.insert(tk.END, f"  {var:<18}", "var_name")
            txt.insert(tk.END, f"  {desc}\n", "var_desc")

        # Want
        txt.insert(tk.END, "✔  What you WANT to see\n", "want_header")
        for item in guide["want"]:
            txt.insert(tk.END, f"  •  {item}\n", "want_item")

        # Avoid
        txt.insert(tk.END, "✘  What to AVOID\n", "avoid_header")
        for item in guide["avoid"]:
            txt.insert(tk.END, f"  •  {item}\n", "avoid_item")

        txt.insert(tk.END, "─" * 80 + "\n", "divider")

    txt.config(state=tk.DISABLED)   # read-only
    txt.see("1.0")                  # scroll to top


def _add_explain_button(fig: plt.Figure) -> None:
    """Add a '? Explain' button that opens a separate scrollable guide window."""

    btn_ax = fig.add_axes([0.865, 0.955, 0.10, 0.030])
    btn_ax.set_facecolor(PANEL)
    for sp in btn_ax.spines.values():
        sp.set_edgecolor(BORDER)
    btn = Button(btn_ax, "?  Explain", color=PANEL, hovercolor="#21262d")
    btn.label.set_color(DIM)
    btn.label.set_fontsize(8.5)

    def _open(_event):
        btn.label.set_color(AMBER)
        fig.canvas.draw_idle()
        _open_explain_window()
        btn.label.set_color(DIM)
        fig.canvas.draw_idle()

    btn.on_clicked(_open)
    fig._explain_btn = btn


# ── Formula subtitle helper ───────────────────────────────────────────────────

def _subtitle(ax, formula: str) -> None:
    """Place a small italic formula line just above the axes, below the title."""
    ax.text(
        0.5, 1.01, formula,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.5, color=DIM, style="italic",
    )


# ── Axis styling helper ───────────────────────────────────────────────────────

def _style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=7.5)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.5, linestyle="--", alpha=0.5)
