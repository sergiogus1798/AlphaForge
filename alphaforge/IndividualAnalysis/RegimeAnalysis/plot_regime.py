"""
plot_regime.py — Interactive Regime Detection & Drawdown Clustering panel.

Opens one dark-themed figure with 4 panels + control bar:

  [0, :]  Price series with regime-coloured background (full width)
  [1, :]  Strategy equity curve with regime shading + drawdown markers (full width)
  [2, 0]  Cumulative returns per regime — strategy vs asset side-by-side
  [2, 1]  Drawdown clustering: runs-test + loss-ACF summary + drawdown depth scatter

Controls at bottom:
  ▾ Select Strategy  |  States [2][3][4][5]  |  Smooth (days): [21]  |  ▶ Run

Usage:
    from alphaforge.IndividualAnalysis.RegimeAnalysis import RegimeAnalysis, plot_regime_report
"""

import warnings

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.widgets import Button, TextBox
import numpy as np
import pandas as pd

from .regime_tests import RegimeReport, DrawdownPeriod


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

_REGIME_COLORS = [
    "#3fb950",   # Bull   — green
    "#f85149",   # Bear   — red
    "#f0c060",   # Hi-Vol — amber
    "#58a6ff",   # extra  — blue
    "#bc8cff",   # extra  — purple
]

_VERDICT_COLOR = {
    "ALPHA":          GREEN,
    "NEG_ALPHA":      RED,
    "BETA":           AMBER,
    "INCONCLUSIVE":   AMBER,
    "INSUFFICIENT":   DIM,
    "RANDOM":         GREEN,
    "CLUSTERED":      RED,
    "REGULAR":        BLUE,
    "INDEPENDENT":    GREEN,
    "MEAN_REVERTING": BLUE,
}

_MIN_SCATTER_DEPTH = 1.0


# ── Public entry point ────────────────────────────────────────────────────────

def plot_regime_report(
    report:         RegimeReport,
    strategy_name:  str  = "Strategy",
    pair:           str  = "",
    all_strategies: dict | None = None,
    reload_fn              = None,   # callable(key, n_states, smooth) -> (RegimeReport, name, pair)
) -> None:
    """
    Open the interactive regime panel.

    Args:
        report:         Initial RegimeReport to display.
        strategy_name:  Label for the title.
        pair:           Instrument label.
        all_strategies: If provided, enables the strategy selector.
        reload_fn:      callable(key, n_states, smooth) -> (RegimeReport, name, pair).
    """
    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    title_obj = fig.suptitle(
        f"Regime Analysis  |  {strategy_name}  |  {pair}",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        top=0.920, bottom=0.11,
        left=0.06, right=0.97,
        hspace=0.55, wspace=0.30,
        height_ratios=[1.1, 1.1, 1.4],
    )

    ax_price  = fig.add_subplot(gs[0, :])
    ax_equity = fig.add_subplot(gs[1, :])
    ax_perf   = fig.add_subplot(gs[2, 0])
    ax_clust  = fig.add_subplot(gs[2, 1])

    for ax in [ax_price, ax_equity, ax_perf, ax_clust]:
        _style_ax(ax)

    # ── Mutable state ─────────────────────────────────────────────────────────
    _state = {
        "report":   report,
        "key":      strategy_name,
        "n_states": 3,
        "smooth":   21,
    }

    # ── Draw function ─────────────────────────────────────────────────────────
    def _draw(rpt: RegimeReport, name: str, pr: str) -> None:
        for ax in [ax_price, ax_equity, ax_perf, ax_clust]:
            ax.cla(); _style_ax(ax)
        _plot_price_regimes(ax_price,  rpt)
        _plot_equity_regimes(ax_equity, rpt)
        _plot_regime_cumulative(ax_perf, rpt)
        _plot_clustering(ax_clust, rpt)
        title_obj.set_text(f"Regime Analysis  |  {name}  |  {pr}")
        fig.canvas.draw_idle()

    _draw(report, strategy_name, pair)
    _add_explain_button(fig)

    # ── Control bar ───────────────────────────────────────────────────────────
    CTRL_Y = 0.018
    CTRL_H = 0.038

    def _make_btn(x, w, label, cb, color=None):
        ax = fig.add_axes([x, CTRL_Y, w, CTRL_H])
        ax.set_facecolor(color or PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        btn = Button(ax, label, color=color or PANEL, hovercolor="#21262d")
        btn.label.set_color(TEXT); btn.label.set_fontsize(8.5)
        btn.on_clicked(cb)
        return btn, ax

    # Strategy selector
    strat_keys = sorted(all_strategies.keys()) if all_strategies else []
    _strat_lbl = fig.text(0.285, CTRL_Y + CTRL_H * 0.75, _state["key"],
                          color=TEXT, fontsize=7.5, ha="left", va="center",
                          fontfamily="monospace")

    btn_sel, _ = _make_btn(0.06, 0.21, "\u25be  Select Strategy", lambda _: None)
    btn_sel.label.set_color(BLUE)

    def _open_selector(_event):
        import tkinter as tk
        import tkinter.ttk as ttk
        popup = tk.Toplevel()
        popup.title("Select Strategy"); popup.configure(bg="#0d1117")
        popup.geometry("420x480")
        tk.Label(popup, text="Select a strategy:", bg="#0d1117",
                 fg=DIM, font=("Consolas", 10)).pack(pady=(10, 4))
        fv = tk.StringVar()
        ttk.Entry(popup, textvariable=fv, font=("Consolas", 10)).pack(
            fill=tk.X, padx=10)
        lb = tk.Listbox(popup, bg="#161b22", fg=TEXT,
                        selectbackground="#1f6feb", font=("Consolas", 10),
                        relief=tk.FLAT, activestyle="none", height=20)
        sb = tk.Scrollbar(popup, command=lb.yview)
        lb.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4))
        lb.pack(fill=tk.BOTH, expand=True, padx=(10, 0), pady=4)

        def _pop(flt=""):
            lb.delete(0, tk.END)
            for k in strat_keys:
                if flt.lower() in k.lower(): lb.insert(tk.END, k)
        fv.trace_add("write", lambda *_: _pop(fv.get())); _pop()

        def _confirm():
            sel = lb.curselection()
            if sel:
                _state["key"] = lb.get(sel[0])
                _strat_lbl.set_text(_state["key"])
                fig.canvas.draw_idle()
            popup.destroy()
        lb.bind("<Double-Button-1>", lambda _: _confirm())
        lb.bind("<Return>", lambda _: _confirm())
        tk.Button(popup, text="Select", command=_confirm,
                  bg="#1f6feb", fg="white", font=("Consolas", 10, "bold"),
                  relief=tk.FLAT).pack(pady=(4, 10))
        popup.lift(); popup.grab_set()

    btn_sel.on_clicked(_open_selector)

    # States buttons
    fig.text(0.30, CTRL_Y + CTRL_H * 0.75, "States:", color=DIM,
             fontsize=8.5, ha="left", va="center")
    _state_btns = {}
    for i, n in enumerate([2, 3, 4, 5]):
        x = 0.355 + i * 0.038
        btn_n, ax_n = _make_btn(x, 0.030, str(n), lambda _, n=n: _set_states(n))
        _state_btns[n] = (btn_n, ax_n)

    # Smooth textbox
    fig.text(0.515, CTRL_Y + CTRL_H * 0.75, "Smooth (days):", color=DIM,
             fontsize=8.5, ha="left", va="center")
    ax_tb = fig.add_axes([0.615, CTRL_Y, 0.05, CTRL_H])
    ax_tb.set_facecolor(PANEL)
    for sp in ax_tb.spines.values(): sp.set_edgecolor(AMBER)
    tb_smooth = TextBox(ax_tb, "", initial="21",
                        color=PANEL, hovercolor="#21262d", label_pad=0)
    tb_smooth.text_disp.set_color(TEXT); tb_smooth.text_disp.set_fontsize(9)

    # Status label
    _status = fig.text(0.68, CTRL_Y + CTRL_H * 0.5, "",
                       color=DIM, fontsize=7.5, ha="left", va="center",
                       fontfamily="monospace")

    # Run button
    btn_run, _ = _make_btn(0.88, 0.085, "\u25b6  Run", lambda _: _on_run())
    btn_run.label.set_color(GREEN); btn_run.label.set_fontsize(9)

    _last_run = {"key": strategy_name, "n_states": 3, "smooth": 21}

    def _refresh_state_btns():
        for n, (btn, ax_n) in _state_btns.items():
            c = AMBER if n == _state["n_states"] else PANEL
            ax_n.set_facecolor(c); btn.color = c

    _refresh_state_btns()

    def _set_states(n):
        _state["n_states"] = n
        _refresh_state_btns()
        fig.canvas.draw_idle()

    def _on_run(_event=None):
        try:
            smooth = max(1, int(tb_smooth.text.strip()))
        except ValueError:
            return
        _state["smooth"] = smooth
        key      = _state["key"]
        n_states = _state["n_states"]

        if (key == _last_run["key"] and
                n_states == _last_run["n_states"] and
                smooth == _last_run["smooth"]):
            return

        if reload_fn is None:
            return

        _status.set_text("Running…"); _status.set_color(AMBER)
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        try:
            new_report, new_name, new_pair = reload_fn(key, n_states, smooth)
        except Exception as exc:
            _status.set_text(f"Error: {exc}"); _status.set_color(RED)
            fig.canvas.draw_idle(); return

        _state["report"] = new_report
        _last_run.update({"key": key, "n_states": n_states, "smooth": smooth})
        _draw(new_report, new_name, new_pair)
        _status.set_text(f"Done — {n_states} states, smooth={smooth}d")
        _status.set_color(GREEN)
        fig.canvas.draw_idle()

    tb_smooth.on_submit(lambda _: _on_run())

    fig._regime_widgets = (btn_sel, btn_run, tb_smooth, _state_btns,
                           _state, _last_run, _status, _strat_lbl)

    plt.show(block=True)


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _plot_price_regimes(ax, report: RegimeReport) -> None:
    """Price series (relative, starts at 1) with regime bands."""
    ax.set_title("Asset Price — Regime Background", color=TEXT, fontsize=10, pad=4)
    ax.set_ylabel("Price (normalised)", color=DIM, fontsize=8)

    close_rel = np.exp(report.price_returns.cumsum())
    close_rel.name = "close"

    ax.plot(close_rel.index, close_rel.values, color=TEXT, lw=0.8, alpha=0.9)
    _shade_regimes(ax, report, close_rel.index)
    _add_regime_legend(ax, report)

    ax.tick_params(axis="x", labelsize=7.5, colors=DIM)
    ax.tick_params(axis="y", labelsize=7.5, colors=DIM)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    # Tighten x-axis
    ax.margins(x=0.01)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)


def _plot_equity_regimes(ax, report: RegimeReport) -> None:
    """Strategy equity curve with regime background + drawdown markers."""
    ax.set_title("Strategy Equity — Regime Shading + Drawdowns",
                 color=TEXT, fontsize=10, pad=4)
    ax.set_ylabel("Equity ($)", color=DIM, fontsize=8)

    eq = report.equity_curve
    ax.plot(eq.index, eq.values, color=BLUE, lw=1.2, alpha=0.95, zorder=3)
    _shade_regimes(ax, report, eq.index)

    for dd in report.drawdown_periods:
        if dd.depth_pct < 0.3:
            continue
        peak_eq   = eq.loc[:dd.peak_date].iloc[-1]
        trough_eq = eq.loc[dd.trough_date]
        color     = RED if dd.depth_pct > 5 else AMBER
        ax.vlines(dd.trough_date, trough_eq, peak_eq,
                  colors=color, lw=1.5, alpha=0.7, zorder=4)
        ax.scatter([dd.trough_date], [trough_eq],
                   marker="v", color=color, s=30, zorder=5)

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.tick_params(axis="x", labelsize=7.5, colors=DIM)
    ax.tick_params(axis="y", labelsize=7.5, colors=DIM)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    # Tighten x-axis
    ax.margins(x=0.01)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)


def _plot_regime_cumulative(ax, report: RegimeReport) -> None:
    """
    For each regime: grouped bars showing total cumulative return (%)
    of the strategy and the asset during that regime.

    This directly answers: "Does my strategy profit in each market state,
    and does it track the asset or move independently?"
    """
    ax.set_title("Cumulative Return per Regime  (Strategy vs Asset)",
                 color=TEXT, fontsize=10, pad=4)
    ax.set_ylabel("Total return in regime (%)", color=DIM, fontsize=8)

    # Recover daily strategy returns from equity curve
    eq = report.equity_curve
    strat_rets = eq.pct_change().fillna(0)

    # Asset returns — price_returns are log-returns; convert to simple returns
    asset_rets = np.expm1(report.price_returns)

    regime_labels = report.regime_labels

    names     = report.regime_names
    n         = len(names)
    x         = np.arange(n)
    w         = 0.35

    strat_totals = []
    asset_totals = []

    for i in range(n):
        mask = regime_labels.reindex(strat_rets.index, method="ffill") == i

        s_in = strat_rets[mask]
        a_in = asset_rets.reindex(strat_rets.index)[mask]

        # Compound the returns
        strat_cum = float((1 + s_in).prod() - 1) * 100
        asset_cum = float((1 + a_in).prod() - 1) * 100

        strat_totals.append(strat_cum)
        asset_totals.append(asset_cum)

    for i, (xi, sc, ac) in enumerate(zip(x, strat_totals, asset_totals)):
        c_strat = GREEN if sc >= 0 else RED
        c_asset = BLUE  if ac >= 0 else AMBER

        b_s = ax.bar(xi - w / 2, sc, width=w, color=c_strat, alpha=0.85,
                     edgecolor=BORDER, linewidth=0.5)
        b_a = ax.bar(xi + w / 2, ac, width=w, color=c_asset, alpha=0.60,
                     edgecolor=BORDER, linewidth=0.5)

        for bar, val in [(b_s, sc), (b_a, ac)]:
            yp = bar[0].get_height()
            va = "bottom" if yp >= 0 else "top"
            ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                    yp + (0.3 if yp >= 0 else -0.3),
                    f"{val:+.1f}%", ha="center", va=va,
                    fontsize=7, color=TEXT)

    ax.axhline(0, color=BORDER, lw=0.8, ls="-")

    rs = report.regime_stats
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{names[i]}\n({rs[i].n_days}d  ·  {rs[i].n_active} trades)"
         for i in range(n)],
        fontsize=8, color=DIM,
    )
    ax.tick_params(axis="y", labelsize=7.5, colors=DIM)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))

    s_patch = mpatches.Patch(color=GREEN, alpha=0.85, label="Strategy return")
    a_patch = mpatches.Patch(color=BLUE,  alpha=0.60, label="Asset return")
    ax.legend(handles=[s_patch, a_patch], fontsize=7.5, facecolor=PANEL,
              edgecolor=BORDER, labelcolor=DIM, loc="upper right")

    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)


def _plot_clustering(ax, report: RegimeReport) -> None:
    """Drawdown scatter + runs-test / ACF text box."""
    ax.set_title("Drawdown Clustering", color=TEXT, fontsize=10, pad=4)
    ax.set_xlabel("Duration (days peak→trough)", color=DIM, fontsize=8)
    ax.set_ylabel("Depth (%)", color=DIM, fontsize=8)

    dds = [d for d in report.drawdown_periods if d.depth_pct >= _MIN_SCATTER_DEPTH]
    if dds:
        regime_names = report.regime_names
        for dd in dds:
            try:
                ridx = regime_names.index(dd.dominant_regime) if dd.dominant_regime else 0
            except ValueError:
                ridx = 0
            color     = _REGIME_COLORS[ridx % len(_REGIME_COLORS)]
            recovered = dd.recovery_date is not None
            marker    = "o" if recovered else "x"
            ax.scatter(dd.duration_days, dd.depth_pct,
                       color=color, marker=marker, s=55, alpha=0.85,
                       edgecolors="none", zorder=4)
    else:
        ax.text(0.5, 0.6, "No significant drawdowns found",
                transform=ax.transAxes, ha="center", va="center",
                color=DIM, fontsize=9)

    rt  = report.runs_test
    lac = report.loss_acf
    info_lines = [
        f"Runs test:  z={rt.z_stat:+.2f}  p={rt.p_value:.3f}" if not np.isnan(rt.z_stat) else "Runs test:  n/a",
        f"   [{rt.verdict}]",
        "",
        f"Loss ACF:   lag1={lac.lag1_acf:+.3f}" if not np.isnan(lac.lag1_acf) else "Loss ACF:   n/a",
        f"   LB p={lac.ljung_box_pval:.3f}  [{lac.verdict}]" if not np.isnan(lac.ljung_box_pval) else f"   [{lac.verdict}]",
    ]
    ax.text(0.03, 0.03, "\n".join(info_lines),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=DIM, zorder=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                      edgecolor=BORDER, alpha=0.95, zorder=10))

    handles = []
    for i, name in enumerate(report.regime_names):
        handles.append(mpatches.Patch(
            color=_REGIME_COLORS[i % len(_REGIME_COLORS)], label=name))
    handles.append(plt.Line2D([0], [0], marker="x", color=DIM, linestyle="none",
                               markersize=6, label="Ongoing DD"))
    ax.legend(handles=handles, fontsize=7, facecolor=PANEL,
              edgecolor=BORDER, labelcolor=DIM, loc="upper left")

    ax.tick_params(axis="x", labelsize=7.5, colors=DIM)
    ax.tick_params(axis="y", labelsize=7.5, colors=DIM)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def _shade_regimes(ax, report: RegimeReport, date_index: pd.DatetimeIndex) -> None:
    labels = report.regime_labels.reindex(date_index, method="ffill").dropna()
    if labels.empty:
        return
    current_state = None
    start_date    = None
    for date, state in labels.items():
        state = int(state)
        if state != current_state:
            if current_state is not None:
                color = _REGIME_COLORS[current_state % len(_REGIME_COLORS)]
                ax.axvspan(start_date, date, color=color, alpha=0.08, lw=0)
            current_state = state
            start_date    = date
    if current_state is not None and start_date is not None:
        color = _REGIME_COLORS[current_state % len(_REGIME_COLORS)]
        ax.axvspan(start_date, labels.index[-1], color=color, alpha=0.08, lw=0)


def _add_regime_legend(ax, report: RegimeReport) -> None:
    handles = [
        mpatches.Patch(
            facecolor=_REGIME_COLORS[i % len(_REGIME_COLORS)],
            alpha=0.5, label=name,
        )
        for i, name in enumerate(report.regime_names)
    ]
    ax.legend(handles=handles, fontsize=7.5, facecolor=PANEL,
              edgecolor=BORDER, labelcolor=DIM,
              loc="upper left", framealpha=0.9)


# ── Explain button ────────────────────────────────────────────────────────────

def _add_explain_button(fig: plt.Figure) -> None:
    btn_ax = fig.add_axes([0.865, 0.960, 0.10, 0.022])
    btn_ax.set_facecolor(PANEL)
    for sp in btn_ax.spines.values(): sp.set_edgecolor(BORDER)
    btn = Button(btn_ax, "?  Explain", color=PANEL, hovercolor="#21262d")
    btn.label.set_color(DIM); btn.label.set_fontsize(8.5)

    def _open(_event):
        btn.label.set_color(AMBER); fig.canvas.draw_idle()
        _open_regime_explain_window()
        btn.label.set_color(DIM); fig.canvas.draw_idle()

    btn.on_clicked(_open)
    fig._regime_explain_btn = btn


# ── Explain window ────────────────────────────────────────────────────────────

def _open_regime_explain_window() -> None:
    import tkinter as tk

    win = tk.Toplevel()
    win.title("Regime Analysis — How to Read the Plots")
    win.configure(bg="#0d1117"); win.geometry("900x760"); win.resizable(True, True)

    frame = tk.Frame(win, bg="#0d1117"); frame.pack(fill=tk.BOTH, expand=True)
    scrollbar = tk.Scrollbar(frame, bg="#21262d", troughcolor="#0d1117")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    txt = tk.Text(frame, bg="#0d1117", fg="#e6edf3", font=("Consolas", 11),
                  yscrollcommand=scrollbar.set, wrap=tk.WORD, padx=28, pady=20,
                  borderwidth=0, highlightthickness=0, cursor="arrow",
                  state=tk.NORMAL)
    txt.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=txt.yview)

    txt.tag_configure("header",     font=("Consolas", 14, "bold"), foreground="#f0c060",
                      spacing1=18, spacing3=6)
    txt.tag_configure("section",    font=("Consolas", 10, "bold"), foreground="#8b949e",
                      spacing1=10, spacing3=2)
    txt.tag_configure("what",       font=("Consolas", 10), foreground="#c9d1d9",
                      spacing3=6, lmargin1=16, lmargin2=16)
    txt.tag_configure("want_item",  font=("Consolas", 10), foreground="#3fb950",
                      lmargin1=28, spacing3=1)
    txt.tag_configure("avoid_item", font=("Consolas", 10), foreground="#f85149",
                      lmargin1=28, spacing3=1)
    txt.tag_configure("divider",    font=("Consolas", 7), foreground="#21262d",
                      spacing1=14, spacing3=14)

    txt.insert(tk.END, "Regime Analysis — How to Read the Plots\n", "header")
    txt.insert(tk.END, "\u2500" * 84 + "\n", "divider")

    guides = [
        ("Asset Price", "top row (full width)",
         "The asset close price (normalised to start at 1) with colour-coded background "
         "bands. Each colour = one HMM regime. Fitted on weekly returns + 4-week rolling "
         "volatility. States sorted by mean weekly return (Bull = best, Bear = worst).",
         ["Stable, long continuous bands", "Bull regime covering most of the sample"],
         ["Fragmented bands (HMM overfit)", "All regimes same proportion — try fewer states"]),
        ("Strategy Equity", "second row (full width)",
         "Cumulative strategy equity with the same regime bands. Drawdown episodes are "
         "marked: red lines = deep (>10%), amber = moderate. Triangle = trough date. "
         "Ask: do my drawdowns only happen in Bear, or are they spread across all regimes?",
         ["Equity growing in ALL coloured regions", "Drawdowns spread across regimes"],
         ["Equity flat/declining in one dominant regime", "All large drawdowns in Bear"]),
        ("Cumulative Return per Regime", "bottom-left",
         "For each regime, two bars: the total compounded return of the strategy (left, "
         "green/red) and the asset (right, blue/amber) during ALL days assigned to that "
         "regime. This directly answers: 'Does my strategy make money in each market state, "
         "and is it tracking the asset or acting independently?'\n\n"
         "Strategy bar green + Asset bar red = strategy finds alpha in bear regimes.\n"
         "Both bars green = strategy profitable but tracks the asset (possible beta).\n"
         "Strategy bar taller = strategy amplifies gains; shorter = it dampens them.",
         ["Strategy bar positive in ALL regimes", "Strategy bar taller than asset in bull"],
         ["Strategy positive ONLY in Bull — regime-dependent edge",
          "Strategy and asset bars nearly identical — pure beta vehicle"]),
        ("Drawdown Clustering", "bottom-right",
         "Scatter: each dot = one drawdown episode, positioned by duration (x) and depth "
         "(y). Colour = dominant regime during that drawdown. Circle = recovered. Cross = "
         "ongoing. Text box: Runs test (z, p) and lag-1 autocorrelation of returns.",
         ["Dots spread across multiple regime colours", "Runs test RANDOM (p>0.05)"],
         ["All deep dots in Bear", "Runs test CLUSTERED", "Many crosses (unrecovered)"]),
    ]

    for title, pos, what, want, avoid in guides:
        txt.insert(tk.END, f"  {title}  [{pos}]\n", "header")
        txt.insert(tk.END, "What it shows\n", "section")
        txt.insert(tk.END, what + "\n", "what")
        txt.insert(tk.END, "Want to see\n", "section")
        for item in want: txt.insert(tk.END, f"  \u2022  {item}\n", "want_item")
        txt.insert(tk.END, "Watch out for\n", "section")
        for item in avoid: txt.insert(tk.END, f"  \u2022  {item}\n", "avoid_item")
        txt.insert(tk.END, "\u2500" * 84 + "\n", "divider")

    txt.config(state=tk.DISABLED); txt.see("1.0")
    win.lift()
