"""
plot_rolling_alpha.py — Rolling alpha/beta overlay on asset price + equity curve.

Three stacked panels sharing the same time axis:
  Top:    Asset close price
  Middle: Strategy equity curve
  Bottom: Rolling annualised alpha (bars) + rolling beta (dashed, right axis)

Controls at the bottom:
  - Strategy selector button (opens a Tk listbox popup)
  - Lookback and Step text inputs
  - Run button (redraws all three panels with the selected strategy + params)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, TextBox
import numpy as np
import pandas as pd

from .rolling_alpha import compute_rolling_alpha

# ── Colour palette ─────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
DIM    = "#8b949e"
AMBER  = "#f0c060"
GREEN  = "#3fb950"
RED    = "#f85149"
BLUE   = "#58a6ff"


# ── Public entry point ─────────────────────────────────────────────────────────

def plot_rolling_alpha(
    all_strategies:  dict,
    load_strategy,           # callable: key -> (aligned_df, price_df, pair_str)
    initial_key:     str,
    lookback:        int   = 252,
    step:            int   = 22,
    initial_capital: float = 10_000.0,
) -> None:
    """
    Open the rolling alpha figure with strategy switcher and parameter controls.

    Args:
        all_strategies:  {key: trades_df} — full strategy dict for the dropdown.
        load_strategy:   callable(key) -> (aligned, price_df, pair).
        initial_key:     Strategy to display on first open.
        lookback:        Initial OLS trailing window (trading days).
        step:            Initial step between OLS windows (trading days).
        initial_capital: Starting equity for the equity curve.
    """
    strategy_keys = list(all_strategies.keys())

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor=BG)

    gs = gridspec.GridSpec(
        3, 1,
        figure        = fig,
        height_ratios = [1, 1, 1.4],
        hspace        = 0.10,
        top           = 0.918,
        bottom        = 0.13,
        left          = 0.06,
        right         = 0.94,
    )

    ax_price  = fig.add_subplot(gs[0])
    ax_equity = fig.add_subplot(gs[1], sharex=ax_price)
    ax_alpha  = fig.add_subplot(gs[2], sharex=ax_price)

    for ax in (ax_price, ax_equity, ax_alpha):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(axis="y", color=BORDER, lw=0.4, alpha=0.5)
        ax.grid(axis="x", color=BORDER, lw=0.3, alpha=0.3)

    plt.setp(ax_price.get_xticklabels(),  visible=False)
    plt.setp(ax_equity.get_xticklabels(), visible=False)

    # Persistent title text objects (updated on rerun)
    suptitle = fig.suptitle("", color=TEXT, fontsize=13, fontweight="bold",
                             x=0.5, y=0.975)
    subtitle = fig.text(0.5, 0.945,
        r"OLS per window:  $r_{strategy} = \alpha + \beta \cdot r_{asset}$   "
        "(residuals excluded)",
        ha="center", va="top", color=DIM, fontsize=8.5)

    # ── Mutable state ──────────────────────────────────────────────────────────
    state = {
        "key":      initial_key,
        "lookback": lookback,
        "step":     step,
        "aligned":  None,
        "price_df": None,
        "pair":     "",
    }
    beta_ax_ref = [None]

    # ── Draw helpers ───────────────────────────────────────────────────────────

    def _draw_all() -> None:
        key      = state["key"]
        lb       = state["lookback"]
        st       = state["step"]
        aligned  = state["aligned"]
        price_df = state["price_df"]
        pair     = state["pair"]

        close  = price_df["Close"].sort_index()
        equity = initial_capital * (1 + aligned["strategy"]).cumprod()

        label = f"{key}  ·  {pair}" if pair else key
        suptitle.set_text(f"Rolling Alpha / Beta  —  {label}")

        # ── Price panel ───────────────────────────────────────────────────────
        ax_price.cla()
        ax_price.set_facecolor(PANEL)
        for sp in ax_price.spines.values():
            sp.set_edgecolor(BORDER)
        ax_price.grid(axis="y", color=BORDER, lw=0.4, alpha=0.5)
        ax_price.grid(axis="x", color=BORDER, lw=0.3, alpha=0.3)
        ax_price.plot(close.index, close.values, color=BLUE, lw=1.1, zorder=3)
        ax_price.set_ylabel(f"{pair} Close" if pair else "Close",
                            color=DIM, fontsize=9)
        ax_price.set_title("Asset Price", color=TEXT, fontsize=10, pad=3)
        ax_price.tick_params(axis="y", labelsize=7.5, colors=DIM)
        plt.setp(ax_price.get_xticklabels(), visible=False)

        # ── Equity panel ──────────────────────────────────────────────────────
        ax_equity.cla()
        ax_equity.set_facecolor(PANEL)
        for sp in ax_equity.spines.values():
            sp.set_edgecolor(BORDER)
        ax_equity.grid(axis="y", color=BORDER, lw=0.4, alpha=0.5)
        ax_equity.grid(axis="x", color=BORDER, lw=0.3, alpha=0.3)
        ax_equity.plot(equity.index, equity.values, color=GREEN, lw=1.1, zorder=3)
        ax_equity.set_ylabel("Equity ($)", color=DIM, fontsize=9)
        ax_equity.set_title("Strategy Equity", color=TEXT, fontsize=10, pad=3)
        ax_equity.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
        )
        ax_equity.tick_params(axis="y", labelsize=7.5, colors=DIM)
        plt.setp(ax_equity.get_xticklabels(), visible=False)

        # ── Alpha panel ───────────────────────────────────────────────────────
        if beta_ax_ref[0] is not None:
            try:
                beta_ax_ref[0].remove()
            except Exception:
                pass
            beta_ax_ref[0] = None

        ax_alpha.cla()
        ax_alpha.set_facecolor(PANEL)
        for sp in ax_alpha.spines.values():
            sp.set_edgecolor(BORDER)
        ax_alpha.grid(axis="y", color=BORDER, lw=0.4, alpha=0.5)
        ax_alpha.grid(axis="x", color=BORDER, lw=0.3, alpha=0.3)
        ax_alpha.set_title("Rolling Alpha & Beta", color=TEXT, fontsize=10, pad=3)
        ax_alpha.set_ylabel("Ann. Alpha (%)", color=TEXT, fontsize=9)
        ax_alpha.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:+.1f}%")
        )
        ax_alpha.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_alpha.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_alpha.tick_params(axis="x", labelsize=8, colors=DIM)
        ax_alpha.tick_params(axis="y", labelsize=7.5, colors=DIM)

        roll = compute_rolling_alpha(aligned, lookback=lb, step=st)
        if roll.empty:
            ax_alpha.text(0.5, 0.5, "Not enough data",
                          transform=ax_alpha.transAxes,
                          ha="center", color=DIM, fontsize=10)
        else:
            bar_width = pd.Timedelta(days=st * 0.85)
            sig   = roll["alpha_tstat"].abs() >= 2.0
            insig = ~sig

            for mask, opac in [(sig, 0.82), (insig, 0.25)]:
                if not mask.any():
                    continue
                sub    = roll[mask]
                colors = [GREEN if v >= 0 else RED for v in sub["alpha_ann"]]
                ax_alpha.bar(sub.index, sub["alpha_ann"] * 100,
                             width=bar_width, color=colors, alpha=opac, zorder=3)

            ax_alpha.axhline(0, color=TEXT, lw=0.8, zorder=4)

            ax_beta = ax_alpha.twinx()
            beta_ax_ref[0] = ax_beta
            ax_beta.set_facecolor(PANEL)
            for sp in ax_beta.spines.values():
                sp.set_edgecolor(BORDER)
            ax_beta.plot(roll.index, roll["beta"],
                         color=AMBER, lw=1.4, ls="--", zorder=5)
            ax_beta.axhline(0, color=AMBER, lw=0.5, ls=":", alpha=0.4)
            ax_beta.set_ylabel("Beta", color=AMBER, fontsize=9)
            ax_beta.tick_params(axis="y", colors=AMBER, labelsize=8)

            ax_alpha.text(
                0.01, 0.97,
                f"lookback = {lb}d  ·  step = {st}d",
                transform=ax_alpha.transAxes, ha="left", va="top",
                color=DIM, fontsize=7.5,
            )

            legend_elements = [
                Patch(facecolor=GREEN, alpha=0.82, label="α > 0  |t| ≥ 2"),
                Patch(facecolor=GREEN, alpha=0.25, label="α > 0  |t| < 2"),
                Patch(facecolor=RED,   alpha=0.82, label="α < 0  |t| ≥ 2"),
                Patch(facecolor=RED,   alpha=0.25, label="α < 0  |t| < 2"),
                Line2D([0], [0], color=AMBER, lw=1.4, ls="--",
                       label="Beta (right axis)"),
            ]
            ax_alpha.legend(
                handles=legend_elements, loc="upper right",
                fontsize=7.5, framealpha=0.35,
                facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
            )

        fig.canvas.draw_idle()

    # ── Load initial strategy ──────────────────────────────────────────────────
    def _load_and_draw():
        key = state["key"]
        print(f"  Loading: {key} ...", flush=True)
        aligned, price_df, pair = load_strategy(key)
        state["aligned"]  = aligned
        state["price_df"] = price_df
        state["pair"]     = pair
        _draw_all()

    _load_and_draw()

    # ── Control bar ────────────────────────────────────────────────────────────

    # Current strategy display label
    strat_label = fig.text(
        0.01, 0.100, initial_key,
        color=AMBER, fontsize=8.5, ha="left", va="center",
        clip_on=True,
    )
    status_text = fig.text(
        0.01, 0.078, "",
        color=DIM, fontsize=7.5, ha="left", va="center",
    )

    # Lookback label + textbox
    fig.text(0.68, 0.068, "Lookback (days):", color=DIM,
             fontsize=8.5, ha="right", va="center")
    ax_tb_lb = fig.add_axes([0.680, 0.050, 0.07, 0.034])
    ax_tb_lb.set_facecolor(PANEL)
    for sp in ax_tb_lb.spines.values():
        sp.set_edgecolor(BORDER)
    tb_lookback = TextBox(ax_tb_lb, "", initial=str(lookback),
                          color=PANEL, hovercolor="#21262d", label_pad=0.02)
    tb_lookback.text_disp.set_color(TEXT)
    tb_lookback.text_disp.set_fontsize(9)

    # Step label + textbox
    fig.text(0.84, 0.068, "Step (days):", color=DIM,
             fontsize=8.5, ha="right", va="center")
    ax_tb_st = fig.add_axes([0.840, 0.050, 0.05, 0.034])
    ax_tb_st.set_facecolor(PANEL)
    for sp in ax_tb_st.spines.values():
        sp.set_edgecolor(BORDER)
    tb_step = TextBox(ax_tb_st, "", initial=str(step),
                      color=PANEL, hovercolor="#21262d", label_pad=0.02)
    tb_step.text_disp.set_color(TEXT)
    tb_step.text_disp.set_fontsize(9)

    # Run button
    ax_run = fig.add_axes([0.900, 0.050, 0.07, 0.034])
    ax_run.set_facecolor(PANEL)
    for sp in ax_run.spines.values():
        sp.set_edgecolor(BORDER)
    btn_run = Button(ax_run, "▶  Run", color=PANEL, hovercolor="#21262d")
    btn_run.label.set_color(GREEN)
    btn_run.label.set_fontsize(9)

    _last_key = {"v": initial_key}

    def _reload_strategy(key: str) -> None:
        """Load data for key and redraw. Updates strat_label and status_text."""
        status_text.set_text("Loading…")
        fig.canvas.draw_idle()
        aligned, price_df, pair = load_strategy(key)
        state["aligned"]  = aligned
        state["price_df"] = price_df
        state["pair"]     = pair
        _last_key["v"]    = key
        strat_label.set_text(key)
        status_text.set_text("")
        _draw_all()

    def _on_run(_event=None):
        try:
            lb = max(30, int(tb_lookback.text.strip()))
            st = max(1,  int(tb_step.text.strip()))
        except ValueError:
            return
        state["lookback"] = lb
        state["step"]     = st
        if state["key"] != _last_key["v"]:
            _reload_strategy(state["key"])
        else:
            _draw_all()

    btn_run.on_clicked(_on_run)
    tb_lookback.on_submit(lambda _: _on_run())
    tb_step.on_submit(lambda _: _on_run())

    # Selector button — defined AFTER _reload_strategy so _confirm can call it
    ax_sel = fig.add_axes([0.01, 0.050, 0.25, 0.034])
    ax_sel.set_facecolor(PANEL)
    for sp in ax_sel.spines.values():
        sp.set_edgecolor(BORDER)
    btn_sel = Button(ax_sel, "▾  Select Strategy", color=PANEL, hovercolor="#21262d")
    btn_sel.label.set_color(BLUE)
    btn_sel.label.set_fontsize(8.5)

    def _open_selector(_event):
        import tkinter as tk
        from tkinter import ttk

        # Under TkAgg the canvas manager owns the Tk root — use it as parent
        try:
            root = fig.canvas.manager.window
        except Exception:
            root = None

        popup = tk.Toplevel(root)
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
                if listbox.get(i) == state["key"]:
                    listbox.selection_set(i)
                    listbox.see(i)
                    break

        filter_var.trace_add("write", lambda *_: _populate(filter_var.get()))
        _populate()

        def _confirm():
            sel = listbox.curselection()
            if sel:
                chosen = listbox.get(sel[0])
                state["key"] = chosen
                popup.destroy()
                # Auto-reload immediately — no need to click Run
                _reload_strategy(chosen)
            else:
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
        popup.focus_force()

    btn_sel.on_clicked(_open_selector)

    # Explain button
    ax_exp = fig.add_axes([0.865, 0.970, 0.10, 0.022])
    ax_exp.set_facecolor(PANEL)
    for sp in ax_exp.spines.values():
        sp.set_edgecolor(BORDER)
    btn_exp = Button(ax_exp, "?  Explain", color=PANEL, hovercolor="#21262d")
    btn_exp.label.set_color(DIM)
    btn_exp.label.set_fontsize(8.5)

    def _on_explain(_event):
        btn_exp.label.set_color(AMBER)
        fig.canvas.draw_idle()
        _open_explain_window()
        btn_exp.label.set_color(DIM)
        fig.canvas.draw_idle()

    btn_exp.on_clicked(_on_explain)

    # Keep widget references alive
    fig._ra_widgets = (tb_lookback, tb_step, btn_run, btn_sel, btn_exp,
                       state, beta_ax_ref, _last_key, strat_label, status_text)

    plt.show(block=True)


# ── Explain window ─────────────────────────────────────────────────────────────

_GUIDES = [
    {
        "number":   "01",
        "title":    "Asset Price",
        "position": "top panel",
        "what": (
            "The raw close price of the traded instrument over the full history. "
            "Use it as a visual reference to identify bull periods (sustained uptrend), "
            "bear periods (sustained downtrend), and ranging periods (sideways). "
            "Then look at the alpha panel: does alpha rise, fall, or stay flat during "
            "each of those phases?"
        ),
        "formula": "Close_t  (raw price, no transformation)",
        "variables": [("Blue line", "Asset close price over time.")],
        "want":  ["Clear, distinct bull/bear/ranging phases to map against alpha"],
        "avoid": ["Nothing to avoid — reference panel only"],
    },
    {
        "number":   "02",
        "title":    "Strategy Equity",
        "position": "middle panel",
        "what": (
            "Cumulative equity from compounding daily strategy returns. Compare its "
            "shape with the price panel above: a strategy growing while the asset "
            "is flat or declining has genuine alpha. A strategy only growing when the "
            "asset rises is likely levered beta."
        ),
        "formula": "Equity_t  =  Capital₀ × ∏(1 + r_strategy_s,  s = 1..t)",
        "variables": [("Green line", "Cumulative strategy equity ($).")],
        "want": [
            "Equity growing regardless of asset price direction",
            "Shallow drawdowns, quickly recovered",
        ],
        "avoid": [
            "Equity mirroring asset price — returns driven by beta",
            "Large drawdowns coinciding with asset bear markets",
        ],
    },
    {
        "number":   "03",
        "title":    "Rolling Alpha & Beta",
        "position": "bottom panel",
        "what": (
            "For each date, OLS is fitted on the trailing lookback window:\n\n"
            "    r_strategy = α + β · r_asset\n\n"
            "Alpha (bars) = market-independent return above passive exposure. "
            "Beta (dashed) = how much the strategy moves with the asset. "
            "Residuals are excluded — only α and β are shown.\n\n"
            "Dim bars (25% opacity) have |t(α)| < 2: not statistically significant. "
            "Bright bars (82%) have |t| ≥ 2: significant.\n\n"
            "Use the Lookback/Step controls and Run to update the chart."
        ),
        "formula": (
            "α̂ = ȳ − β̂x̄     β̂ = Cov(r_strat, r_asset) / Var(r_asset)\n"
            "t(α) = α̂ / SE(α̂)"
        ),
        "variables": [
            ("Green bar (bright)",  "Positive alpha, significant |t| ≥ 2."),
            ("Green bar (dim)",     "Positive alpha, NOT significant |t| < 2."),
            ("Red bar (bright)",    "Negative alpha, significant."),
            ("Red bar (dim)",       "Negative alpha, NOT significant."),
            ("Amber dashed line",   "Rolling beta (right axis). 0 = market-neutral."),
            ("Lookback",            "Trailing days used per OLS fit."),
            ("Step",                "Days between windows (22 ≈ 1 month)."),
        ],
        "want": [
            "Consistently bright green bars across full history",
            "Beta near zero throughout — not a disguised long/short",
            "Alpha positive even during asset bear markets",
        ],
        "avoid": [
            "Alpha only bright during bull markets — disguised beta",
            "All bars dim — try a longer lookback",
            "Alpha flipping sign repeatedly — unstable edge",
        ],
    },
]

_OVERALL = (
    "What this figure answers\n"
    "─────────────────────────\n"
    "Is my strategy's profitability driven by skill (alpha) or passive market "
    "exposure (beta) — and does it matter which regime we're in?\n\n"
    "OLS decomposition per window:\n"
    "  α  — return that exists even if the market goes nowhere\n"
    "  β  — return that is just riding the asset direction\n"
    "  ε  — unexplained noise (excluded)\n\n"
    "A robust strategy should have:\n"
    "  • α > 0 and significant across ALL market phases\n"
    "  • β near zero throughout\n"
    "  • No correlation between alpha and asset trend\n\n"
    "Lookback sensitivity:\n"
    "  • Short lookback (63d)  → responsive, noisy\n"
    "  • Long lookback (252d)  → stable, slow to detect shifts\n"
    "  • Step = 22 ≈ monthly resolution, good cost/accuracy trade-off"
)


def _open_explain_window() -> None:
    import tkinter as tk

    win = tk.Toplevel()
    win.title("Rolling Alpha / Beta — How to Read the Plots")
    win.configure(bg="#0d1117")
    win.geometry("900x760")
    win.resizable(True, True)

    frame = tk.Frame(win, bg="#0d1117")
    frame.pack(fill=tk.BOTH, expand=True)

    sb = tk.Scrollbar(frame, bg="#21262d", troughcolor="#0d1117",
                      activebackground="#30363d")
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    txt = tk.Text(
        frame, bg="#0d1117", fg="#e6edf3",
        font=("Consolas", 11), wrap=tk.WORD, relief=tk.FLAT,
        padx=18, pady=14, spacing1=2, spacing2=3,
        yscrollcommand=sb.set, state=tk.NORMAL,
    )
    txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.config(command=txt.yview)

    txt.tag_configure("header",      foreground="#58a6ff", font=("Consolas", 14, "bold"))
    txt.tag_configure("section",     foreground="#f0c060", font=("Consolas", 11, "bold"))
    txt.tag_configure("what",        foreground="#e6edf3", font=("Consolas", 10))
    txt.tag_configure("formula_box", foreground="#3fb950", font=("Consolas", 10),
                      background="#0d2010", lmargin1=20, lmargin2=20)
    txt.tag_configure("var_name",    foreground="#f0c060", font=("Consolas", 10, "bold"))
    txt.tag_configure("var_desc",    foreground="#8b949e", font=("Consolas", 10))
    txt.tag_configure("want_item",   foreground="#3fb950", font=("Consolas", 10))
    txt.tag_configure("avoid_item",  foreground="#f85149", font=("Consolas", 10))
    txt.tag_configure("divider",     foreground="#30363d", font=("Consolas", 10))
    txt.tag_configure("overall",     foreground="#e6edf3", font=("Consolas", 10))

    txt.insert(tk.END, "Rolling Alpha / Beta — Plot Guide\n\n", "header")

    for g in _GUIDES:
        txt.insert(tk.END,
                   f"Plot {g['number']}  ·  {g['title']}  ({g['position']})\n",
                   "section")
        txt.insert(tk.END, "─" * 70 + "\n", "divider")
        txt.insert(tk.END, "\nWhat it shows\n", "section")
        txt.insert(tk.END, g["what"] + "\n", "what")
        txt.insert(tk.END, "\nFormula\n", "section")
        txt.insert(tk.END, "  " + g["formula"] + "\n", "formula_box")
        txt.insert(tk.END, "\nVariables\n", "section")
        for name, desc in g["variables"]:
            txt.insert(tk.END, f"  {name:<28}", "var_name")
            txt.insert(tk.END, f"  {desc}\n", "var_desc")
        txt.insert(tk.END, "\nWhat you want to see\n", "section")
        for item in g["want"]:
            txt.insert(tk.END, f"  ✓  {item}\n", "want_item")
        txt.insert(tk.END, "\nWhat to watch out for\n", "section")
        for item in g["avoid"]:
            txt.insert(tk.END, f"  ✗  {item}\n", "avoid_item")
        txt.insert(tk.END, "\n\n", "what")

    txt.insert(tk.END, "─" * 70 + "\n", "divider")
    txt.insert(tk.END, _OVERALL + "\n", "overall")
    txt.config(state=tk.DISABLED)
    win.lift()
