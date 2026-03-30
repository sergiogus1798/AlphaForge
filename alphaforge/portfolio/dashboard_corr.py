"""
dashboard_corr.py — Deep Correlation Analysis dashboard.

Analyses the strategies within a selected portfolio combination:

  Panel 1 (top-left)    : Pearson correlation matrix heatmap
                          Diagonal cells show the weight for the selected method.
  Panel 2 (top-right)   : Rolling pairwise Pearson correlations over time.
                          ≤10 pairs → line chart;  >10 pairs → time×pair heatmap.
  Panel 3 (bottom-left) : PCA scree plot (bar = explained var, line = cumulative).
  Panel 4 (bottom-right): PCA loadings heatmap  (component × strategy).

Selectors (right margin):
  RadioButtons — switch between the 4 weighting methods.
  < Prev / Next > buttons — navigate between combinations.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, RadioButtons

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.combo_result import (
    CombinationResult, METHODS, METHOD_COLORS, METHOD_LABELS,
)
from alphaforge.portfolio.generator.filters import (
    _monthly_pnl, _rolling_correlations,
)


# ── Theme ──────────────────────────────────────────────────────────────────────

BG     = "#0f1117"
AX_BG  = "#1a1a2e"
BORDER = "#2a2a4a"
CYAN   = "#00d4ff"
AMBER  = "#f0a500"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
TEXT   = "#e0e0e0"
DIM    = "#888888"

# PCA panels use a light background so tick labels remain readable
PCA_BG   = "#eef0f8"
PCA_TEXT = "#1a1a2a"
PCA_GRID = "#b0b4cc"


# ── Data helpers ───────────────────────────────────────────────────────────────

def _daily_pnl_combo(
    combination: tuple[str, ...],
    strategies: dict,
) -> pd.DataFrame:
    """
    Build an aligned daily P&L matrix for the strategies in one combination.
    Rows = calendar days that had at least one trade; columns = strategy names.
    """
    series: dict[str, pd.Series] = {}
    for name in combination:
        df    = strategies[name]
        daily = df.groupby(df["Close time"].dt.date)["Profit/Loss"].sum()
        daily.index = pd.DatetimeIndex(daily.index)
        series[name] = daily

    mat = pd.DataFrame(series).fillna(0.0)
    return mat.loc[(mat != 0).any(axis=1)]


def _short(name: str) -> str:
    return name.split("/")[-1]


def _compute_pca(
    pnl_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    PCA via numpy SVD on daily P&L matrix.

    Returns:
        explained   : variance explained per component (%)
        cumulative  : cumulative explained variance (%)
        loadings_df : DataFrame (components × strategies)
    """
    n_comp = min(len(pnl_df.columns), len(pnl_df))
    X      = pnl_df.values - pnl_df.values.mean(axis=0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    var       = s[:n_comp] ** 2
    explained  = var / var.sum() * 100
    cumulative = np.cumsum(explained)

    short_labels = [_short(c) for c in pnl_df.columns]
    loadings_df  = pd.DataFrame(
        Vt[:n_comp, :],
        index=[f"PC{i + 1}" for i in range(n_comp)],
        columns=short_labels,
    )
    return explained, cumulative, loadings_df


# ── Axis styling ───────────────────────────────────────────────────────────────

def _style(ax) -> None:
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT, labelsize=7)


def _short_pair(a: str, b: str) -> str:
    """'AUDJPY/Strategy 4.18.162' × 'GBPJPY/Strategy 1.10.137' → '4.18×1.10'"""
    def nums(name: str) -> str:
        parts = name.split("/")[-1].replace("Strategy ", "").split(".")
        return ".".join(parts[:2])
    return f"{nums(a)}×{nums(b)}"


def _add_colorbar(fig, im, ax, **kw):
    """Add a colorbar, safely removing any previous one attached to this ax."""
    if hasattr(ax, "_cb"):
        try:
            ax._cb.remove()
        except Exception:
            pass
    ax._cb = fig.colorbar(im, ax=ax, **kw)
    return ax._cb


# ── Panel 1: Correlation matrix ────────────────────────────────────────────────

def _draw_corr_matrix(
    fig,
    ax,
    pnl_df: pd.DataFrame,
    weights: dict | None,
    method_label: str,
) -> None:
    ax.cla()
    _style(ax)

    labels = [_short(c) for c in pnl_df.columns]
    n      = len(labels)
    corr   = pnl_df.corr(method="pearson").values

    im = ax.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", color=TEXT, fontsize=7)
    ax.set_yticklabels(labels, color=TEXT, fontsize=7)
    ax.set_title(
        f"Pearson Correlation  ·  weights: {method_label}",
        color=TEXT, fontsize=9,
    )

    cols = list(pnl_df.columns)
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if i == j and weights:
                w   = weights.get(cols[i], 0.0)
                txt = f"w={w:.2f}"
                tc  = CYAN
            else:
                txt = f"{val:.2f}"
                tc  = "#111111" if abs(val) < 0.40 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=5.5, color=tc)

    _add_colorbar(fig, im, ax, fraction=0.046, pad=0.04)


# ── Panel 2: Rolling pairwise correlations ─────────────────────────────────────

def _draw_rolling_corrs(
    fig,
    ax,
    combination: tuple[str, ...],
    strategies: dict,
    window_months: int,
    threshold: float = 0.35,
) -> None:
    """
    Line chart: one line per pair.
    Pairs that never breach the threshold → thin gray.
    Pairs that breach at least once → colored + thicker, labelled.
    """
    ax.cla()
    _style(ax)

    pairs = list(itertools.combinations(combination, 2))

    rolling: dict[tuple, pd.Series] = {}
    for a, b in pairs:
        roll = _rolling_correlations(
            _monthly_pnl(strategies[a]),
            _monthly_pnl(strategies[b]),
            window_months,
        )
        if not roll.empty:
            rolling[(a, b)] = roll["pearson"]

    if not rolling:
        ax.set_title("Rolling Correlations — insufficient data", color=DIM, fontsize=9)
        return

    # Separate breaching vs clean pairs
    breaching = {k: v for k, v in rolling.items() if v.abs().max() > threshold}
    clean     = {k: v for k, v in rolling.items() if v.abs().max() <= threshold}

    # Draw clean pairs first (background, thin, gray, no label)
    for (a, b), series in clean.items():
        x = [p.to_timestamp() for p in series.index]
        ax.plot(x, series.values, color="#444466", lw=0.7, alpha=0.45)

    # Draw breaching pairs on top with distinct colors and labels
    try:
        cmap = plt.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")

    for k, ((a, b), series) in enumerate(breaching.items()):
        x   = [p.to_timestamp() for p in series.index]
        lbl = _short_pair(a, b)
        ax.plot(x, series.values, color=cmap(k % 10), lw=1.6, alpha=0.9, label=lbl)

    # Threshold lines
    ax.axhline( threshold, color=RED, lw=1.0, ls="--", alpha=0.7,
                label=f"+{threshold} threshold")
    ax.axhline(-threshold, color=RED, lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0, color=BORDER, lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Pearson r", color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.grid(True, color=BORDER, lw=0.4)

    n_breach = len(breaching)
    ax.set_title(
        f"Rolling {window_months}m Pearson  ·  {len(pairs)} pairs  "
        f"({n_breach} breach >{threshold})",
        color=TEXT, fontsize=9,
    )

    if breaching:
        ax.legend(
            fontsize=6.5, facecolor=AX_BG, labelcolor=TEXT,
            loc="upper left", framealpha=0.85,
            ncol=max(1, n_breach // 6),
        )


# ── Panels 3 & 4: PCA ─────────────────────────────────────────────────────────

def _draw_pca(
    fig,
    ax_scree,
    ax_load,
    pnl_df: pd.DataFrame,
) -> None:
    explained, cumulative, loadings = _compute_pca(pnl_df)
    n_show     = min(len(explained), 6)
    components = np.arange(1, n_show + 1)

    # ── Scree plot ─────────────────────────────────────────────────────────────
    ax_scree.cla()
    ax_scree.set_facecolor(PCA_BG)
    for sp in ax_scree.spines.values():
        sp.set_edgecolor(PCA_GRID)
    ax_scree.tick_params(colors=PCA_TEXT, labelsize=7)

    ax_scree.bar(components, explained[:n_show], color=CYAN, alpha=0.82, zorder=2)
    ax_scree.set_xlabel("Component", color=PCA_TEXT, fontsize=8)
    ax_scree.set_ylabel("Expl. Var %", color="#0080aa", fontsize=8)
    ax_scree.set_title("PCA — Scree", color=PCA_TEXT, fontsize=9)
    ax_scree.set_xticks(components)
    ax_scree.grid(True, color=PCA_GRID, lw=0.4, zorder=0)

    ax2 = ax_scree.twinx()
    ax2.plot(components, cumulative[:n_show],
             color=AMBER, lw=2, marker="o", markersize=4, zorder=3)
    ax2.axhline(80, color="#999999", ls="--", lw=0.8, alpha=0.7)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Cumul. %", color=AMBER, fontsize=7.5)
    ax2.tick_params(colors=AMBER, labelsize=7)
    ax2.patch.set_visible(False)   # keep twinx transparent so bars show through

    # Annotate bars with explained %
    for i, v in enumerate(explained[:n_show]):
        ax_scree.text(i + 1, v + 0.8, f"{v:.1f}%",
                      ha="center", va="bottom", color=PCA_TEXT, fontsize=6.5)

    # ── Loadings heatmap ───────────────────────────────────────────────────────
    ax_load.cla()
    ax_load.set_facecolor(PCA_BG)
    for sp in ax_load.spines.values():
        sp.set_edgecolor(PCA_GRID)
    ax_load.tick_params(colors=PCA_TEXT, labelsize=7)

    load_data = loadings.iloc[:n_show, :].values
    n_strat   = load_data.shape[1]

    im = ax_load.imshow(load_data, cmap="RdYlGn_r",
                        vmin=-1, vmax=1, aspect="auto")
    ax_load.set_yticks(range(n_show))
    ax_load.set_yticklabels(loadings.index[:n_show], color=PCA_TEXT, fontsize=7)
    ax_load.set_xticks(range(n_strat))
    ax_load.set_xticklabels(
        loadings.columns, rotation=40, ha="right", color=PCA_TEXT, fontsize=7,
    )
    ax_load.set_title("PCA Loadings  (component × strategy)", color=PCA_TEXT, fontsize=9)

    for i in range(n_show):
        for j in range(n_strat):
            v = load_data[i, j]
            # RdYlGn_r: near ±1 = dark cell → white text; near 0 = yellow = use dark text
            tc = "#111111" if abs(v) < 0.40 else "white"
            ax_load.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=5.5, color=tc)

    _add_colorbar(fig, im, ax_load, fraction=0.046, pad=0.04)


# ── Full render ────────────────────────────────────────────────────────────────

def _render_corr(
    fig,
    ax_corr, ax_roll, ax_scree, ax_load,
    nav_label,
    cr: CombinationResult,
    idx: int,
    n: int,
    strategies: dict,
    method: str,
    window_months: int,
    roll_threshold: float,
) -> None:
    pnl_df = _daily_pnl_combo(cr.combination, strategies)

    vp      = cr.portfolios.get(method)
    weights = vp.weights if vp else None
    mlabel  = METHOD_LABELS[method]

    _draw_corr_matrix(fig, ax_corr, pnl_df, weights, mlabel)
    _draw_rolling_corrs(fig, ax_roll, cr.combination, strategies, window_months, roll_threshold)
    _draw_pca(fig, ax_scree, ax_load, pnl_df)

    fig.suptitle(
        f"Deep Correlation  ·  Combination #{idx + 1}/{n}  ·  "
        f"{'  ×  '.join(_short(s) for s in cr.combination)}  ·  {mlabel}",
        color=CYAN, fontsize=11, y=0.99,
    )
    nav_label.set_text(f"Combination {idx + 1} of {n}")
    fig.canvas.draw_idle()


# ── Entry point ────────────────────────────────────────────────────────────────

def run_correlation_dashboard(
    combinations: list[CombinationResult],
    config: PortfolioConfig,
    strategies: dict,
    window_months: int | None = None,
    _show: bool = True,
) -> None:
    """
    Open the deep correlation analysis window.

    Args:
        combinations  : ranked list from the pipeline.
        config        : PortfolioConfig (window_months default taken from it).
        strategies    : raw strategies dict (name → DataFrame).
        window_months : rolling window length in months (default: config value).
    """
    if not combinations:
        print("  No combinations to display.")
        return

    if window_months is None:
        window_months = getattr(config, "rolling_window_months", 36)

    n     = len(combinations)
    state = {"idx": 0, "method": "equal"}

    # ── Figure & layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 11), facecolor=BG)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[1, 1.6],
        height_ratios=[1, 1],
        hspace=0.42,
        wspace=0.32,
        left=0.08, right=0.81,
        top=0.93, bottom=0.12,
    )

    ax_corr  = fig.add_subplot(gs[0, 0])
    ax_roll  = fig.add_subplot(gs[0, 1])
    ax_scree = fig.add_subplot(gs[1, 0])
    ax_load  = fig.add_subplot(gs[1, 1])

    # ── Navigation buttons ─────────────────────────────────────────────────────
    ax_nav   = fig.add_axes([0.38, 0.02, 0.18, 0.045])
    ax_nav.set_facecolor(BG); ax_nav.axis("off")
    nav_label = ax_nav.text(
        0.5, 0.5, "",
        transform=ax_nav.transAxes,
        color=TEXT, ha="center", va="center", fontsize=10,
    )

    ax_prev = fig.add_axes([0.29, 0.02, 0.08, 0.045])
    ax_next = fig.add_axes([0.57, 0.02, 0.08, 0.045])
    btn_prev = Button(ax_prev, "< Prev", color=AX_BG, hovercolor=BORDER)
    btn_next = Button(ax_next, "Next >",  color=AX_BG, hovercolor=BORDER)
    btn_prev.label.set_color(TEXT)
    btn_next.label.set_color(TEXT)

    # ── Method radio buttons ───────────────────────────────────────────────────
    radio_ax = fig.add_axes([0.83, 0.32, 0.15, 0.36])
    radio_ax.set_facecolor("#0e1228")
    for sp in radio_ax.spines.values():
        sp.set_edgecolor(BORDER)
    radio_ax.set_title("Method", color=CYAN, fontsize=9, pad=6)

    radio = RadioButtons(
        radio_ax,
        labels=[METHOD_LABELS[m] for m in METHODS],
        activecolor=CYAN,
    )
    # Style each radio label with its method colour
    for lbl, m in zip(radio.labels, METHODS):
        lbl.set_color(METHOD_COLORS[m])
        lbl.set_fontsize(9)

    # ── Callbacks ──────────────────────────────────────────────────────────────
    roll_threshold = getattr(config, "max_rolling_corr", 0.35)

    def _refresh() -> None:
        _render_corr(
            fig,
            ax_corr, ax_roll, ax_scree, ax_load,
            nav_label,
            combinations[state["idx"]], state["idx"], n,
            strategies, state["method"], window_months, roll_threshold,
        )

    def on_prev(event):
        state["idx"] = max(state["idx"] - 1, 0)
        _refresh()

    def on_next(event):
        state["idx"] = min(state["idx"] + 1, n - 1)
        _refresh()

    def on_method(label):
        for m in METHODS:
            if METHOD_LABELS[m] == label:
                state["method"] = m
                break
        _refresh()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    radio.on_clicked(on_method)

    # Keep widget references alive — without this they get GC'd when _show=False
    # causes the function to return, killing all callbacks.
    fig._refs = [btn_prev, btn_next, radio, state]

    _refresh()
    if _show:
        plt.show()
