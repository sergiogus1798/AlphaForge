"""
run_corr_viz.py — Correlation matrix visualisation for the full strategy universe.

Produces two figures:
  Figure 1 — Static matrices (Pearson, Spearman, Co-loss, Overlap)
  Figure 2 — Rolling max |correlation| (max |Pearson| and max |Spearman|
             across all 36-month windows for each pair)

Strategy labels are coloured by asset class.
Cells that *fail* the filter threshold are marked with a red border.

Usage:
    python scripts/run_corr_viz.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
from itertools import combinations as iter_combos

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from tqdm import tqdm

from alphaforge.loader.loader import load_folder
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.universe import build_universe
from alphaforge.portfolio.generator.filters import _monthly_pnl, _rolling_correlations
from alphaforge.paths import STRATEGIES_APPROVED


# ── Colour palette ────────────────────────────────────────────────────────────

BG       = "#0f1117"
AX_BG    = "#16213e"
TEXT     = "#e0e0e0"
GRID     = "#2a2a4a"
BORDER   = "#3a3a5a"
FAIL_CLR = "#e74c3c"

ASSET_COLORS: dict[str, str] = {
    "AUDJPY":  "#00d4ff",
    "AUDUSD":  "#26de81",
    "EURJPY":  "#f0a500",
    "GBPJPY":  "#ff6b6b",
    "GBPUSD":  "#ff9ff3",
    "NAS100":  "#54a0ff",
    "NIKKEI":  "#a29bfe",
    "USDJPY":  "#feca57",
    "XAUUSD":  "#ff9f43",
}

ASSET_ABBREV: dict[str, str] = {
    "AUDJPY": "AJ", "AUDUSD": "AU", "EURJPY": "EJ",
    "GBPJPY": "GJ", "GBPUSD": "GU", "NAS100": "NQ",
    "NIKKEI": "NK", "USDJPY": "UJ", "XAUUSD": "XA",
}


# ── Label helpers ─────────────────────────────────────────────────────────────

def _short(name: str) -> str:
    """'AUDJPY/Strategy 8.11.130' → 'AJ·8.11'"""
    parts = name.split("/")
    asset = parts[0]
    strat = parts[-1].replace("Strategy ", "")
    # Keep at most two dot-separated numbers for brevity
    nums = strat.split(".")
    strat_short = ".".join(nums[:2])
    return f"{ASSET_ABBREV.get(asset, asset[:2])}·{strat_short}"


def _asset(name: str) -> str:
    return name.split("/")[0]


def _color_ticks(ax, names: list[str], axis: str = "both") -> None:
    """Colour x/y tick labels by their asset."""
    colors = [ASSET_COLORS.get(_asset(n), TEXT) for n in names]
    if axis in ("x", "both"):
        for tick, c in zip(ax.get_xticklabels(), colors):
            tick.set_color(c)
    if axis in ("y", "both"):
        for tick, c in zip(ax.get_yticklabels(), colors):
            tick.set_color(c)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _style_ax(ax) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, length=2, width=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def _draw_heatmap(
    ax,
    mat: np.ndarray,
    labels: list[str],
    names: list[str],
    title: str,
    cmap,
    vmin: float,
    vmax: float,
    thresh: float | None = None,
    unit_diag: bool = True,
    fmt: str = ".2f",
) -> None:
    """
    Draw a square heatmap with labelled axes and an optional fail-threshold overlay.

    thresh — pairs where |value| > thresh get a red border.
    unit_diag — if True the diagonal is forced to 1.0 (correlation matrices).
    """
    n = len(labels)
    display = mat.copy()
    if unit_diag:
        np.fill_diagonal(display, 1.0 if vmax >= 1 else vmax)

    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5, ha="center", va="top")
    ax.set_yticklabels(labels, fontsize=5.5, va="center")
    _color_ticks(ax, names)

    ax.set_title(title, color=TEXT, fontsize=9, pad=6, fontweight="bold")
    _style_ax(ax)

    # Red border on cells that exceed the threshold (off-diagonal only)
    if thresh is not None:
        for i in range(n):
            for j in range(n):
                if i != j and abs(display[i, j]) > thresh:
                    ax.add_patch(mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor=FAIL_CLR,
                        linewidth=0.5, zorder=3,
                    ))

    # Thin white lines every N rows to separate asset groups visually
    # (find asset-boundary indices)
    boundaries = []
    prev = _asset(names[0])
    for idx in range(1, n):
        curr = _asset(names[idx])
        if curr != prev:
            boundaries.append(idx - 0.5)
            prev = curr
    for b in boundaries:
        ax.axhline(b, color="#ffffff", linewidth=0.4, alpha=0.25)
        ax.axvline(b, color="#ffffff", linewidth=0.4, alpha=0.25)

    return im


# ── Rolling max |corr| matrix ─────────────────────────────────────────────────

def _build_rolling_matrix(
    names: list[str],
    monthly_cache: dict,
    window: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute max |Pearson| and max |Spearman| across all rolling windows for every pair.
    Returns two (N×N) numpy arrays.
    """
    n = len(names)
    pearson_max  = np.zeros((n, n))
    spearman_max = np.zeros((n, n))

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    with tqdm(pairs, desc="  Rolling matrix", unit="pair", disable=not verbose) as bar:
        for i, j in bar:
            roll = _rolling_correlations(
                monthly_cache[names[i]], monthly_cache[names[j]], window
            )
            if roll.empty:
                v_p = v_s = 0.0
            else:
                v_p = float(roll["pearson"].abs().max())
                v_s = float(roll["spearman"].abs().max())
            pearson_max[i, j] = pearson_max[j, i] = v_p
            spearman_max[i, j] = spearman_max[j, i] = v_s

    return pearson_max, spearman_max


# ── Asset legend ──────────────────────────────────────────────────────────────

def _asset_legend(fig, names: list[str]) -> None:
    """Add a compact asset colour legend to the figure."""
    seen_assets = []
    for n in names:
        a = _asset(n)
        if a not in seen_assets:
            seen_assets.append(a)

    handles = [
        mpatches.Patch(
            facecolor=ASSET_COLORS.get(a, TEXT),
            label=f"{ASSET_ABBREV.get(a, a[:2])} = {a}",
            linewidth=0,
        )
        for a in seen_assets
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(seen_assets),
        fontsize=7,
        facecolor=AX_BG,
        edgecolor=BORDER,
        labelcolor=TEXT,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config = PortfolioConfig()

    print()
    print("=" * 56)
    print("  AlphaForge — Correlation Matrix Visualisation")
    print("=" * 56)

    strategies = load_folder(STRATEGIES_APPROVED)
    if not strategies:
        print("  No strategies found in strategies/approved/. Exiting.")
        sys.exit(1)
    print(f"  {len(strategies)} strategies loaded.\n")

    # ── Build universe (static matrices) ─────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe = build_universe(strategies, same_asset_window_hours=config.same_asset_window_hours)

    names  = universe.names
    labels = [_short(n) for n in names]
    n      = len(names)

    pearson  = universe.pearson_matrix.values
    spearman = universe.spearman_matrix.values
    co_loss  = universe.co_loss_matrix.values
    overlap  = universe.overlap_matrix.values.astype(float)

    # ── Build rolling matrices ────────────────────────────────────────────────
    monthly_cache = {name: _monthly_pnl(df) for name, df in strategies.items()}
    roll_p, roll_s = _build_rolling_matrix(
        names, monthly_cache, config.rolling_window_months, verbose=True
    )
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1 — Static correlation matrices
    # ─────────────────────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(
        2, 2, figsize=(22, 20),
        facecolor=BG,
        gridspec_kw={"hspace": 0.35, "wspace": 0.12},
    )
    fig1.suptitle(
        f"Static Correlation Matrices — {n} strategies",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98,
    )

    # Diverging colourmap centred at 0 (red = high positive, blue = high negative)
    div_cmap  = plt.cm.RdBu_r
    seq_cmap  = plt.cm.YlOrRd         # sequential for co-loss
    bool_cmap = mcolors.ListedColormap(["#1a1a2e", "#e74c3c"])  # no-overlap / conflict

    panels = [
        (axes1[0, 0], pearson,  "Pearson Correlation (monthly P&L)",
         div_cmap,  -1,  1, config.max_pearson_corr),
        (axes1[0, 1], spearman, "Spearman Correlation (monthly P&L)",
         div_cmap,  -1,  1, config.max_spearman_corr),
        (axes1[1, 0], co_loss,  f"Co-loss Frequency  (threshold ≤ {config.max_co_loss_freq})",
         seq_cmap,   0, 0.6, config.max_co_loss_freq),
        (axes1[1, 1], overlap,  "Same-Asset Conflict  (red = conflict)",
         bool_cmap,  0,  1, None),
    ]

    ims = []
    for ax, mat, title, cmap, vmin, vmax, thresh in panels:
        im = _draw_heatmap(ax, mat, labels, names, title, cmap, vmin, vmax,
                           thresh=thresh, unit_diag=(cmap is div_cmap))
        ims.append((ax, im, vmin, vmax))

    # Colorbars
    for ax, im, vmin, vmax in ims:
        cbar = fig1.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.85)
        cbar.ax.tick_params(colors=TEXT, labelsize=7)
        cbar.outline.set_edgecolor(BORDER)

    # Threshold annotation
    for (ax, mat, title, cmap, vmin, vmax, thresh) in panels:
        if thresh is not None:
            n_fail = int(np.sum(np.abs(np.triu(mat, 1)) > thresh))
            ax.set_xlabel(
                f"Red border = fails threshold ({n_fail} pairs above limit)",
                color=FAIL_CLR, fontsize=7, labelpad=4,
            )

    _asset_legend(fig1, names)
    fig1.subplots_adjust(bottom=0.07)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2 — Rolling correlation (max |corr| across all windows)
    # ─────────────────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(
        1, 2, figsize=(22, 10),
        facecolor=BG,
        gridspec_kw={"wspace": 0.12},
    )
    fig2.suptitle(
        f"Rolling Correlation — Max |corr| across all {config.rolling_window_months}-month windows",
        color=TEXT, fontsize=13, fontweight="bold", y=1.01,
    )

    roll_panels = [
        (axes2[0], roll_p, f"Max |Pearson|  (threshold ≤ {config.max_rolling_corr})",
         config.max_rolling_corr),
        (axes2[1], roll_s, f"Max |Spearman|  (threshold ≤ {config.max_rolling_corr})",
         config.max_rolling_corr),
    ]

    for ax, mat, title, thresh in roll_panels:
        im = _draw_heatmap(
            ax, mat, labels, names, title,
            cmap=plt.cm.YlOrRd, vmin=0, vmax=1,
            thresh=thresh, unit_diag=False,
        )
        cbar = fig2.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.85)
        cbar.ax.tick_params(colors=TEXT, labelsize=7)
        cbar.outline.set_edgecolor(BORDER)
        n_fail = int(np.sum(np.triu(mat, 1) > thresh))
        ax.set_xlabel(
            f"Red border = fails threshold ({n_fail} pairs above limit)",
            color=FAIL_CLR, fontsize=7, labelpad=4,
        )

    _asset_legend(fig2, names)
    fig2.subplots_adjust(bottom=0.10)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 3 — Combined filter pass/fail heatmap
    # ─────────────────────────────────────────────────────────────────────────
    # For each pair, how many filters does it fail? (0 = all pass, green)
    fail_pearson  = (np.abs(pearson)  > config.max_pearson_corr).astype(int)
    fail_spearman = (np.abs(spearman) > config.max_spearman_corr).astype(int)
    fail_coloss   = (co_loss          > config.max_co_loss_freq).astype(int)
    fail_overlap  = overlap.astype(int)
    fail_roll_p   = (roll_p           > config.max_rolling_corr).astype(int)
    fail_roll_s   = (roll_s           > config.max_rolling_corr).astype(int)

    # Number of filters failed per pair (0 – 6)
    fail_count = fail_pearson + fail_spearman + fail_coloss + fail_overlap + fail_roll_p + fail_roll_s
    np.fill_diagonal(fail_count, 0)

    # Boolean: pair is compatible (all 6 pass)?
    compatible = (fail_count == 0).astype(float)

    # Count compatible pairs
    n_pairs      = n * (n - 1) // 2
    n_compatible = int(np.sum(np.triu(compatible, 1)))
    n_overlap    = int(np.sum(np.triu(fail_overlap, 1)))

    fig3, axes3 = plt.subplots(
        1, 2, figsize=(22, 10),
        facecolor=BG,
        gridspec_kw={"wspace": 0.12},
    )
    fig3.suptitle(
        f"Filter Compatibility — {n_compatible}/{n_pairs} pairs pass all filters",
        color=TEXT, fontsize=13, fontweight="bold", y=1.01,
    )

    # Left: # of filters failed per pair
    im3a = _draw_heatmap(
        axes3[0], fail_count, labels, names,
        f"Filters failed per pair  (0 = fully compatible, dark = good)",
        cmap=plt.cm.RdYlGn_r, vmin=0, vmax=6,
        thresh=None, unit_diag=False,
    )
    cbar3a = fig3.colorbar(im3a, ax=axes3[0], fraction=0.03, pad=0.01, shrink=0.85,
                            ticks=[0, 1, 2, 3, 4, 5, 6])
    cbar3a.ax.tick_params(colors=TEXT, labelsize=7)
    cbar3a.outline.set_edgecolor(BORDER)

    # Right: fully compatible pairs (green) vs any failure (red)
    compat_cmap = mcolors.ListedColormap(["#e74c3c", "#26de81"])
    im3b = _draw_heatmap(
        axes3[1], compatible, labels, names,
        f"Compatible pairs  (green = all filters pass, {n_compatible}/{n_pairs} pairs)",
        cmap=compat_cmap, vmin=0, vmax=1,
        thresh=None, unit_diag=True,
    )

    # Stats annotation
    axes3[1].set_xlabel(
        f"Same-asset conflicts: {n_overlap}  |  "
        f"Compatible pairs (no same-asset): {n_compatible - 0}  |  "
        f"Total pairs: {n_pairs}",
        color=TEXT, fontsize=7, labelpad=4,
    )

    _asset_legend(fig3, names)
    fig3.subplots_adjust(bottom=0.10)

    # Print summary
    sep = "-" * 49
    print(f"\n  {sep}")
    print(f"  Pair Compatibility Summary")
    print(f"  {sep}")
    print(f"  Total pairs          : {n_pairs}")
    print(f"  Same-asset conflicts : {n_overlap}  (auto-fail)")
    print(f"  Fail Pearson (>={config.max_pearson_corr})  : {int(np.sum(np.triu(fail_pearson, 1)))}")
    print(f"  Fail Spearman (>={config.max_spearman_corr}) : {int(np.sum(np.triu(fail_spearman, 1)))}")
    print(f"  Fail Co-loss (>={config.max_co_loss_freq})  : {int(np.sum(np.triu(fail_coloss, 1)))}")
    print(f"  Fail Roll.Pearson    : {int(np.sum(np.triu(fail_roll_p, 1)))}")
    print(f"  Fail Roll.Spearman   : {int(np.sum(np.triu(fail_roll_s, 1)))}")
    print(f"  Compatible pairs     : {n_compatible} / {n_pairs}  ({n_compatible/n_pairs*100:.1f}%)")
    print(f"  {sep}\n")

    plt.show()


if __name__ == "__main__":
    main()