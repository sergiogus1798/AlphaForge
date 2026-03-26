"""
correlation_deep.py — Deep correlation and pool structure analysis.

Analyses the full strategy pool (not just pairs within a portfolio) to reveal
the underlying structure: clusters, diversification, factor exposures.

Four analyses:
  1. Full correlation matrix heatmap (Pearson + Spearman)
  2. PCA — how many independent return drivers exist in the pool
  3. Rolling correlation heatmap — how relationships change over time
  4. Dendrogram — hierarchical cluster structure (same method as HRP)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


BG    = "#0f1117"
AX_BG = "#1a1a2e"
TEXT  = "#e0e0e0"
GRID  = "#2a2a4a"
CYAN  = "#00d4ff"
AMBER = "#f0a500"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _daily_pnl_matrix(strategies: dict) -> pd.DataFrame:
    """
    Build an aligned daily P&L matrix for all strategies.
    Columns = strategy names, rows = calendar days.
    Missing days filled with 0. Pure flat days (weekends) dropped.
    """
    series = {}
    for name, df in strategies.items():
        daily = df.set_index("Close time")["Profit/Loss"].resample("D").sum()
        series[name] = daily
    mat = pd.DataFrame(series).fillna(0.0)
    return mat.loc[(mat != 0).any(axis=1)]


def _short(name: str) -> str:
    """Shorten strategy name for axis labels."""
    return name.split("/")[-1]


# ── 1. Correlation matrices ───────────────────────────────────────────────────

def plot_correlation_matrices(
    strategies: dict,
    figsize: tuple = (16, 7),
) -> None:
    """
    Side-by-side Pearson and Spearman correlation heatmaps for all strategies.
    """
    pnl  = _daily_pnl_matrix(strategies)
    labels = [_short(n) for n in pnl.columns]

    pearson  = pnl.corr(method="pearson")
    spearman = pnl.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=BG)
    fig.suptitle("Strategy Pool — Correlation Matrices", color=TEXT, fontsize=12)

    for ax, corr, title in zip(axes, [pearson, spearman], ["Pearson", "Spearman"]):
        ax.set_facecolor(AX_BG)
        im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", color=TEXT, fontsize=7)
        ax.set_yticklabels(labels, color=TEXT, fontsize=7)
        ax.set_title(title, color=TEXT, fontsize=10)

        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(val) < 0.7 else "white")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# ── 2. PCA ────────────────────────────────────────────────────────────────────

def plot_pca(
    strategies: dict,
    figsize: tuple = (14, 5),
) -> None:
    """
    PCA on daily P&L matrix.

    Left plot: explained variance per component (scree plot).
    Right plot: cumulative explained variance.
    Helps answer: how many independent return drivers exist in the strategy pool?
    """
    pnl = _daily_pnl_matrix(strategies)
    n_components = min(len(pnl.columns), len(pnl))

    # PCA via numpy SVD (no sklearn dependency)
    X = pnl.values - pnl.values.mean(axis=0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    variance = s[:n_components] ** 2
    explained     = variance / variance.sum() * 100
    cumulative    = np.cumsum(explained)
    components    = np.arange(1, n_components + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=BG)
    fig.suptitle("PCA — Independent Return Drivers in Strategy Pool", color=TEXT, fontsize=12)

    for ax in axes:
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT)
        ax.grid(True, color=GRID, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # Scree plot
    axes[0].bar(components, explained, color=CYAN, alpha=0.8)
    axes[0].set_xlabel("Component", color=TEXT)
    axes[0].set_ylabel("Explained Variance (%)", color=TEXT)
    axes[0].set_title("Variance per Component", color=TEXT, fontsize=9)
    axes[0].tick_params(colors=TEXT)

    # Cumulative
    axes[1].plot(components, cumulative, color=CYAN, linewidth=2, marker="o", markersize=4)
    axes[1].axhline(80, color=AMBER, linestyle="--", linewidth=1, label="80% threshold")
    axes[1].axhline(95, color="#e74c3c", linestyle="--", linewidth=1, label="95% threshold")
    axes[1].set_xlabel("Components", color=TEXT)
    axes[1].set_ylabel("Cumulative Variance (%)", color=TEXT)
    axes[1].set_title("Cumulative Explained Variance", color=TEXT, fontsize=9)
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXT)
    axes[1].tick_params(colors=TEXT)

    plt.tight_layout()
    plt.show()


# ── 3. Rolling correlation heatmap ────────────────────────────────────────────

def plot_rolling_correlation_heatmap(
    strategies: dict,
    pair: tuple[str, str],
    window_months: int = 12,
    figsize: tuple = (14, 4),
) -> None:
    """
    Rolling correlation between a specific strategy pair over time.
    Shows both Pearson and Spearman on the same axes.
    """
    from alphaforge.portfolio.generator.filters import _monthly_pnl, _rolling_correlations

    a_name, b_name = pair
    a_m = _monthly_pnl(strategies[a_name])
    b_m = _monthly_pnl(strategies[b_name])
    roll = _rolling_correlations(a_m, b_m, window_months)

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT)

    if not roll.empty:
        x = [p.to_timestamp() for p in roll.index]
        ax.plot(x, roll["pearson"],  color=CYAN,  linewidth=1.5, label="Pearson")
        ax.plot(x, roll["spearman"], color=AMBER, linewidth=1.5, linestyle="--", label="Spearman")
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.axhline(0.3, color="#e74c3c", linewidth=1, linestyle=":", label="0.30 threshold")

    ax.set_title(
        f"Rolling {window_months}m Correlation: {_short(a_name)} × {_short(b_name)}",
        color=TEXT, fontsize=10,
    )
    ax.set_ylabel("Correlation", color=TEXT)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXT)

    plt.tight_layout()
    plt.show()


# ── 4. Dendrogram ─────────────────────────────────────────────────────────────

def plot_dendrogram(
    strategies: dict,
    figsize: tuple = (14, 5),
) -> None:
    """
    Hierarchical clustering dendrogram of the strategy pool.
    Strategies that cluster together are most similar in their return profiles.
    Uses the same distance metric as HRP (sqrt((1 - corr) / 2)).
    """
    pnl    = _daily_pnl_matrix(strategies)
    labels = [_short(n) for n in pnl.columns]
    corr   = pnl.corr().values
    corr   = np.clip(corr, -1.0, 1.0)
    dist   = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

    condensed = squareform(dist, checks=False)
    condensed = np.where(condensed == 0, 1e-10, condensed)
    link = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT)

    dendrogram(
        link,
        labels=labels,
        ax=ax,
        color_threshold=0.6 * max(link[:, 2]),
        above_threshold_color=TEXT,
        leaf_font_size=8,
    )

    ax.set_title("Strategy Pool — Hierarchical Clustering Dendrogram", color=TEXT, fontsize=11)
    ax.set_ylabel("Distance", color=TEXT)
    ax.tick_params(axis="x", colors=TEXT, labelsize=8)
    ax.tick_params(axis="y", colors=TEXT)

    plt.tight_layout()
    plt.show()


# ── Run all ───────────────────────────────────────────────────────────────────

def run_deep_analysis(strategies: dict) -> None:
    """Run all four deep correlation analyses on the strategy pool."""
    print("  Running deep correlation analysis on strategy pool...\n")
    plot_correlation_matrices(strategies)
    plot_pca(strategies)
    plot_dendrogram(strategies)
    print("  Done. Use plot_rolling_correlation_heatmap(strategies, pair) for specific pairs.\n")
