"""
universe.py — Pre-computation of the strategy universe.

Builds all matrices needed by the generation pipeline once at startup,
so that per-combination lookups during generation are pure O(K²) index ops.

Pre-computed:
  - daily_pnl      [days × N]  : closed P&L per calendar day per strategy ($0 fill)
  - monthly_pnl    [months × N]: aggregated monthly P&L (pairwise NaN-aligned for corr)
  - pearson_matrix [N × N]     : full-period Pearson correlation on monthly P&L
  - spearman_matrix[N × N]     : full-period Spearman correlation on monthly P&L
  - co_loss_matrix [N × N]     : co-loss frequency on monthly P&L
  - overlap_matrix [N × N] bool: True if pair has same-asset same-day trade conflict

Date gap handling:
  Strategies with different start dates coexist in the daily matrix — absent
  days are filled with $0 (realistic: a strategy not yet running contributes
  nothing). Correlation matrices use pairwise overlapping months only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class Universe:
    """
    Pre-computed universe matrices for the portfolio generation pipeline.
    All index/column labels use the strategy name strings.
    """
    names          : list[str]
    daily_pnl      : pd.DataFrame          # [days × N], $0-filled
    monthly_pnl    : pd.DataFrame          # [months × N], NaN where no data
    pearson_matrix : pd.DataFrame          # [N × N] full-period Pearson
    spearman_matrix: pd.DataFrame          # [N × N] full-period Spearman
    co_loss_matrix : pd.DataFrame          # [N × N] co-loss frequency
    overlap_matrix : pd.DataFrame          # [N × N] bool — same-asset conflict
    name_to_idx    : dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}

    # ── Fast lookup helpers ────────────────────────────────────────────────────

    def pearson(self, a: str, b: str) -> float:
        return float(self.pearson_matrix.loc[a, b])

    def spearman(self, a: str, b: str) -> float:
        return float(self.spearman_matrix.loc[a, b])

    def co_loss(self, a: str, b: str) -> float:
        return float(self.co_loss_matrix.loc[a, b])

    def has_overlap(self, a: str, b: str) -> bool:
        return bool(self.overlap_matrix.loc[a, b])


# ── Internal builders ──────────────────────────────────────────────────────────

def _build_daily_pnl(strategies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the full daily P&L matrix [days × N].

    Each strategy's P&L is attributed to the calendar day its trade closed.
    Days with no trades for a strategy are filled with $0.
    The matrix spans the full date range of the entire universe — strategies
    not yet active in early dates contribute $0 (not NaN).
    """
    series = {}
    for name, df in strategies.items():
        daily = (
            df.set_index("Close time")["Profit/Loss"]
            .resample("D")
            .sum()
        )
        series[name] = daily

    mat = pd.DataFrame(series)

    # Full date range: from earliest close across all strategies
    full_range = pd.date_range(mat.index.min(), mat.index.max(), freq="D")
    mat = mat.reindex(full_range, fill_value=0.0).fillna(0.0)

    return mat


def _build_monthly_pnl(strategies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the monthly P&L matrix [months × N].

    Each column is the strategy's monthly P&L. Months with no trades are NaN
    (not zero) — this preserves pairwise overlap detection in correlation.
    """
    series = {}
    for name, df in strategies.items():
        s = df.set_index("Close time")["Profit/Loss"].copy()
        s.index = s.index.to_period("M")
        monthly = s.groupby(s.index).sum()
        series[name] = monthly

    # Use outer join — NaN where a strategy has no data for that month
    mat = pd.DataFrame(series)
    return mat


def _pairwise_pearson(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute full-period Pearson correlation using pairwise overlapping months."""
    names = list(monthly.columns)
    n = len(names)
    mat = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            a = monthly.iloc[:, i]
            b = monthly.iloc[:, j]
            both = a.notna() & b.notna()
            if both.sum() < 6:
                corr = 0.0
            else:
                corr, _ = pearsonr(a[both].values, b[both].values)
            mat[i, j] = corr
            mat[j, i] = corr

    return pd.DataFrame(mat, index=names, columns=names)


def _pairwise_spearman(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute full-period Spearman correlation using pairwise overlapping months."""
    names = list(monthly.columns)
    n = len(names)
    mat = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            a = monthly.iloc[:, i]
            b = monthly.iloc[:, j]
            both = a.notna() & b.notna()
            if both.sum() < 6:
                corr = 0.0
            else:
                corr, _ = spearmanr(a[both].values, b[both].values)
            mat[i, j] = corr
            mat[j, i] = corr

    return pd.DataFrame(mat, index=names, columns=names)


def _pairwise_co_loss(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute co-loss frequency matrix.
    co_loss(A, B) = months where both A < 0 AND B < 0 / overlapping months.
    """
    names = list(monthly.columns)
    n = len(names)
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            a = monthly.iloc[:, i]
            b = monthly.iloc[:, j]
            both = a.notna() & b.notna()
            total = both.sum()
            if total < 6:
                freq = 0.0
            else:
                freq = float(((a[both] < 0) & (b[both] < 0)).sum() / total)
            mat[i, j] = freq
            mat[j, i] = freq

    return pd.DataFrame(mat, index=names, columns=names)


def _pairwise_overlap(
    strategies: dict[str, pd.DataFrame],
    same_day: bool = True,
) -> pd.DataFrame:
    """
    Compute boolean conflict matrix.
    overlap(A, B) = True if any trade opens on the same asset on the same day.
    """
    names = list(strategies.keys())
    n = len(names)
    mat = np.zeros((n, n), dtype=bool)

    # Pre-build key sets per strategy
    key_sets = {}
    for name, df in strategies.items():
        if same_day:
            key_sets[name] = set(zip(df["Symbol"], df["Open time"].dt.date))
        else:
            key_sets[name] = set(zip(df["Symbol"], df["Open time"]))

    for i in range(n):
        for j in range(i + 1, n):
            conflict = len(key_sets[names[i]] & key_sets[names[j]]) > 0
            mat[i, j] = conflict
            mat[j, i] = conflict

    return pd.DataFrame(mat, index=names, columns=names)


# ── Public builder ─────────────────────────────────────────────────────────────

def build_universe(
    strategies: dict[str, pd.DataFrame],
    same_asset_same_day: bool = True,
    verbose: bool = True,
) -> Universe:
    """
    Pre-compute all universe matrices from the loaded strategy pool.

    Args:
        strategies          : dict name → DataFrame (from load_folder)
        same_asset_same_day : conflict granularity (True = day, False = bar)
        verbose             : print progress to terminal

    Returns:
        Universe dataclass with all pre-computed matrices.
    """
    names = list(strategies.keys())
    n     = len(names)
    t0    = time.time()

    def _log(msg: str) -> None:
        if verbose:
            print(f"  {msg}")

    _log(f"Building universe  ({n} strategies)")
    _log("-" * 48)

    # Daily P&L matrix
    _log("  [1/5]  Daily P&L matrix ...")
    t = time.time()
    daily_pnl = _build_daily_pnl(strategies)
    _log(f"         Done  ({len(daily_pnl)} days)  [{time.time()-t:.1f}s]")

    # Monthly P&L matrix
    _log("  [2/5]  Monthly P&L matrix ...")
    t = time.time()
    monthly_pnl = _build_monthly_pnl(strategies)
    _log(f"         Done  ({len(monthly_pnl)} months)  [{time.time()-t:.1f}s]")

    # Pearson matrix
    _log(f"  [3/5]  Pearson correlation matrix  ({n}x{n}) ...")
    t = time.time()
    pearson_mat = _pairwise_pearson(monthly_pnl)
    _log(f"         Done  [{time.time()-t:.1f}s]")

    # Spearman matrix
    _log(f"  [4/5]  Spearman correlation matrix ({n}x{n}) ...")
    t = time.time()
    spearman_mat = _pairwise_spearman(monthly_pnl)
    _log(f"         Done  [{time.time()-t:.1f}s]")

    # Co-loss + overlap (fast, combine into one step log)
    _log(f"  [5/5]  Co-loss & overlap matrices  ({n}x{n}) ...")
    t = time.time()
    co_loss_mat  = _pairwise_co_loss(monthly_pnl)
    overlap_mat  = _pairwise_overlap(strategies, same_day=same_asset_same_day)
    _log(f"         Done  [{time.time()-t:.1f}s]")

    _log(f"-" * 48)
    _log(f"  Universe ready  (total {time.time()-t0:.1f}s)\n")

    return Universe(
        names           = names,
        daily_pnl       = daily_pnl,
        monthly_pnl     = monthly_pnl,
        pearson_matrix  = pearson_mat,
        spearman_matrix = spearman_mat,
        co_loss_matrix  = co_loss_mat,
        overlap_matrix  = overlap_mat,
    )
