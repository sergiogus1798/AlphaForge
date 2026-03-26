"""
weighting.py — Portfolio weight allocation methods (Step 2).

Four methods, all operating on aligned daily P&L series:

  1. Equal Weight     : w_i = 1/n
  2. Minimum Variance : minimise w^T Σ w  (quadratic programming)
  3. Risk Parity      : w_i ∝ 1/σ_i
  4. HRP              : Hierarchical Risk Parity (López de Prado, 2016)

All methods return a dict {strategy_name: weight} with weights summing to 1.
Weights are always positive (long-only).

The caller then derives risk_per_trade_i = base_risk × w_i × n_strategies,
so equal weight always produces exactly base_risk for every strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize


# ── Daily P&L alignment ───────────────────────────────────────────────────────

def build_daily_pnl(
    strategies: dict[str, pd.DataFrame],
    names: list[str],
) -> pd.DataFrame:
    """
    Build an aligned daily P&L matrix for a set of strategies.

    Each column = one strategy. Days with no trades = 0.
    Rows span the full date range covered by any strategy.

    Uses Close time to attribute P&L to the day the trade closed.
    """
    series = {}
    for name in names:
        df = strategies[name]
        daily = (
            df.set_index("Close time")["Profit/Loss"]
            .resample("D")
            .sum()
        )
        series[name] = daily

    pnl = pd.DataFrame(series).fillna(0.0)

    # Drop days where ALL strategies are flat (weekends, holidays)
    pnl = pnl.loc[(pnl != 0).any(axis=1)]

    return pnl


# ── 1. Equal Weight ───────────────────────────────────────────────────────────

def equal_weight(names: list[str]) -> dict[str, float]:
    """
    w_i = 1/n for all strategies.
    """
    n = len(names)
    return {name: 1.0 / n for name in names}


# ── 2. Minimum Variance ───────────────────────────────────────────────────────

def min_variance(pnl: pd.DataFrame) -> dict[str, float]:
    """
    Find weights that minimise total portfolio variance.

    Solves:  min  w^T Σ w
             s.t. sum(w) = 1,  w_i >= 0

    Falls back to equal weight if optimisation fails or covariance matrix
    is singular (can happen with very few data points).
    """
    names = list(pnl.columns)
    n = len(names)
    cov = pnl.cov().values

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n  # equal weight starting point

    result = minimize(
        portfolio_variance,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        # Fallback: equal weight
        return equal_weight(names)

    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()  # re-normalise after clipping
    return dict(zip(names, weights.tolist()))


# ── 3. Risk Parity ────────────────────────────────────────────────────────────

def risk_parity(pnl: pd.DataFrame) -> dict[str, float]:
    """
    Weight each strategy inversely proportional to its daily P&L volatility.

    w_i = (1/σ_i) / sum(1/σ_j)

    Each strategy contributes equally to total portfolio risk (exact in the
    uncorrelated case, approximate otherwise — fast and robust).

    Strategies with zero volatility receive equal weight with the others.
    """
    names = list(pnl.columns)
    vols = pnl.std().values

    # Guard against zero volatility
    vols = np.where(vols == 0, np.nan, vols)
    inv_vols = np.where(np.isnan(vols), 0.0, 1.0 / vols)

    if inv_vols.sum() == 0:
        return equal_weight(names)

    weights = inv_vols / inv_vols.sum()
    return dict(zip(names, weights.tolist()))


# ── 4. HRP (Hierarchical Risk Parity) ────────────────────────────────────────

def _corr_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to a proper distance matrix (López de Prado eq.)."""
    return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))


def _quasi_diagonalise(link: np.ndarray, n: int) -> list[int]:
    """Return the leaf order from hierarchical clustering (quasi-diagonalisation)."""
    return list(leaves_list(link))


def _recursive_bisection(
    cov: np.ndarray,
    sort_ix: list[int],
) -> np.ndarray:
    """
    Allocate weights by recursive bisection of the sorted covariance matrix.

    At each split: variance of left cluster vs right cluster determines how
    to divide the risk budget between the two halves.
    """
    n = len(sort_ix)
    weights = np.ones(n)

    clusters = [sort_ix]  # start with one cluster containing all strategies

    while clusters:
        # Split each cluster into two halves
        new_clusters = []
        for cluster in clusters:
            if len(cluster) == 1:
                continue

            mid = len(cluster) // 2
            left  = cluster[:mid]
            right = cluster[mid:]

            # Variance of each sub-cluster (using inverse-variance weighting within)
            def cluster_var(idxs: list[int]) -> float:
                sub_cov = cov[np.ix_(idxs, idxs)]
                sub_vols = np.sqrt(np.diag(sub_cov))
                sub_vols = np.where(sub_vols == 0, 1e-10, sub_vols)
                w = (1.0 / sub_vols) / (1.0 / sub_vols).sum()
                return float(w @ sub_cov @ w)

            var_left  = cluster_var(left)
            var_right = cluster_var(right)

            total = var_left + var_right
            alpha = 1.0 - (var_left / total) if total > 0 else 0.5

            # Multiply weights: left cluster gets (1-alpha), right gets alpha
            for i in left:
                weights[sort_ix.index(i)] *= (1.0 - alpha)
            for i in right:
                weights[sort_ix.index(i)] *= alpha

            new_clusters.extend([left, right])

        clusters = new_clusters

    return weights


def hrp(pnl: pd.DataFrame) -> dict[str, float]:
    """
    Hierarchical Risk Parity (López de Prado, 2016).

    Steps:
      1. Compute correlation + covariance matrix from daily P&L
      2. Convert correlation to distance matrix
      3. Hierarchical clustering (Ward linkage)
      4. Quasi-diagonalise: reorder strategies by cluster similarity
      5. Recursive bisection: allocate risk budget top-down

    Falls back to Risk Parity if clustering fails (e.g. < 2 strategies).
    """
    names = list(pnl.columns)
    n = len(names)

    if n < 2:
        return equal_weight(names)

    corr = pnl.corr().values
    cov  = pnl.cov().values

    # Clip correlation to [-1, 1] to handle floating point noise
    corr = np.clip(corr, -1.0, 1.0)

    dist = _corr_to_distance(corr)

    # scipy linkage expects condensed distance matrix
    condensed = squareform(dist, checks=False)
    # Avoid zero distances (identical strategies) which break linkage
    condensed = np.where(condensed == 0, 1e-10, condensed)

    try:
        link = linkage(condensed, method="ward")
    except Exception:
        return risk_parity(pnl)

    sort_ix = _quasi_diagonalise(link, n)
    raw_weights = _recursive_bisection(cov, sort_ix)

    raw_weights = np.clip(raw_weights, 0.0, None)
    total = raw_weights.sum()
    if total == 0:
        return equal_weight(names)

    weights = raw_weights / total
    return dict(zip(names, weights.tolist()))


# ── Dispatcher ────────────────────────────────────────────────────────────────

METHODS = ("equal", "min_variance", "risk_parity", "hrp")


def compute_weights(
    combination: tuple[str, ...],
    strategies: dict[str, pd.DataFrame],
    method: str = "hrp",
) -> dict[str, float]:
    """
    Compute portfolio weights for a combination of strategies.

    Args:
        combination : tuple of strategy names
        strategies  : dict name → DataFrame
        method      : one of "equal", "min_variance", "risk_parity", "hrp"

    Returns:
        dict {strategy_name: weight}, weights sum to 1.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {METHODS}")

    names = list(combination)

    if method == "equal":
        return equal_weight(names)

    pnl = build_daily_pnl(strategies, names)

    if method == "min_variance":
        return min_variance(pnl)
    elif method == "risk_parity":
        return risk_parity(pnl)
    elif method == "hrp":
        return hrp(pnl)


def compute_all_weights(
    combination: tuple[str, ...],
    strategies: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    """
    Compute weights for a combination using ALL four methods.

    Returns:
        dict {method_name: {strategy_name: weight}}
    """
    return {method: compute_weights(combination, strategies, method) for method in METHODS}


def build_weighted_portfolio(
    combination: tuple[str, ...],
    strategies: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Build a combined trades DataFrame with each strategy's Profit/Loss scaled
    by its weight allocation.

    Scaling formula: P&L_i × w_i × n_strategies
    This preserves the total risk budget while redistributing between strategies.
    Equal weight (w_i = 1/n) leaves every trade at its original P&L.

    Returns a single DataFrame with all trades sorted chronologically.
    """
    n = len(combination)
    frames = []
    for name in combination:
        df = strategies[name].copy()
        df["Profit/Loss"] = df["Profit/Loss"] * weights[name] * n
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values("Close time").reset_index(drop=True)
