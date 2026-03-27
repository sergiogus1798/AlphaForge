"""
fitness.py — Composite fitness scoring for combination ranking.

Combines three metrics into a single score using configurable weights:

  fitness = w1 * norm(return_dd_ratio)
          + w2 * norm(yearly_avg_pct_return)
          + w3 * norm(winning_months_pct)

Each metric is min-max normalised across the full candidate pool before
weighting, so the three components are always on the same [0, 1] scale
regardless of their raw units.

Designed to be extended for genetic generation — score_combinations()
returns a plain list of (combo, vp, score) triples that can be used as
the fitness function in any selection/mutation loop.
"""

from __future__ import annotations

import numpy as np

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.validator import ValidPortfolio


# ── Normalisation helper ──────────────────────────────────────────────────────

def _minmax(values: np.ndarray) -> np.ndarray:
    """
    Min-max normalise an array to [0, 1].
    Returns 0.5 for every element if all values are identical (flat pool).
    """
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


# ── Public API ────────────────────────────────────────────────────────────────

def score_combinations(
    candidates: list[tuple[tuple[str, ...], ValidPortfolio]],
    config: PortfolioConfig,
) -> list[tuple[tuple[str, ...], ValidPortfolio, float]]:
    """
    Compute a normalised fitness score for every candidate combination and
    return them sorted by score descending.

    Args:
        candidates : list of (combo_tuple, ValidPortfolio) from the equal-weight pass
        config     : PortfolioConfig — reads fitness_weight_* fields

    Returns:
        List of (combo, vp, score) sorted best-first.
        score is in [0, 1].
    """
    if not candidates:
        return []

    combos = [c for c, _ in candidates]
    vps    = [v for _, v in candidates]

    # ── Extract raw metric values ─────────────────────────────────────────────
    return_dd     = np.array([v.metrics.get("return_dd_ratio",       0.0) for v in vps], dtype=float)
    annual_return = np.array([v.metrics.get("yearly_avg_pct_return", 0.0) for v in vps], dtype=float)
    winning_mon   = np.array([v.metrics.get("winning_months_pct",    0.0) for v in vps], dtype=float)

    # ── Normalise each component across the pool ──────────────────────────────
    norm_rdd  = _minmax(return_dd)
    norm_ret  = _minmax(annual_return)
    norm_win  = _minmax(winning_mon)

    # ── Weighted composite ────────────────────────────────────────────────────
    w1 = config.fitness_weight_return_dd
    w2 = config.fitness_weight_annual_return
    w3 = config.fitness_weight_winning_months

    scores = w1 * norm_rdd + w2 * norm_ret + w3 * norm_win

    # ── Sort best-first ───────────────────────────────────────────────────────
    ranked = sorted(
        zip(combos, vps, scores.tolist()),
        key=lambda t: t[2],
        reverse=True,
    )

    return ranked
