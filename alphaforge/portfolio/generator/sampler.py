"""
sampler.py — Randomly sample strategy combinations for portfolio generation.

Produces a list of unique combinations (tuples of strategy names) of sizes
ranging from min_strategies to max_strategies, drawn from the available pool.
"""

import random
from itertools import combinations

from alphaforge.portfolio.config import PortfolioConfig


def _all_combinations(strategy_names: list[str], min_n: int, max_n: int) -> list[tuple]:
    """
    Return every possible combination of sizes min_n..max_n.
    Used when the pool is small enough that exhaustive search is feasible.
    """
    result = []
    for n in range(min_n, max_n + 1):
        result.extend(combinations(strategy_names, n))
    return result


def sample_combinations(
    strategies: dict,
    config: PortfolioConfig,
) -> list[tuple[str, ...]]:
    """
    Generate a list of strategy combinations to evaluate.

    If the total number of possible combinations is <= config.n_portfolios,
    returns all of them (exhaustive). Otherwise randomly samples n_portfolios
    unique combinations, spread across all valid sizes (min to max strategies).

    Args:
        strategies : dict mapping strategy name → DataFrame (only keys are used)
        config     : PortfolioConfig with min/max sizes, n_portfolios, random_seed

    Returns:
        List of tuples, each tuple being a unique set of strategy names.
    """
    names = list(strategies.keys())
    min_n = config.min_strategies
    max_n = min(config.max_strategies, len(names))

    if max_n < min_n:
        raise ValueError(
            f"Not enough strategies ({len(names)}) to form a portfolio "
            f"of at least {min_n}."
        )

    # Count total possible combinations
    total_possible = sum(
        _n_combinations(len(names), n) for n in range(min_n, max_n + 1)
    )

    if total_possible <= config.n_portfolios:
        # Small pool — just enumerate everything
        return _all_combinations(names, min_n, max_n)

    # Large pool — random sampling
    rng = random.Random(config.random_seed)
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []

    # Spread samples evenly across portfolio sizes
    sizes = list(range(min_n, max_n + 1))
    per_size = config.n_portfolios // len(sizes)
    remainder = config.n_portfolios % len(sizes)

    for i, n in enumerate(sizes):
        target = per_size + (1 if i < remainder else 0)
        attempts = 0
        max_attempts = target * 20  # avoid infinite loop on small pools

        while len([c for c in result if len(c) == n]) < target and attempts < max_attempts:
            combo = tuple(sorted(rng.sample(names, n)))
            if combo not in seen:
                seen.add(combo)
                result.append(combo)
            attempts += 1

    return result


def _n_combinations(n: int, k: int) -> int:
    """Compute n choose k without importing math.comb (available Python 3.8+)."""
    from math import comb
    return comb(n, k)
