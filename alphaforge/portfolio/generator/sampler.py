"""
sampler.py — Randomly sample strategy combinations for portfolio generation.

Produces a list of unique combinations (tuples of strategy names) of sizes
ranging from min_strategies to max_strategies, drawn from the available pool.

Sampling budget is allocated proportionally to the combinatorial space at
each size: if C(N,8) is 50% of all possible combinations then ~50% of the
budget goes to size-8 portfolios. This ensures the sampler explores the
combination space in proportion to how large each size bucket actually is,
rather than wasting equal budget on the much smaller size-3 bucket.

Each size bucket is capped at its total possible combinations — if the
proportional allocation exceeds the pool, we enumerate exhaustively for
that size and redistribute the leftover budget to larger sizes.
"""

import random
from itertools import combinations
from math import comb

from alphaforge.portfolio.config import PortfolioConfig


def _all_combinations(strategy_names: list[str], min_n: int, max_n: int) -> list[tuple]:
    """Return every possible combination of sizes min_n..max_n (exhaustive)."""
    result = []
    for n in range(min_n, max_n + 1):
        result.extend(combinations(strategy_names, n))
    return result


def _proportional_targets(
    n_strategies: int,
    sizes: list[int],
    budget: int,
) -> list[int]:
    """
    Allocate `budget` samples across `sizes` proportionally to C(n_strategies, k).

    Each bucket is capped at its actual pool size. Any budget freed by a cap
    is redistributed proportionally among the remaining (uncapped) buckets.
    This continues until all buckets are either satisfied or capped.

    Returns a list of integer targets, one per size, in the same order.
    """
    pool = [comb(n_strategies, k) for k in sizes]
    targets = [0] * len(sizes)
    remaining_budget = budget
    uncapped = list(range(len(sizes)))

    while uncapped and remaining_budget > 0:
        total_pool = sum(pool[i] for i in uncapped)
        new_uncapped = []
        leftover = 0

        for i in uncapped:
            alloc = round(pool[i] / total_pool * remaining_budget)
            if alloc >= pool[i]:
                targets[i] = pool[i]   # enumerate exhaustively — cap it
                leftover += alloc - pool[i]
            else:
                targets[i] = alloc
                new_uncapped.append(i)

        remaining_budget = leftover
        uncapped = new_uncapped

    # Distribute any rounding remainder to the largest uncapped bucket
    total_assigned = sum(targets)
    if total_assigned < budget and uncapped:
        largest = max(uncapped, key=lambda i: pool[i])
        extra = min(budget - total_assigned, pool[largest] - targets[largest])
        targets[largest] += extra

    return targets


def sample_combinations(
    strategies: dict,
    config: PortfolioConfig,
) -> list[tuple[str, ...]]:
    """
    Generate a list of strategy combinations to evaluate.

    If the total number of possible combinations is <= config.n_portfolios,
    returns all of them (exhaustive). Otherwise randomly samples n_portfolios
    unique combinations with the budget allocated proportionally to the
    combinatorial space at each size.

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

    sizes = list(range(min_n, max_n + 1))
    total_possible = sum(comb(len(names), n) for n in sizes)

    if total_possible <= config.n_portfolios:
        return _all_combinations(names, min_n, max_n)

    # Proportional budget allocation
    targets = _proportional_targets(len(names), sizes, config.n_portfolios)

    rng  = random.Random(config.random_seed)
    seen : set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []

    for n, target in zip(sizes, targets):
        if target == comb(len(names), n):
            # Exhaustive for this size
            for combo in combinations(names, n):
                combo = tuple(sorted(combo))
                if combo not in seen:
                    seen.add(combo)
                    result.append(combo)
        else:
            # Random sampling for this size
            count    = 0
            attempts = 0
            max_attempts = target * 20

            while count < target and attempts < max_attempts:
                combo = tuple(sorted(rng.sample(names, n)))
                if combo not in seen:
                    seen.add(combo)
                    result.append(combo)
                    count += 1
                attempts += 1

    return result
