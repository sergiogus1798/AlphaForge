"""
genetic.py — Genetic algorithm portfolio generator.

Evolves a population of strategy combinations across multiple generations,
using tournament selection, union crossover, and swap/add/remove mutation.
All filters (static + DD + rolling) are enforced as hard constraints — only
valid chromosomes enter or remain in the population.

Fitness is the same composite score used by the random pipeline:
  fitness = w1 * norm(return_dd) + w2 * norm(annual_return) + w3 * norm(winning_months)

Metrics are normalised across the current population each generation so
selection pressure adapts as the population improves over time.

Key optimisation: a per-pair rolling correlation cache is built once before
the main loop. Since rolling checks are the most expensive filter, caching
them means each (A, B) pair is computed at most once across the entire run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations as iter_combinations

import pandas as pd
from tqdm import tqdm

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.filters import (
    FilterStats,
    _apply_static_filters,
    _monthly_pnl,
    _rolling_correlations,
)
from alphaforge.portfolio.generator.weighting import (
    compute_all_weights,
    build_weighted_portfolio,
    equal_weight,
)
from alphaforge.portfolio.generator.scaler import rescale
from alphaforge.portfolio.generator.validator import validate, ValidPortfolio
from alphaforge.portfolio.generator.combo_result import CombinationResult
from alphaforge.portfolio.generator.fitness import score_combinations


# ── Rolling pair cache ────────────────────────────────────────────────────────

class _RollingCache:
    """
    Precomputes and caches the rolling correlation filter result per strategy pair.

    build() iterates all N*(N-1)/2 pairs and stores True/False for each.
    Subsequent calls to passes(a, b) are O(1) dict lookups.
    """

    def __init__(self) -> None:
        self._cache: dict[frozenset, bool] = {}

    def build(
        self,
        strategy_names: list[str],
        monthly_cache: dict[str, pd.Series],
        config: PortfolioConfig,
        verbose: bool = True,
    ) -> None:
        pairs = [
            (a, b)
            for i, a in enumerate(strategy_names)
            for b in strategy_names[i + 1:]
        ]
        latest_period = max(s.index.max() for s in monthly_cache.values())
        recent_cutoff = latest_period - config.recent_years * 12

        with tqdm(pairs, desc="  Rolling cache", unit="pair", disable=not verbose) as bar:
            for a, b in bar:
                key  = frozenset({a, b})
                roll = _rolling_correlations(
                    monthly_cache[a], monthly_cache[b], config.rolling_window_months
                )
                if roll.empty:
                    self._cache[key] = True
                    continue
                ok = True
                for period, row in roll.iterrows():
                    threshold = (
                        config.max_rolling_corr_recent
                        if period > recent_cutoff
                        else config.max_rolling_corr
                    )
                    if abs(row["pearson"]) > threshold or abs(row["spearman"]) > threshold:
                        ok = False
                        break
                self._cache[key] = ok

        if verbose:
            n_ok = sum(self._cache.values())
            print(f"  Rolling cache: {n_ok}/{len(pairs)} pairs compatible.\n")

    def passes(self, a: str, b: str) -> bool:
        return self._cache.get(frozenset({a, b}), True)

    def combo_passes(self, combo: tuple[str, ...]) -> bool:
        names = list(combo)
        return all(
            self.passes(names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        )


# ── Individual ────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    combo     : tuple[str, ...]
    vp        : ValidPortfolio
    fitness   : float = 0.0          # set by _score_population()


# ── Validity check ────────────────────────────────────────────────────────────

def _is_valid(
    combo        : tuple[str, ...],
    strategies   : dict,
    config       : PortfolioConfig,
    monthly_cache: dict[str, pd.Series],
    rolling_cache: _RollingCache,
    universe,
) -> ValidPortfolio | None:
    """
    Run all filters on a candidate combo. Returns a ValidPortfolio on success,
    None on any filter failure or DD breach.
    """
    stats = FilterStats()

    # 1. Static filters (O(1) with universe)
    if not _apply_static_filters(
        combo, strategies, config, monthly_cache, stats, universe=universe
    ):
        return None

    # 2. Rolling filter (O(1) with cache)
    if not rolling_cache.combo_passes(combo):
        return None

    # 3. Raw DD check (equal-weight, unscaled)
    weights   = equal_weight(list(combo))
    portfolio = build_weighted_portfolio(combo, strategies, weights)
    daily     = portfolio.groupby(portfolio["Close time"].dt.date)["Profit/Loss"].sum()
    worst_day = float(daily.min())
    dr        = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    equity    = daily.reindex(dr, fill_value=0.0).cumsum()
    total_eq  = config.account_balance + equity
    raw_dd    = float((total_eq.cummax() - total_eq).max())

    if (worst_day < -config.daily_loss_limit_usd or
            raw_dd > config.total_drawdown_limit_usd):
        return None

    # 4. Scale and validate
    scaled = rescale(combo, "equal", weights, portfolio, config)
    return validate(scaled, config)


# ── Fitness scoring ───────────────────────────────────────────────────────────

def _score_population(population: list[Individual], config: PortfolioConfig) -> None:
    """
    Normalise fitness across the current population and update each individual.
    Uses the same composite metric weights as the random pipeline.
    """
    candidates = [(ind.combo, ind.vp) for ind in population]
    scored     = score_combinations(candidates, config)
    score_map  = {combo: s for combo, _, s in scored}
    for ind in population:
        ind.fitness = score_map.get(ind.combo, 0.0)


# ── Genetic operators ─────────────────────────────────────────────────────────

def _tournament_select(
    population   : list[Individual],
    tournament_k : int,
    rng          : random.Random,
) -> Individual:
    """Pick `tournament_k` individuals at random; return the one with highest fitness."""
    contestants = rng.sample(population, min(tournament_k, len(population)))
    return max(contestants, key=lambda ind: ind.fitness)


def _crossover(
    parent_a: Individual,
    parent_b: Individual,
    config  : PortfolioConfig,
    rng     : random.Random,
) -> tuple[str, ...]:
    """
    Union crossover: take the set union of both parents' strategies,
    then sample a random subset of size in [min_n, max_n].
    """
    union = list(set(parent_a.combo) | set(parent_b.combo))
    size  = rng.randint(
        config.min_strategies,
        min(config.max_strategies, len(union)),
    )
    return tuple(sorted(rng.sample(union, size)))


def _mutate(
    combo      : tuple[str, ...],
    all_names  : list[str],
    config     : PortfolioConfig,
    rng        : random.Random,
) -> tuple[str, ...]:
    """
    Apply one random mutation:
      swap   — replace one strategy with a random unused one
      add    — add one unused strategy          (only if size < max)
      remove — drop one strategy                (only if size > min)
    """
    lst       = list(combo)
    available = [s for s in all_names if s not in lst]
    ops       = ["swap"]
    if len(lst) < config.max_strategies and available:
        ops.append("add")
    if len(lst) > config.min_strategies:
        ops.append("remove")

    op = rng.choice(ops)

    if op == "swap" and available:
        lst[rng.randrange(len(lst))] = rng.choice(available)
    elif op == "add":
        lst.append(rng.choice(available))
    elif op == "remove":
        lst.pop(rng.randrange(len(lst)))

    return tuple(sorted(lst))


# ── Seeding ───────────────────────────────────────────────────────────────────

def _seed_population(
    strategies   : dict,
    config       : PortfolioConfig,
    monthly_cache: dict[str, pd.Series],
    rolling_cache: _RollingCache,
    universe,
    rng          : random.Random,
    verbose      : bool,
) -> list[Individual]:
    """
    Build the initial population.

    Strategy: first try random combos (fast). If the target size isn't reached
    after many attempts, fall back to exhaustive enumeration of smaller sizes
    (size 3, then 4) to guarantee at least some valid individuals.
    """
    names       = list(strategies.keys())
    target      = config.ga_population_size
    population  : list[Individual] = []
    seen        : set[tuple]       = set()
    max_attempts = target * config.ga_seed_attempts_multiplier

    if verbose:
        print(f"  Seeding initial population (target: {target} individuals)...")

    with tqdm(
        total=target,
        desc="  Seeding",
        unit="ind",
        disable=not verbose,
    ) as bar:

        # ── Phase 1: random sampling ──────────────────────────────────────────
        for _ in range(max_attempts):
            if len(population) >= target:
                break
            size  = rng.randint(config.min_strategies,
                                min(config.max_strategies, len(names)))
            combo = tuple(sorted(rng.sample(names, size)))
            if combo in seen:
                continue
            seen.add(combo)
            vp = _is_valid(combo, strategies, config, monthly_cache, rolling_cache, universe)
            if vp is not None:
                population.append(Individual(combo=combo, vp=vp))
                bar.update(1)

        # ── Phase 2: exhaustive fallback (size 3 → 4 → ...) ──────────────────
        if len(population) < target:
            bar.set_description("  Seeding (exhaustive fallback)")
            for size in range(config.min_strategies,
                              min(config.max_strategies, len(names)) + 1):
                if len(population) >= target:
                    break
                for combo in iter_combinations(names, size):
                    if len(population) >= target:
                        break
                    combo = tuple(sorted(combo))
                    if combo in seen:
                        continue
                    seen.add(combo)
                    vp = _is_valid(
                        combo, strategies, config, monthly_cache,
                        rolling_cache, universe,
                    )
                    if vp is not None:
                        population.append(Individual(combo=combo, vp=vp))
                        bar.update(1)

    if verbose:
        print(f"  Seeded {len(population)} valid individuals.\n")

    return population


# ── Main GA loop ──────────────────────────────────────────────────────────────

def run_genetic(
    strategies: dict,
    config    : PortfolioConfig,
    universe  ,
    verbose   : bool = True,
) -> list[CombinationResult]:
    """
    Run the genetic algorithm and return the top combinations as CombinationResult
    objects (same format as run_pipeline), sorted by fitness descending.

    Args:
        strategies : loaded strategy DataFrames
        config     : PortfolioConfig — reads both filter and GA parameters
        universe   : pre-built Universe for O(1) pairwise lookups
        verbose    : print progress

    Returns:
        list[CombinationResult] sorted best-first by fitness score.
        May be shorter than config.top_combinations if fewer valid
        individuals were found.
    """
    rng = random.Random(config.random_seed)

    # ── Pre-compute monthly P&L cache ─────────────────────────────────────────
    monthly_cache: dict[str, pd.Series] = {
        name: _monthly_pnl(df) for name, df in strategies.items()
    }

    # ── Build rolling pair cache (one-time cost) ──────────────────────────────
    rolling_cache = _RollingCache()
    rolling_cache.build(
        list(strategies.keys()), monthly_cache, config, verbose=verbose
    )

    # ── Seed initial population ───────────────────────────────────────────────
    population = _seed_population(
        strategies, config, monthly_cache, rolling_cache, universe, rng, verbose
    )

    if not population:
        if verbose:
            print("  GA: no valid individuals found during seeding. "
                  "Try relaxing filter thresholds.\n")
        return []

    _score_population(population, config)
    population.sort(key=lambda ind: ind.fitness, reverse=True)

    names       = list(strategies.keys())
    elite_n     = max(1, int(len(population) * config.ga_elite_fraction))
    best_fitness = population[0].fitness
    stagnation   = 0
    seen_combos  : set[tuple] = {ind.combo for ind in population}

    if verbose:
        print(f"  GA: population={len(population)}, elite={elite_n}, "
              f"generations={config.ga_generations}, "
              f"stagnation_limit={config.ga_max_stagnation}\n")

    log_every = 5

    # ── Generation loop ───────────────────────────────────────────────────────
    gen_bar = tqdm(
        range(config.ga_generations),
        desc="  Evolving",
        unit="gen",
        disable=not verbose,
    )
    for gen in gen_bar:

        # Elite survives unchanged
        next_pop: list[Individual] = population[:elite_n]

        # Fill the rest with offspring
        attempts_budget = (len(population) - elite_n) * 50
        attempts        = 0
        accepted        = 0

        while len(next_pop) < len(population) and attempts < attempts_budget:
            attempts += 1

            # Crossover or inject a random individual
            if rng.random() < config.ga_crossover_prob and len(population) >= 2:
                pa    = _tournament_select(population, config.ga_tournament_size, rng)
                pb    = _tournament_select(population, config.ga_tournament_size, rng)
                combo = _crossover(pa, pb, config, rng)
            else:
                size  = rng.randint(config.min_strategies,
                                    min(config.max_strategies, len(names)))
                combo = tuple(sorted(rng.sample(names, size)))

            # Mutation
            if rng.random() < config.ga_mutation_prob:
                combo = _mutate(combo, names, config, rng)

            if combo in seen_combos:
                continue

            # Validate offspring
            vp = _is_valid(
                combo, strategies, config, monthly_cache, rolling_cache, universe
            )
            if vp is None:
                continue

            seen_combos.add(combo)
            next_pop.append(Individual(combo=combo, vp=vp))
            accepted += 1

        population = next_pop
        _score_population(population, config)
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        current_best = population[0].fitness
        accept_rate  = accepted / attempts if attempts > 0 else 0.0
        gen_bar.set_postfix(
            best=f"{current_best:.4f}",
            pop=len(population),
            accept=f"{accept_rate:.1%}",
            stag=stagnation,
        )

        # Periodic snapshot
        if verbose and (gen + 1) % log_every == 0:
            best_ind = population[0]
            m        = best_ind.vp.metrics
            tqdm.write(
                f"  Gen {gen+1:>4d}/{config.ga_generations}"
                f"  fitness={current_best:.4f}"
                f"  R/DD={m.get('return_dd_ratio', 0):.2f}"
                f"  ret={m.get('yearly_avg_pct_return', 0):.1f}%"
                f"  win_mo={m.get('winning_months_pct', 0):.0%}"
                f"  accept={accept_rate:.1%}"
                f"  stag={stagnation}"
                f"  n={len(best_ind.combo)}"
            )

        # Stagnation check
        if current_best > best_fitness + 1e-6:
            best_fitness = current_best
            stagnation   = 0
        else:
            stagnation += 1
            if stagnation >= config.ga_max_stagnation:
                if verbose:
                    tqdm.write(
                        f"\n  Early stop: no improvement for "
                        f"{config.ga_max_stagnation} generations.\n"
                    )
                break

    if verbose:
        print(f"\n  GA complete. Final population: {len(population)} individuals.\n")

    # ── Build CombinationResult for top-N ────────────────────────────────────
    top = population[: config.top_combinations]
    results: list[CombinationResult] = []

    with tqdm(
        top,
        desc="  Weighting top combinations",
        unit="combo",
        disable=not verbose,
    ) as bar:
        for rank_idx, ind in enumerate(bar, 1):
            combo      = ind.combo
            portfolios = {"equal": ind.vp}

            weights_all = compute_all_weights(combo, strategies)
            for method in ("min_variance", "risk_parity", "hrp"):
                weights   = weights_all[method]
                portfolio = build_weighted_portfolio(combo, strategies, weights)
                scaled    = rescale(combo, method, weights, portfolio, config)
                portfolios[method] = validate(scaled, config)

            results.append(CombinationResult(
                combination   = combo,
                rank          = rank_idx,
                raw_return_dd = ind.vp.metrics.get("return_dd_ratio", 0.0),
                raw_metrics   = ind.vp.metrics,
                portfolios    = portfolios,
                fitness_score = ind.fitness,
            ))

    return results
