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
from alphaforge.portfolio.generator.parallel import pool_executor, resolve_workers


# ── Parallel worker helpers ───────────────────────────────────────────────────

_rc_monthly_cache  = None
_rc_config         = None
_rc_recent_cutoff  = None


def _init_rc_workers(monthly_cache, config, recent_cutoff) -> None:
    global _rc_monthly_cache, _rc_config, _rc_recent_cutoff
    _rc_monthly_cache = monthly_cache
    _rc_config        = config
    _rc_recent_cutoff = recent_cutoff


def _rc_worker(pair: tuple) -> tuple:
    """Worker: compute rolling cache pass/fail for one pair."""
    a, b = pair
    roll = _rolling_correlations(
        _rc_monthly_cache[a], _rc_monthly_cache[b],
        _rc_config.rolling_window_months,
    )
    if roll.empty:
        return (a, b, True)
    ok = True
    for period, row in roll.iterrows():
        threshold = (
            _rc_config.max_rolling_corr_recent
            if period > _rc_recent_cutoff
            else _rc_config.max_rolling_corr
        )
        if abs(row["pearson"]) > threshold or abs(row["spearman"]) > threshold:
            ok = False
            break
    return (a, b, ok)


_gs_names  = None
_gs_adj    = None
_gs_config = None


def _init_gs_workers(names, adj, config) -> None:
    global _gs_names, _gs_adj, _gs_config
    _gs_names  = names
    _gs_adj    = adj
    _gs_config = config


def _gs_batch_worker(args: tuple) -> list:
    """Worker: run N greedy walks, return list of valid combos found."""
    n_attempts, seed = args
    rng     = random.Random(seed)
    results = []
    for _ in range(n_attempts):
        shuffled  = _gs_names[:]
        rng.shuffle(shuffled)
        combo_set = []
        for candidate in shuffled:
            if len(combo_set) >= _gs_config.max_strategies:
                break
            if all(candidate in _gs_adj[existing] for existing in combo_set):
                combo_set.append(candidate)
        if len(combo_set) >= _gs_config.min_strategies:
            results.append(tuple(sorted(combo_set)))
    return results


_resolve_workers = resolve_workers  # local alias


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
        from concurrent.futures import ProcessPoolExecutor

        pairs = [
            (a, b)
            for i, a in enumerate(strategy_names)
            for b in strategy_names[i + 1:]
        ]
        latest_period = max(s.index.max() for s in monthly_cache.values())
        recent_cutoff = latest_period - config.recent_years * 12
        n_workers     = _resolve_workers(config.n_workers)

        if n_workers > 1:
            from concurrent.futures import as_completed
            with pool_executor(n_workers, _init_rc_workers,
                               (monthly_cache, config, recent_cutoff)) as pool:
                futures = [pool.submit(_rc_worker, pair) for pair in pairs]
                with tqdm(total=len(pairs), desc="  Rolling cache",
                          unit="pair", disable=not verbose) as bar:
                    for future in as_completed(futures):
                        a, b, ok = future.result()
                        self._cache[frozenset({a, b})] = ok
                        bar.update(1)
        else:
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


# ── Compatibility graph ───────────────────────────────────────────────────────

def _build_adjacency(
    names        : list[str],
    strategies   : dict,
    config       : PortfolioConfig,
    monthly_cache: dict[str, pd.Series],
    rolling_cache: _RollingCache,
    universe,
) -> dict[str, set[str]]:
    """
    Build a compatibility adjacency set: adj[A] = {B, C, ...} iff A passes
    ALL enabled pairwise filters with B, C, etc.

    Done in O(N²) using the pre-built Universe matrices and RollingCache — both
    are already computed before seeding, so this is pure index lookups.
    """
    adj: dict[str, set[str]] = {n: set() for n in names}
    dummy = FilterStats()

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not _apply_static_filters(
                (a, b), strategies, config, monthly_cache, dummy, universe=universe
            ):
                continue
            if config.enable_rolling_filter and not rolling_cache.passes(a, b):
                continue
            adj[a].add(b)
            adj[b].add(a)

    return adj


def _sample_clique(
    adj          : dict[str, set[str]],
    names        : list[str],
    size         : int,
    rng          : random.Random,
    start_weights: list[float] | None = None,
    max_restarts : int = 20,
) -> tuple[str, ...] | None:
    """
    Grow a clique of `size` in the compatibility graph by greedy intersection.

    Start from a (weighted) random strategy.  At each step, intersect the
    candidate set with the new node's adjacency — every node added is
    guaranteed compatible with ALL already-chosen nodes.

    start_weights biases the starting node toward underrepresented strategies
    for population diversity.

    Returns None if no clique of the required size is found after max_restarts.
    """
    for _ in range(max_restarts):
        if start_weights:
            start = rng.choices(names, weights=start_weights, k=1)[0]
        else:
            start = rng.choice(names)

        combo      = [start]
        candidates = set(adj[start])          # all compatible with start

        while len(combo) < size and candidates:
            nxt = rng.choice(list(candidates))
            combo.append(nxt)
            candidates &= adj[nxt]            # keep only those compatible with nxt too
            candidates.discard(nxt)

        if len(combo) == size:
            return tuple(sorted(combo))

    return None   # could not grow a clique of the required size


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
    Run structural filters on a candidate combo.  Returns a ValidPortfolio on
    success, None if any hard constraint is violated.

    Hard constraints (cause rejection):
      1. Static correlation filters  — Pearson, Spearman, co-loss, same-asset
      2. Rolling correlation filter  — all windows must be within threshold

    Soft constraint (NOT a hard gate here):
      Drawdown is handled by the fitness function via return_dd_ratio, which
      naturally selects away from bad-DD combos without blocking exploration.
      The final ValidPortfolio carries dd_failed=True/False for display.
    """
    stats = FilterStats()

    # 1. Static filters (O(1) with universe)
    if not _apply_static_filters(
        combo, strategies, config, monthly_cache, stats, universe=universe
    ):
        return None

    # 2. Rolling filter (O(1) with cache)
    if config.enable_rolling_filter and not rolling_cache.combo_passes(combo):
        return None

    # 3. Scale and validate (dd_failed flag set inside validate())
    weights   = equal_weight(list(combo))
    portfolio = build_weighted_portfolio(combo, strategies, weights)
    scaled    = rescale(combo, "equal", weights, portfolio, config)
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


def _absolute_fitness(ind: Individual, config: PortfolioConfig) -> float:
    """
    Absolute (non-normalised) weighted composite of the three fitness metrics.
    Used for stagnation tracking so the score is consistent across generations,
    unlike the population-normalised fitness which drifts as the pool changes.
    """
    m = ind.vp.metrics
    return (
        config.fitness_weight_return_dd      * m.get("return_dd_ratio",       0.0)
        + config.fitness_weight_annual_return  * m.get("yearly_avg_pct_return", 0.0)
        + config.fitness_weight_winning_months * m.get("winning_months_pct",    0.0)
    )


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


# ── Greedy independent-set seeding ───────────────────────────────────────────

def _seed_population_greedy(
    strategies   : dict,
    config       : PortfolioConfig,
    monthly_cache: dict[str, pd.Series],
    rolling_cache: _RollingCache,
    universe,
    rng          : random.Random,
    verbose      : bool,
) -> list[Individual]:
    """
    Greedy independent-set seeding.

    Each individual is built as follows:
      1. Shuffle all strategy names randomly.
      2. Start with the first strategy.
      3. Walk through the shuffled list — add a candidate if it passes ALL
         pairwise filters against every strategy already in the combo.
         Skip (do NOT discard from the universe) if it conflicts with anyone.
      4. Accept the combo if size >= min_strategies.

    Because all filter results are precomputed (universe matrices + rolling cache),
    every compatibility check is an O(1) dict lookup — no extra correlation
    computation happens during seeding.

    Being correlated with one member only blocks that specific pairing; the
    strategy can still appear in other individuals where it is compatible.
    """
    names  = list(strategies.keys())
    target = config.ga_population_size

    # ── Build adjacency graph (O(1) lookups thereafter) ──────────────────────
    if verbose:
        print("  Building compatibility graph...")
    adj = _build_adjacency(
        names, strategies, config, monthly_cache, rolling_cache, universe
    )
    n_pairs  = len(names) * (len(names) - 1) // 2
    n_compat = sum(len(v) for v in adj.values()) // 2
    if verbose:
        print(f"  Compatible pairs: {n_compat}/{n_pairs} "
              f"({n_compat / max(1, n_pairs) * 100:.1f}%)\n")

    if n_compat == 0:
        if verbose:
            print("  No compatible pairs — check filter thresholds.\n")
        return []

    population : list[Individual] = []
    seen       : set[tuple]       = set()
    n_workers  = _resolve_workers(config.n_workers)

    if verbose:
        print(f"  Greedy seeding (target: {target} individuals, workers: {n_workers})...")

    with tqdm(total=target, desc="  Seeding", unit="ind", disable=not verbose) as bar:

        stagnation   = 0
        max_stagnation = config.ga_seed_stagnation

        if n_workers > 1:
            batch_size  = max(200, target * 20)
            max_batches = config.ga_seed_attempts_multiplier // 20
            seeds       = [rng.randint(0, 2**31) for _ in range(max_batches)]
            tasks       = [(batch_size, s) for s in seeds]

            with pool_executor(n_workers, _init_gs_workers, (names, adj, config)) as pool:
                for batch in pool.map(_gs_batch_worker, tasks):
                    new_this_batch = 0
                    for combo in batch:
                        if len(population) >= target:
                            break
                        if combo in seen:
                            stagnation += 1
                            continue
                        seen.add(combo)
                        stagnation = 0
                        new_this_batch += 1
                        weights_eq = equal_weight(list(combo))
                        portfolio  = build_weighted_portfolio(combo, strategies, weights_eq)
                        scaled     = rescale(combo, "equal", weights_eq, portfolio, config)
                        vp         = validate(scaled, config)
                        population.append(Individual(combo=combo, vp=vp))
                        bar.update(1)
                    if len(population) >= target or stagnation >= max_stagnation:
                        if stagnation >= max_stagnation and verbose:
                            tqdm.write(f"  Seeding saturated after {stagnation:,} "
                                       f"consecutive duplicates — stopping early.")
                        break
        else:
            attempts     = 0
            max_attempts = target * config.ga_seed_attempts_multiplier

            while len(population) < target and attempts < max_attempts:
                attempts += 1
                shuffled  = names[:]
                rng.shuffle(shuffled)
                combo_set = []
                for candidate in shuffled:
                    if len(combo_set) >= config.max_strategies:
                        break
                    if all(candidate in adj[existing] for existing in combo_set):
                        combo_set.append(candidate)
                if len(combo_set) < config.min_strategies:
                    stagnation += 1
                else:
                    combo = tuple(sorted(combo_set))
                    if combo in seen:
                        stagnation += 1
                    else:
                        seen.add(combo)
                        stagnation = 0
                        weights_eq = equal_weight(list(combo))
                        portfolio  = build_weighted_portfolio(combo, strategies, weights_eq)
                        scaled     = rescale(combo, "equal", weights_eq, portfolio, config)
                        vp         = validate(scaled, config)
                        population.append(Individual(combo=combo, vp=vp))
                        bar.update(1)

                if stagnation >= max_stagnation:
                    if verbose:
                        tqdm.write(f"  Seeding saturated after {stagnation:,} "
                                   f"consecutive duplicates — stopping early.")
                    break

                if attempts % 500 == 0:
                    rate = len(population) / attempts if attempts else 0.0
                    bar.set_postfix(tried=f"{attempts:,}", found=len(population),
                                    rate=f"{rate:.1%}")

    if verbose:
        sizes = [len(ind.combo) for ind in population]
        if sizes:
            from collections import Counter
            dist = dict(sorted(Counter(sizes).items()))
            print(f"  Seeded {len(population)} individuals. Size distribution: {dist}\n")
        else:
            print("  Seeded 0 individuals.\n")

    return population


# ── Pair-growth seeding ───────────────────────────────────────────────────────

def _seed_population_pair_growth(
    strategies   : dict,
    config       : PortfolioConfig,
    monthly_cache: dict[str, pd.Series],
    rolling_cache: _RollingCache,
    universe,
    rng          : random.Random,
    verbose      : bool,
) -> list[Individual]:
    """
    Pair-growth seeding algorithm.

    Each individual is grown as follows:
      1. Seed with a random valid pair (both strategies pass all pairwise filters).
      2. At each growth step, try to add a PAIR of strategies — both must be
         compatible with every strategy already in the combo, and with each other.
         Fall back to adding a SINGLE strategy if no valid pair is found after
         MAX_PAIR_TRIES attempts.
      3. Stop when 200 consecutive attempts (pair or single) all fail to find
         anything compatible — the combo is stuck.
      4. Accept if size >= min_strategies, discard otherwise.

    This approach is better than clique-based sampling for large target sizes
    because it grows the combo in jumps of 2, reaching size 8+ far more reliably.
    """
    MAX_STUCK = 200

    names  = list(strategies.keys())
    target = config.ga_population_size

    # ── Build adjacency graph ─────────────────────────────────────────────────
    if verbose:
        print("  Building compatibility graph...")
    adj = _build_adjacency(
        names, strategies, config, monthly_cache, rolling_cache, universe
    )
    n_pairs  = len(names) * (len(names) - 1) // 2
    n_compat = sum(len(v) for v in adj.values()) // 2
    if verbose:
        print(f"  Compatible pairs: {n_compat}/{n_pairs} "
              f"({n_compat / max(1, n_pairs) * 100:.1f}%)\n")

    if n_compat == 0:
        if verbose:
            print("  No compatible pairs — check filter thresholds.\n")
        return []

    # Precompute list of all valid seed pairs for fast random selection
    valid_pairs = [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1:]
        if b in adj[a]
    ]

    population : list[Individual] = []
    seen       : set[tuple]       = set()

    if verbose:
        print(f"  Pair-growth seeding (target: {target} individuals)...")

    with tqdm(total=target, desc="  Seeding", unit="ind", disable=not verbose) as bar:
        overall_attempts = 0
        max_overall = target * config.ga_seed_attempts_multiplier

        while len(population) < target and overall_attempts < max_overall:
            overall_attempts += 1

            # ── Step 1: pick a random valid seed pair ─────────────────────────
            seed_a, seed_b = rng.choice(valid_pairs)
            combo_set = {seed_a, seed_b}

            # ── Step 2: grow the combo ────────────────────────────────────────
            stuck = 0
            while stuck < MAX_STUCK and len(combo_set) < config.max_strategies:
                # Strategies compatible with every member of current combo
                candidates = [
                    n for n in names
                    if n not in combo_set
                    and all(n in adj[c] for c in combo_set)
                ]

                if not candidates:
                    break

                # Try to add a pair first
                added = False
                pair_tries = 0
                rng.shuffle(candidates)
                for i in range(len(candidates)):
                    if len(combo_set) + 2 > config.max_strategies:
                        break
                    a = candidates[i]
                    # find a partner for a among remaining candidates
                    partners = [
                        candidates[j] for j in range(i + 1, len(candidates))
                        if candidates[j] in adj[a]
                    ]
                    if partners:
                        b = rng.choice(partners)
                        combo_set.add(a)
                        combo_set.add(b)
                        added = True
                        stuck = 0
                        break
                    pair_tries += 1
                    if pair_tries >= MAX_STUCK:
                        break

                # Fall back to single strategy
                if not added:
                    if candidates and len(combo_set) < config.max_strategies:
                        combo_set.add(rng.choice(candidates))
                        stuck = 0
                    else:
                        stuck += 1

            # ── Step 3: accept or discard ─────────────────────────────────────
            if len(combo_set) < config.min_strategies:
                continue

            combo = tuple(sorted(combo_set))
            if combo in seen:
                continue
            seen.add(combo)

            weights_eq = equal_weight(list(combo))
            portfolio  = build_weighted_portfolio(combo, strategies, weights_eq)
            scaled     = rescale(combo, "equal", weights_eq, portfolio, config)
            vp         = validate(scaled, config)

            population.append(Individual(combo=combo, vp=vp))
            bar.update(1)

            if overall_attempts % 200 == 0:
                rate = len(population) / overall_attempts if overall_attempts else 0.0
                bar.set_postfix(tried=f"{overall_attempts:,}", found=len(population),
                                rate=f"{rate:.1%}")

    if verbose:
        sizes = [len(ind.combo) for ind in population]
        if sizes:
            from collections import Counter
            dist = dict(sorted(Counter(sizes).items()))
            print(f"  Seeded {len(population)} individuals. Size distribution: {dist}\n")
        else:
            print("  Seeded 0 individuals.\n")

    return population


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
    Build the initial population using graph-based clique sampling.

    Improvements over naive random sampling:
      1. Adjacency graph — all pairwise filters evaluated once upfront;
         clique growth guarantees every drawn combo passes all filters
         without re-checking pairs.  Hit rate goes from ~0% to ~100%.
      2. Stratified size buckets — equal quota per combo size so all
         sizes are represented, not just the central tendency.
      3. Dissimilarity weighting — starting node biased toward strategies
         not yet in the population to maximise diversity.
      4. Exhaustive fallback uses adjacency sets (O(1) pair lookup).
    """
    names  = list(strategies.keys())
    target = config.ga_population_size
    sizes  = list(range(
        config.min_strategies,
        min(config.max_strategies, len(names)) + 1,
    ))

    # ── Build adjacency graph (O(N²) index lookups, done once) ───────────────
    if verbose:
        print("  Building compatibility graph...")

    adj = _build_adjacency(
        names, strategies, config, monthly_cache, rolling_cache, universe
    )

    n_pairs  = len(names) * (len(names) - 1) // 2
    n_compat = sum(len(v) for v in adj.values()) // 2
    if verbose:
        print(f"  Compatible pairs: {n_compat}/{n_pairs} "
              f"({n_compat / max(1, n_pairs) * 100:.1f}%)\n")

    if n_compat == 0:
        if verbose:
            print("  No compatible pairs at all — check filter thresholds.\n")
        return []

    # ── Stratified size quotas ────────────────────────────────────────────────
    n_sizes   = len(sizes)
    base      = target // n_sizes
    remainder = target % n_sizes
    quota     = {s: base + (1 if i < remainder else 0)
                 for i, s in enumerate(sizes)}   # sum == target

    population : list[Individual] = []
    seen       : set[tuple]       = set()
    attempts   = 0
    max_attempts = target * config.ga_seed_attempts_multiplier

    if verbose:
        print(f"  Seeding initial population (target: {target} individuals)...")

    with tqdm(
        total=target,
        desc="  Seeding",
        unit="ind",
        disable=not verbose,
    ) as bar:

        # ── Phase 1: graph-based clique sampling ──────────────────────────────
        # Start with uniform weights; updated every 10 new individuals
        uniform = 1.0 / len(names)
        sw = [uniform] * len(names)
        diversity_update = 0

        while len(population) < target and attempts < max_attempts:
            # Choose a size from buckets that still have quota; fall back freely
            remaining_sizes = [s for s in sizes if quota.get(s, 0) > 0]
            size = rng.choice(remaining_sizes if remaining_sizes else sizes)

            # Dissimilarity weights: recompute every 10 new individuals
            if len(population) > diversity_update:
                freq = {n: 0 for n in names}
                for ind in population:
                    for s in ind.combo:
                        freq[s] += 1
                sw    = [1.0 / (freq[n] + 1) for n in names]
                sw_sum = sum(sw)
                sw    = [w / sw_sum for w in sw]
                diversity_update = len(population) + 10

            combo = _sample_clique(adj, names, size, rng, start_weights=sw)
            attempts += 1

            if combo is None or combo in seen:
                if attempts % 200 == 0:
                    rate = len(population) / attempts if attempts else 0.0
                    bar.set_postfix(tried=f"{attempts:,}", found=len(population),
                                    rate=f"{rate:.1%}")
                continue

            seen.add(combo)

            # Filters are guaranteed by the graph — only rescale + validate needed
            weights_eq = equal_weight(list(combo))
            portfolio  = build_weighted_portfolio(combo, strategies, weights_eq)
            scaled     = rescale(combo, "equal", weights_eq, portfolio, config)
            vp         = validate(scaled, config)

            population.append(Individual(combo=combo, vp=vp))
            quota[size] = quota.get(size, 0) - 1
            bar.update(1)

            if attempts % 200 == 0:
                rate = len(population) / attempts if attempts else 0.0
                bar.set_postfix(tried=f"{attempts:,}", found=len(population),
                                rate=f"{rate:.1%}")

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
    if config.ga_use_greedy:
        population = _seed_population_greedy(
            strategies, config, monthly_cache, rolling_cache, universe, rng, verbose
        )
    elif config.ga_use_pair_growth:
        population = _seed_population_pair_growth(
            strategies, config, monthly_cache, rolling_cache, universe, rng, verbose
        )
    else:
        population = _seed_population(
            strategies, config, monthly_cache, rolling_cache, universe, rng, verbose
        )

    if not population:
        if verbose:
            print("  GA: no valid individuals found during seeding. "
                  "Try relaxing filter thresholds.\n")
        return []

    if len(population) < config.ga_population_size and verbose:
        print(f"  GA: seeded {len(population)}/{config.ga_population_size} individuals "
              f"— proceeding with partial population.\n")

    _score_population(population, config)
    population.sort(key=lambda ind: ind.fitness, reverse=True)

    names       = list(strategies.keys())
    elite_n     = max(1, int(len(population) * config.ga_elite_fraction))
    best_abs_fitness = _absolute_fitness(population[0], config)
    stagnation       = 0
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

        # Stagnation check — absolute composite (R/DD + ret + win_mo), not normalized
        current_abs_fitness = _absolute_fitness(population[0], config)
        if current_abs_fitness > best_abs_fitness + 1e-6:
            best_abs_fitness = current_abs_fitness
            stagnation       = 0
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
