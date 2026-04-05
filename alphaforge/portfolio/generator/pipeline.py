"""
pipeline.py — Full portfolio generation pipeline.

Filter order (applied before expensive rolling correlation check):
  1. Sample combinations
  2. Static correlation filters   (Pearson, Spearman, co-loss, same-asset)
  3. Raw DD check                 (equal-weight scaled → max daily loss + total DD)
  4. Rolling correlation filters  (12-month windows, recent stricter threshold)
  5. Rank by return/DD            (equal-weight scaled) → keep top_combinations
  6. Apply all 4 weighting methods to the top-N → CombinationResult per combo

Auto-regeneration: if a round yields 0 ranked combinations, re-samples with
an offset seed up to config.max_rounds times.
"""

from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.universe import build_universe
from alphaforge.portfolio.generator.sampler import sample_combinations
from alphaforge.portfolio.generator.filters import (
    filter_static_only,
    filter_rolling_only,
    FilterStats,
)
from alphaforge.portfolio.generator.weighting import (
    compute_all_weights,
    build_weighted_portfolio,
    METHODS,
    equal_weight,
)
from alphaforge.portfolio.generator.scaler import rescale
from alphaforge.portfolio.generator.validator import validate, ValidPortfolio
from alphaforge.portfolio.generator.combo_result import CombinationResult
from alphaforge.portfolio.generator.fitness import score_combinations
from alphaforge.portfolio.generator.wf import compute_all_wf_equities
from alphaforge.portfolio.generator.parallel import pool_executor
from alphaforge.portfolio.generator.filters import _resolve_workers


# ── DD check parallel workers ─────────────────────────────────────────────────

_dd_strategies = None
_dd_config     = None


def _init_dd_workers(strategies, config) -> None:
    global _dd_strategies, _dd_config
    _dd_strategies = strategies
    _dd_config     = config


def _dd_worker(combo: tuple) -> tuple:
    """Worker: raw DD check + equal-weight scale for one combo."""
    import pandas as _pd

    weights   = equal_weight(list(combo))
    portfolio = build_weighted_portfolio(combo, _dd_strategies, weights)

    daily = (
        portfolio
        .groupby(portfolio["Close time"].dt.date)["Profit/Loss"]
        .sum()
    )
    worst_day = float(daily.min())

    dr     = _pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    equity = daily.reindex(dr, fill_value=0.0).cumsum()
    total_eq = _dd_config.account_balance + equity
    peak     = total_eq.cummax()
    raw_dd   = float((peak - total_eq).max())

    if (worst_day < -_dd_config.daily_loss_limit_usd or
            raw_dd > _dd_config.total_drawdown_limit_usd):
        return (combo, False, None)

    scaled = rescale(combo, "equal", weights, portfolio, _dd_config)
    vp     = validate(scaled, _dd_config)
    return (combo, True, vp)


# ── Step 3: Raw DD check (equal-weight scaled) ────────────────────────────────

def _raw_dd_filter(
    combinations: list[tuple[str, ...]],
    strategies: dict,
    config: PortfolioConfig,
    stats: FilterStats,
    verbose: bool = True,
) -> tuple[list[tuple[str, ...]], dict[tuple, ValidPortfolio]]:
    """
    Check each combination at base risk (equal-weight, NO scaling).

    Rejects combinations where the raw portfolio already breaches either:
      - daily loss limit  : worst single day  < -daily_loss_limit_usd
      - total DD limit    : max drawdown       >  total_drawdown_limit_usd

    If the raw portfolio is within both limits before any scaling, the
    combination is a sound base — scaling and full DD validation are
    applied later (after weighting) by validate().

    Also pre-computes the equal-weight SCALED portfolio for survivors so
    the ranking step (Step 5) can reuse it without recomputation.

    Returns:
        (dd_passed_combos, {combo: equal_weight_scaled_ValidPortfolio})
    """
    import pandas as pd

    from concurrent.futures import as_completed as _as_completed

    passed    : list[tuple] = []
    equal_vps : dict[tuple, ValidPortfolio] = {}
    n_workers  = _resolve_workers(config.n_workers)

    if n_workers > 1:
        from concurrent.futures import as_completed
        with pool_executor(n_workers, _init_dd_workers, (strategies, config)) as pool:
            futures = [pool.submit(_dd_worker, combo) for combo in combinations]
            with tqdm(total=len(combinations), desc="  DD check (raw base-risk)",
                      unit="combo", disable=not verbose) as bar:
                for future in as_completed(futures):
                    combo, ok, vp = future.result()
                    if ok:
                        passed.append(combo)
                        equal_vps[combo] = vp
                    else:
                        stats.rejected_dd += 1
                    bar.update(1)
                    bar.set_postfix(passed=len(passed))
    else:
        with tqdm(
            combinations,
            desc="  DD check (raw base-risk)",
            unit="combo",
            disable=not verbose,
        ) as bar:
            for combo in bar:
                weights   = equal_weight(list(combo))
                portfolio = build_weighted_portfolio(combo, strategies, weights)

                daily = (
                    portfolio
                    .groupby(portfolio["Close time"].dt.date)["Profit/Loss"]
                    .sum()
                )
                worst_day = float(daily.min())

                dr     = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
                equity = daily.reindex(dr, fill_value=0.0).cumsum()
                total_eq = config.account_balance + equity
                peak     = total_eq.cummax()
                raw_dd   = float((peak - total_eq).max())

                if (worst_day < -config.daily_loss_limit_usd or
                        raw_dd > config.total_drawdown_limit_usd):
                    stats.rejected_dd += 1
                    bar.set_postfix(passed=len(passed), tried=bar.n)
                    continue

                scaled = rescale(combo, "equal", weights, portfolio, config)
                vp     = validate(scaled, config)

                passed.append(combo)
                equal_vps[combo] = vp
                bar.set_postfix(passed=len(passed), tried=bar.n)

    stats.passed_dd = len(passed)

    if verbose:
        rej = len(combinations) - len(passed)
        print(f"\n  DD check: {len(passed)}/{len(combinations)} passed "
              f"({rej} exceeded daily or total-DD limit at base risk).\n")

    return passed, equal_vps


# ── Step 5: Rank by fitness score ────────────────────────────────────────────

def _rank_combinations(
    combinations: list[tuple[str, ...]],
    equal_vps: dict[tuple, ValidPortfolio],
    config: PortfolioConfig,
    verbose: bool = True,
) -> list[tuple[tuple[str, ...], ValidPortfolio, float]]:
    """
    Score rolling-passed combinations with the composite fitness function
    (return/DD × w1 + annual return × w2 + winning months × w3),
    all normalised across the pool. Returns the top config.top_combinations.
    """
    candidates = [
        (combo, equal_vps[combo])
        for combo in combinations
        if combo in equal_vps
    ]

    scored = score_combinations(candidates, config)
    top    = scored[: config.top_combinations]

    if verbose and top:
        print(
            f"  Ranking: {len(scored)} combination(s), top {len(top)} selected  "
            f"(fitness: {top[-1][2]:.4f}–{top[0][2]:.4f}, "
            f"weights R/DD={config.fitness_weight_return_dd:.0%} "
            f"ret={config.fitness_weight_annual_return:.0%} "
            f"win_mo={config.fitness_weight_winning_months:.0%}).\n"
        )

    return top  # list of (combo, vp, fitness_score)


# ── Step 6: Apply all 4 weighting methods ────────────────────────────────────

def _build_combination_results(
    top_ranked: list[tuple[tuple[str, ...], ValidPortfolio, float]],
    strategies: dict,
    config: PortfolioConfig,
    verbose: bool = True,
) -> list[CombinationResult]:
    """
    For each combination, apply min_variance / risk_parity / hrp.
    (equal is already computed and passed in.)
    Returns list of CombinationResult sorted by fitness_score descending.
    """
    results: list[CombinationResult] = []

    with tqdm(
        top_ranked,
        desc="  Weighting (4 methods)",
        unit="combo",
        disable=not verbose,
    ) as bar:
        for rank_idx, (combo, equal_vp, fitness) in enumerate(bar, 1):
            portfolios: dict = {"equal": equal_vp}

            weights_all = compute_all_weights(combo, strategies)
            for method in ("min_variance", "risk_parity", "hrp"):
                weights   = weights_all[method]
                portfolio = build_weighted_portfolio(combo, strategies, weights)
                scaled    = rescale(combo, method, weights, portfolio, config)
                portfolios[method] = validate(scaled, config)

            results.append(CombinationResult(
                combination   = combo,
                rank          = rank_idx,
                raw_return_dd = equal_vp.metrics.get("return_dd_ratio", 0.0),
                raw_metrics   = equal_vp.metrics,
                portfolios    = portfolios,
                fitness_score = fitness,
            ))

    return results


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Summary of a full pipeline run."""
    combinations       : list[CombinationResult]
    total_sampled      : int
    total_filter_stats : list[FilterStats]
    rounds_used        : int
    succeeded          : bool

    @property
    def count(self) -> int:
        return len(self.combinations)

    @property
    def total_portfolios(self) -> int:
        return self.count * len(METHODS)

    @property
    def valid_portfolios(self) -> int:
        return sum(cr.n_valid_methods for cr in self.combinations)

    def report(self) -> None:
        print(f"\n  ══ Pipeline Result ════════════════════════════")
        print(f"  Rounds used          : {self.rounds_used}")
        print(f"  Combinations tried   : {self.total_sampled}")
        print(f"  Top combinations     : {self.count}")
        print(f"  Total portfolios     : {self.total_portfolios}  "
              f"({self.valid_portfolios} passed DD validation)")
        if self.combinations:
            rdd = [cr.raw_return_dd for cr in self.combinations]
            print(f"  Return/DD range      : {min(rdd):.2f} – {max(rdd):.2f}")
        print(f"  ═══════════════════════════════════════════════\n")

    def _dominant_rejection(self) -> str:
        totals = {
            "pearson"    : sum(s.rejected_pearson    for s in self.total_filter_stats),
            "spearman"   : sum(s.rejected_spearman   for s in self.total_filter_stats),
            "co_loss"    : sum(s.rejected_co_loss    for s in self.total_filter_stats),
            "tail_corr"  : sum(s.rejected_tail_corr  for s in self.total_filter_stats),
            "same_asset" : sum(s.rejected_same_asset for s in self.total_filter_stats),
            "dd"         : sum(s.rejected_dd         for s in self.total_filter_stats),
            "rolling"    : sum(s.rejected_rolling    for s in self.total_filter_stats),
        }
        return max(totals, key=totals.get)

    def warn_if_empty(self) -> None:
        if self.combinations:
            return
        dominant = self._dominant_rejection()
        labels = {
            "pearson"    : "Pearson correlation     → lower max_pearson_corr",
            "spearman"   : "Spearman correlation    → lower max_spearman_corr",
            "co_loss"    : "Co-loss frequency       → raise max_co_loss_freq",
            "tail_corr"  : "Tail correlation        → raise max_tail_corr or lower tail_percentile",
            "same_asset" : "Same-asset conflict     → increase same_asset_window_hours",
            "dd"         : "Total drawdown (DD check)→ raise total_drawdown_limit_pct",
            "rolling"    : "Rolling correlation     → raise max_rolling_corr / max_rolling_corr_recent",
        }
        print("\n  No combinations found after all rounds.")
        print(f"  Dominant rejection: {labels.get(dominant, dominant)}")
        print("  Adjust the threshold in PortfolioConfig and retry.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_pipeline(
    strategies: dict,
    config: PortfolioConfig,
    verbose: bool = True,
) -> PipelineResult:
    """
    Run the full portfolio generation pipeline.

    Filter order:
      1. Static correlation (Pearson, Spearman, co-loss, same-asset)
      2. Raw DD check        (equal-weight scaled → total-DD validation)
      3. Rolling correlation (36-month windows)
      4. Rank by return/DD  → top_combinations selected
      5. Apply 4 methods    → CombinationResult per combo
    """
    all_combos   : list[CombinationResult] = []
    all_stats    : list[FilterStats]       = []
    total_sampled: int = 0
    rounds_used  : int = 0

    universe = build_universe(
        strategies,
        same_asset_window_hours=config.same_asset_window_hours,
        verbose=verbose,
    )

    for round_n in range(config.max_rounds):
        rounds_used = round_n + 1
        round_seed  = (
            config.random_seed + round_n * 1000
            if config.random_seed is not None
            else None
        )
        round_config = PortfolioConfig(
            **{k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        )
        round_config.random_seed = round_seed

        if verbose:
            print(f"\n  ── Round {round_n + 1}/{config.max_rounds} "
                  f"(seed={round_seed}) ──────────────────────────")

        combinations = sample_combinations(strategies, round_config)
        total_sampled += len(combinations)

        # ── Step 1: Static correlation filters ────────────────────────────────
        static_passed, monthly_cache, stats = filter_static_only(
            combinations, strategies, round_config,
            verbose=verbose, universe=universe,
        )

        if not static_passed:
            all_stats.append(stats)
            if verbose:
                print("  No combinations survived static filters. Trying next round...")
            continue

        # ── Step 2: Raw DD check (before rolling — cheap to compute) ──────────
        dd_passed, equal_vps = _raw_dd_filter(
            static_passed, strategies, round_config, stats, verbose=verbose,
        )

        if not dd_passed:
            all_stats.append(stats)
            if verbose:
                print("  No combinations survived DD check. Trying next round...")
            continue

        # ── Step 3: Rolling correlation filters ────────────────────────────────
        rolling_passed, stats = filter_rolling_only(
            dd_passed, monthly_cache, round_config,
            verbose=verbose, stats=stats,
        )
        all_stats.append(stats)

        if not rolling_passed:
            if verbose:
                print("  No combinations survived rolling filters. Trying next round...")
            continue

        # ── Step 4: Rank by return/DD → top_combinations ──────────────────────
        top_ranked = _rank_combinations(
            rolling_passed, equal_vps, round_config, verbose=verbose,
        )

        if not top_ranked:
            if verbose:
                print("  No combinations to rank. Trying next round...")
            continue

        # ── Step 5: Apply all 4 weighting methods ─────────────────────────────
        combo_results = _build_combination_results(
            top_ranked, strategies, round_config, verbose=verbose,
        )
        all_combos.extend(combo_results)

        if verbose:
            n_valid = sum(cr.n_valid_methods for cr in combo_results)
            print(f"\n  Round {round_n + 1}: {len(combo_results)} combination(s), "
                  f"{n_valid} portfolios passed DD validation.")

        break  # first successful round — stop

    # Walk-forward equity (computed once, after pipeline, on final combo list)
    if all_combos:
        compute_all_wf_equities(all_combos, strategies, config, verbose=verbose)

    # Sort by fitness score descending
    all_combos.sort(key=lambda cr: cr.fitness_score, reverse=True)

    result = PipelineResult(
        combinations       = all_combos,
        total_sampled      = total_sampled,
        total_filter_stats = all_stats,
        rounds_used        = rounds_used,
        succeeded          = len(all_combos) > 0,
    )

    if verbose:
        result.report()
        result.warn_if_empty()

    return result
