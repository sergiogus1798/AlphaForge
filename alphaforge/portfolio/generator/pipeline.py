"""
pipeline.py — Full portfolio generation pipeline orchestrator (Steps 1–4).

Ties together: sampler → filters → weighting → scaler → validator.

Auto-regeneration: if a round produces 0 valid portfolios, re-samples with a
different random seed up to config.max_rounds times. If still 0 after all
rounds, reports which filter stage dominated rejections so the user can adjust
thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tqdm import tqdm

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.sampler import sample_combinations
from alphaforge.portfolio.generator.filters import filter_combinations, FilterStats
from alphaforge.portfolio.generator.weighting import (
    compute_all_weights,
    build_weighted_portfolio,
    METHODS,
)
from alphaforge.portfolio.generator.scaler import rescale
from alphaforge.portfolio.generator.validator import validate, ValidPortfolio


@dataclass
class PipelineResult:
    """Summary of a full pipeline run."""
    valid_portfolios       : list[ValidPortfolio]
    total_sampled          : int
    total_filter_stats     : list[FilterStats]
    rounds_used            : int
    succeeded              : bool

    @property
    def count(self) -> int:
        return len(self.valid_portfolios)

    def report(self) -> None:
        print(f"\n  ══ Pipeline Result ════════════════════════════")
        print(f"  Rounds used       : {self.rounds_used}")
        print(f"  Combinations tried: {self.total_sampled}")
        print(f"  Valid portfolios  : {self.count}")
        if self.valid_portfolios:
            sharpes = [vp.sharpe for vp in self.valid_portfolios]
            print(f"  Sharpe range      : {min(sharpes):.2f} – {max(sharpes):.2f}")
        print(f"  ═══════════════════════════════════════════════\n")

    def _dominant_rejection(self) -> str:
        """Identify which filter stage caused the most rejections overall."""
        totals = {
            "pearson"    : sum(s.rejected_pearson     for s in self.total_filter_stats),
            "spearman"   : sum(s.rejected_spearman    for s in self.total_filter_stats),
            "co_loss"    : sum(s.rejected_co_loss     for s in self.total_filter_stats),
            "same_asset" : sum(s.rejected_same_asset  for s in self.total_filter_stats),
            "rolling"    : sum(s.rejected_rolling     for s in self.total_filter_stats),
        }
        return max(totals, key=totals.get)

    def warn_if_empty(self) -> None:
        if self.valid_portfolios:
            return
        dominant = self._dominant_rejection()
        labels = {
            "pearson"   : "Pearson correlation     → lower max_pearson_corr",
            "spearman"  : "Spearman correlation    → lower max_spearman_corr",
            "co_loss"   : "Co-loss frequency       → raise max_co_loss_freq",
            "same_asset": "Same-asset conflict     → set same_asset_same_day=False",
            "rolling"   : "Rolling correlation     → raise max_rolling_corr / max_rolling_corr_recent",
        }
        print("\n  ⚠  No valid portfolios found after all rounds.")
        print(f"  Dominant rejection: {labels.get(dominant, dominant)}")
        print("  Adjust the threshold in PortfolioConfig and retry.\n")


def run_pipeline(
    strategies: dict,
    config: PortfolioConfig,
    verbose: bool = True,
) -> PipelineResult:
    """
    Run the full portfolio generation pipeline.

    Steps per round:
      1. Sample combinations (sampler)
      2. Filter — static + rolling (filters)
      3. For each survivor × each weighting method:
           a. Compute weights
           b. Build weighted portfolio DataFrame
           c. Rescale to daily loss limit
           d. Validate against total drawdown limit
      4. Collect valid portfolios

    If 0 valid portfolios after a round AND more rounds remain, re-sample
    with an offset seed and retry.

    Args:
        strategies : dict name → DataFrame (pre-loaded strategies)
        config     : PortfolioConfig
        verbose    : print progress

    Returns:
        PipelineResult with all valid portfolios and diagnostics.
    """
    all_valid      : list[ValidPortfolio] = []
    all_stats      : list[FilterStats]   = []
    total_sampled  : int = 0
    rounds_used    : int = 0

    for round_n in range(config.max_rounds):
        rounds_used = round_n + 1

        # Vary seed each round so we don't re-sample identical combinations
        round_seed = (
            config.random_seed + round_n * 1000
            if config.random_seed is not None
            else None
        )
        round_config = PortfolioConfig(
            **{k: v for k, v in config.__dict__.items() if not k.startswith("_")},
        )
        round_config.random_seed = round_seed

        if verbose:
            print(f"\n  ── Round {round_n + 1}/{config.max_rounds} "
                  f"(seed={round_seed}) ──────────────────────────")

        combinations = sample_combinations(strategies, round_config)
        total_sampled += len(combinations)

        passed, stats = filter_combinations(combinations, strategies, round_config, verbose=verbose)
        all_stats.append(stats)

        if not passed:
            if verbose:
                print("  No combinations survived filters. Trying next round...")
            continue

        # Steps 2–4: weight → scale → validate
        round_valid: list[ValidPortfolio] = []

        with tqdm(
            passed,
            desc="  Weighting & validating",
            unit="combo",
            disable=not verbose,
        ) as bar:
            for combo in bar:
                weights_all = compute_all_weights(combo, strategies)
                for method in METHODS:
                    weights    = weights_all[method]
                    portfolio  = build_weighted_portfolio(combo, strategies, weights)
                    scaled     = rescale(combo, method, weights, portfolio, config)
                    valid      = validate(scaled, config)
                    if valid:
                        round_valid.append(valid)
                bar.set_postfix(valid=len(round_valid))

        all_valid.extend(round_valid)

        if verbose:
            print(f"\n  Round {round_n + 1}: {len(round_valid)} valid portfolio(s) found.")

        if all_valid:
            break

    # Sort by Sharpe descending
    all_valid.sort(key=lambda vp: vp.sharpe, reverse=True)

    result = PipelineResult(
        valid_portfolios   = all_valid,
        total_sampled      = total_sampled,
        total_filter_stats = all_stats,
        rounds_used        = rounds_used,
        succeeded          = len(all_valid) > 0,
    )

    if verbose:
        result.report()
        result.warn_if_empty()

    return result
