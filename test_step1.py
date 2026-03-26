"""
test_step1.py — Quick test of the Step 1 portfolio generation pipeline.

Loads all strategies from TradingData/, samples combinations, runs static +
rolling correlation filters, reports results, and plots rolling correlations
for the static survivors so you can inspect why they failed the rolling stage.
"""

from alphaforge.loader.loader import load_folder
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.sampler import sample_combinations
from alphaforge.portfolio.generator.filters import (
    filter_combinations,
    filter_static_only,
    plot_rolling_correlations,
)

# ── Config ────────────────────────────────────────────────────────────────────

config = PortfolioConfig(random_seed=42)

# ── Load strategies ───────────────────────────────────────────────────────────

print("=" * 60)
print("  AlphaForge — Step 1 Test")
print("=" * 60)

strategies = load_folder("TradingData")
print(f"\n  {len(strategies)} strategies loaded.\n")

# ── Sample combinations ───────────────────────────────────────────────────────

combinations = sample_combinations(strategies, config)
print(f"  {len(combinations)} combinations sampled "
      f"(sizes {config.min_strategies}–{config.max_strategies}).\n")

# ── Static filters (kept separate so we can inspect survivors) ────────────────

static_passed, monthly_cache, _ = filter_static_only(combinations, strategies, config)

# ── Plot rolling correlations for the static survivors ────────────────────────

if static_passed:
    print(f"  Plotting rolling correlations for up to 3 static survivors...\n")
    plot_rolling_correlations(static_passed, monthly_cache, config, max_combos=3)

# ── Full filter run (static + rolling) ───────────────────────────────────────

passed, stats = filter_combinations(combinations, strategies, config, verbose=True)

# ── Results ───────────────────────────────────────────────────────────────────

print(f"  Final result: {len(passed)} portfolio(s) passed all filters.\n")

if passed:
    print("  Passing combinations:")
    for i, combo in enumerate(passed, 1):
        print(f"    {i:>3}. {' | '.join(combo)}")
else:
    print("  No portfolios passed. Consider relaxing the thresholds in PortfolioConfig.")
