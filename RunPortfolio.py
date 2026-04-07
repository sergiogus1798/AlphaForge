"""
RunPortfolio.py — Portfolio generation entry point.

Full pipeline:
  1. Load strategies from strategies/approved/
  2. Run generation pipeline (Steps 1–4): filter → weight → scale → validate
  3. Run MAE stress test on valid portfolios (Step 5)
  4. Launch Portfolio Explorer dashboard

Edit PortfolioConfig below to adjust all thresholds and parameters.
"""

import matplotlib.pyplot as plt

from alphaforge.loader.loader import load_folder
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.pipeline import run_pipeline
from alphaforge.portfolio.stress.mae_stress import stress_test_all
from alphaforge.portfolio.dashboard import run_portfolio_dashboard
from alphaforge.portfolio.dashboard_corr import run_correlation_dashboard

# ── Configuration ─────────────────────────────────────────────────────────────

config = PortfolioConfig(
    # Portfolio size
    min_strategies           = 3,
    max_strategies           = 10,

    # Static correlation filters
    max_pearson_corr         = 0.30,
    max_spearman_corr        = 0.40,
    max_co_loss_freq         = 0.10,
    same_asset_window_hours  = 8.0,

    # Rolling correlation filters
    rolling_window_months    = 12,
    max_rolling_corr         = 0.35,
    max_rolling_corr_recent  = 0.30,
    recent_years             = 3,

    # Generation
    n_portfolios             = 1_000,
    max_rounds               = 5,
    random_seed              = 42,
)

# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  AlphaForge — Portfolio Generation")
print("=" * 65)

# Load
strategies = load_folder("strategies/approved")
print(f"\n  {len(strategies)} strategies loaded.\n")

# Pipeline (Steps 1–4)
result = run_pipeline(strategies, config, verbose=True)

if not result.succeeded:
    print("  No combinations generated. Adjust thresholds in RunPortfolio.py.")
else:
    # Step 5: MAE stress test (informational)
    stress_test_all(result.combinations, config, verbose=True)

    # Launch dashboard
    # Build all windows first (non-blocking), then show them all together
    run_portfolio_dashboard(result.combinations, config, strategies=strategies, _show=False)
    print("  Launching deep correlation analysis...")
    run_correlation_dashboard(result.combinations, config, strategies, _show=False)
    plt.show()   # blocks here; all windows open simultaneously
