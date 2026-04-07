"""
config.py — Configuration dataclass for portfolio generation.

All thresholds and parameters are defined here in one place.
Pass a PortfolioConfig instance through the entire pipeline.
"""

from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    # ── Account parameters ───────────────────────────────────────────────────
    account_balance: float = 60_000.0
    daily_loss_limit_pct: float = 0.045       # 4%  → prop firm default is 5%
    total_drawdown_limit_pct: float = 0.09   # 9%  → prop firm default is 10%
    base_risk_per_trade: float = 100.0       # $100/trade — matches backtest unit

    # ── Portfolio size ────────────────────────────────────────────────────────
    min_strategies: int = 6
    max_strategies: int = 7

    # ── Static correlation filter thresholds ─────────────────────────────────
    max_pearson_corr: float = 0.30           # monthly P&L, linear
    max_spearman_corr: float = 0.30          # monthly P&L, rank/monotonic
    max_co_loss_freq: float = 0.30           # fraction of months both strategies lose
    same_asset_window_hours: float = 8.0     # min hours between trades on the same asset (0 = disabled)
    tail_percentile: float = 0.30            # bottom N% of months used for tail correlation
    max_tail_corr: float = 0.30             # max allowed correlation during tail months

    # ── Filter toggles (set False to disable individual filters) ─────────────
    enable_co_loss_filter:   bool = True
    enable_tail_corr_filter: bool = True
    enable_same_asset_filter: bool = True
    enable_rolling_filter:   bool = True

    # ── Rolling correlation filter thresholds ────────────────────────────────
    rolling_window_months: int = 60          # size of each rolling window (months)
    max_rolling_corr: float = 0.40           # all windows must be below this
    max_rolling_corr_recent: float = 0.30    # windows within the recent period must be below this
    recent_years: int = 3                    # trailing years considered "recent"

    # ── Fitness function weights (must sum to 1.0) ───────────────────────────
    fitness_weight_return_dd:      float = 0.80  # return / max-drawdown ratio
    fitness_weight_annual_return:  float = 0.10  # annualised % return
    fitness_weight_winning_months: float = 0.10  # fraction of months with positive P&L

    # ── Random generation parameters ─────────────────────────────────────────
    n_portfolios: int = 250000                # number of random combinations to attempt
    max_rounds: int = 1                      # auto-regeneration rounds if all portfolios rejected
    random_seed: int | None = None
    top_combinations: int = 10               # top-N combos (by equal-weight return/DD) to weight

    # ── Genetic algorithm parameters ─────────────────────────────────────────
    ga_population_size: int = 300             # individuals per generation
    ga_generations: int = 100               # maximum number of generations
    ga_elite_fraction: float = 0.20          # top fraction that survive unchanged each generation
    ga_tournament_size: int = 4              # contestants per tournament selection
    ga_crossover_prob: float = 0.80          # probability of crossover vs random new individual
    ga_mutation_prob: float = 0.40           # probability of mutating an offspring
    ga_max_stagnation: int = 10              # stop early if best fitness unchanged for N generations
    ga_seed_attempts_multiplier: int = 2000 # random attempts per individual during seeding (target × this)
    ga_seed_stagnation: int = 10000          # stop seeding after this many consecutive attempts with no new combo found
    ga_use_pair_growth: bool = False         # use pair-growth seeding instead of clique-based seeding
    ga_use_greedy: bool = True              # use greedy independent-set seeding (recommended for large portfolios)

    # ── Parallelism ───────────────────────────────────────────────────────────
    n_workers: int = -1                      # -1 = all CPU cores, 1 = single-threaded (for debugging)

    # ── Walk-forward parameters ──────────────────────────────────────────────
    wf_is_years: int = 6                # in-sample lookback window (years)
    wf_oos_years: int = 3               # out-of-sample window per step (years)

    # ── Computed properties ──────────────────────────────────────────────────
    @property
    def daily_loss_limit_usd(self) -> float:
        """Maximum allowed daily loss in USD."""
        return self.account_balance * self.daily_loss_limit_pct

    @property
    def total_drawdown_limit_usd(self) -> float:
        """Maximum allowed total drawdown in USD."""
        return self.account_balance * self.total_drawdown_limit_pct

    def __post_init__(self) -> None:
        if self.min_strategies < 1:
            raise ValueError("min_strategies must be at least 1.")
        if self.max_strategies < self.min_strategies:
            raise ValueError("max_strategies must be >= min_strategies.")
        if not (0 < self.daily_loss_limit_pct < 1):
            raise ValueError("daily_loss_limit_pct must be between 0 and 1.")
        if not (0 < self.total_drawdown_limit_pct < 1):
            raise ValueError("total_drawdown_limit_pct must be between 0 and 1.")
        if not (0 < self.max_co_loss_freq <= 1):
            raise ValueError("max_co_loss_freq must be between 0 and 1.")
        if self.max_rolling_corr_recent > self.max_rolling_corr:
            raise ValueError("max_rolling_corr_recent must be <= max_rolling_corr.")
