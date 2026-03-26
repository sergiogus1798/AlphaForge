"""
config.py — Configuration dataclass for portfolio generation.

All thresholds and parameters are defined here in one place.
Pass a PortfolioConfig instance through the entire pipeline.
"""

from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    # ── Account parameters ───────────────────────────────────────────────────
    account_balance: float = 10_000.0
    daily_loss_limit_pct: float = 0.04       # 4%  → prop firm default is 5%
    total_drawdown_limit_pct: float = 0.09   # 9%  → prop firm default is 10%
    base_risk_per_trade: float = 100.0       # $100/trade — matches backtest unit

    # ── Portfolio size ────────────────────────────────────────────────────────
    min_strategies: int = 2
    max_strategies: int = 8

    # ── Static correlation filter thresholds ─────────────────────────────────
    max_pearson_corr: float = 0.30           # monthly P&L, linear
    max_spearman_corr: float = 0.40          # monthly P&L, rank/monotonic
    max_co_loss_freq: float = 0.20           # fraction of months both strategies lose
    same_asset_same_day: bool = True         # True = same day; False = tighten to same bar

    # ── Rolling correlation filter thresholds ────────────────────────────────
    rolling_window_months: int = 36          # size of each rolling window (months)
    max_rolling_corr: float = 0.35           # all windows must be below this
    max_rolling_corr_recent: float = 0.30    # windows within the recent period must be below this
    recent_years: int = 3                    # trailing years considered "recent"

    # ── Generation parameters ────────────────────────────────────────────────
    n_portfolios: int = 50000                # number of random combinations to attempt
    max_rounds: int = 5                      # auto-regeneration rounds if all portfolios rejected
    random_seed: int | None = None
    top_combinations: int = 50               # top-N combos (by equal-weight return/DD) to weight

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
