"""
validator.py — Drawdown validation after rescaling (Step 4).

Takes a ScaledResult (already rescaled to fit the daily loss limit) and checks
whether the full equity curve stays within the total drawdown limit.

If it passes, returns a ValidPortfolio with full metrics pre-computed.
If it fails, returns None.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.scaler import ScaledResult
from alphaforge.metrics.metrics import compute_metrics


@dataclass
class ValidPortfolio:
    """A portfolio that has passed all filters, weighting, rescaling, and DD validation."""

    scaled          : ScaledResult
    max_drawdown_usd: float
    max_drawdown_pct: float
    metrics         : dict                    # full compute_metrics output on scaled portfolio
    stress          : object | None = None    # filled later by mae_stress (StressResult)

    # ── Convenience pass-throughs ─────────────────────────────────────────────

    @property
    def combination(self) -> tuple[str, ...]:
        return self.scaled.combination

    @property
    def method(self) -> str:
        return self.scaled.method

    @property
    def weights(self) -> dict[str, float]:
        return self.scaled.weights

    @property
    def scale_factor(self) -> float:
        return self.scaled.scale_factor

    @property
    def critical_day(self):
        return self.scaled.critical_day

    @property
    def worst_day_loss(self) -> float:
        return self.scaled.worst_day_loss

    @property
    def risk_per_trade(self) -> dict[str, float]:
        return self.scaled.risk_per_trade

    @property
    def portfolio_df(self) -> pd.DataFrame:
        return self.scaled.portfolio_df

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe_ratio", float("nan"))

    @property
    def label(self) -> str:
        method_label = self.method.replace("_", " ").title()
        n = len(self.combination)
        return f"{method_label} | {n} strats | Sharpe {self.sharpe:.2f}"


def _compute_drawdown(portfolio_df: pd.DataFrame, initial_capital: float) -> tuple[float, float]:
    """
    Compute max drawdown (USD and %) on the rescaled equity curve.

    Returns:
        (max_drawdown_usd, max_drawdown_pct)
    """
    daily = (
        portfolio_df
        .groupby(portfolio_df["Close time"].dt.date)["Profit/Loss"]
        .sum()
    )
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    equity = daily.reindex(date_range, fill_value=0.0).cumsum()

    total_equity = initial_capital + equity
    peak = total_equity.cummax()
    dd_usd = (peak - total_equity).max()
    dd_pct = (dd_usd / peak[( peak - total_equity).idxmax()]) * 100

    return float(dd_usd), float(dd_pct)


def validate(
    scaled: ScaledResult,
    config: PortfolioConfig,
) -> ValidPortfolio | None:
    """
    Validate a rescaled portfolio against the total drawdown limit.

    Computes the full equity curve after rescaling and rejects the portfolio
    if max drawdown exceeds config.total_drawdown_limit_usd.

    Args:
        scaled : ScaledResult from the rescaling step
        config : PortfolioConfig with total_drawdown_limit

    Returns:
        ValidPortfolio if it passes, None if it fails.
    """
    dd_usd, dd_pct = _compute_drawdown(scaled.portfolio_df, config.account_balance)

    if dd_usd > config.total_drawdown_limit_usd:
        return None

    metrics = compute_metrics(scaled.portfolio_df, initial_capital=config.account_balance)

    return ValidPortfolio(
        scaled           = scaled,
        max_drawdown_usd = dd_usd,
        max_drawdown_pct = dd_pct,
        metrics          = metrics,
    )
