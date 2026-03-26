"""
scaler.py — Critical day detection and position sizing rescale (Step 3).

Given a weighted portfolio (combined trades DataFrame with scaled P&L),
finds the single worst calendar day and derives a scale factor that makes
that day fit exactly within the daily loss limit.

This maximises position size: the portfolio runs as large as possible while
guaranteeing that the worst historical day never exceeded the daily limit.

Output: ScaledResult dataclass with all diagnostics + rescaled portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from alphaforge.portfolio.config import PortfolioConfig


@dataclass
class ScaledResult:
    """All outputs from the critical-day rescaling step."""
    combination     : tuple[str, ...]
    method          : str                    # weighting method used
    weights         : dict[str, float]       # w_i (sum to 1)
    critical_day    : date                   # date of worst day
    worst_day_loss  : float                  # P&L on that day (negative = loss, $)
    scale_factor    : float                  # daily_limit / abs(worst_day_loss)
    risk_per_trade  : dict[str, float]       # final $ risk per trade per strategy
    portfolio_df    : pd.DataFrame           # combined trades with rescaled P&L


def _daily_pnl(portfolio_df: pd.DataFrame) -> pd.Series:
    """
    Aggregate closed P&L by calendar day.
    Returns a Series indexed by date, sorted ascending.
    """
    return (
        portfolio_df
        .groupby(portfolio_df["Close time"].dt.date)["Profit/Loss"]
        .sum()
        .sort_index()
    )


def find_critical_day(portfolio_df: pd.DataFrame) -> tuple[date, float]:
    """
    Find the calendar day with the largest loss in the portfolio.

    Returns:
        (critical_day, worst_day_pnl)
        worst_day_pnl is negative if it was a losing day.
        If no losing day exists, returns the last date and 0.0.
    """
    daily = _daily_pnl(portfolio_df)

    worst_idx = daily.idxmin()
    worst_val = daily[worst_idx]

    return worst_idx, float(worst_val)


def rescale(
    combination: tuple[str, ...],
    method: str,
    weights: dict[str, float],
    portfolio_df: pd.DataFrame,
    config: PortfolioConfig,
) -> ScaledResult:
    """
    Detect the critical day and rescale the portfolio to fit within the
    daily loss limit.

    Scale factor:
        If the worst day lost more than the daily limit:
            scale_factor = daily_loss_limit_usd / abs(worst_day_loss)
        If the worst day is within the limit already:
            scale_factor = daily_loss_limit_usd / abs(worst_day_loss)
            → this will be > 1, meaning we can SIZE UP and still be safe.
        If no losing day exists:
            scale_factor = 1.0  (no adjustment possible or needed)

    The scale factor is applied uniformly to all P&L values.
    Final risk per trade for strategy i:
        base_risk × w_i × n_strategies × scale_factor

    Args:
        combination  : strategy names in the portfolio
        method       : weighting method label
        weights      : {name: w_i}, sum to 1
        portfolio_df : combined weighted trades (P&L already scaled by weights)
        config       : PortfolioConfig

    Returns:
        ScaledResult with all diagnostics and the rescaled DataFrame.
    """
    n = len(combination)
    critical_day, worst_day_loss = find_critical_day(portfolio_df)

    if worst_day_loss >= 0:
        # No losing days in history — no rescaling possible
        scale_factor = 1.0
    else:
        scale_factor = config.daily_loss_limit_usd / abs(worst_day_loss)

    # Apply scale factor to P&L
    scaled_df = portfolio_df.copy()
    scaled_df["Profit/Loss"] = scaled_df["Profit/Loss"] * scale_factor

    # Final risk per trade per strategy
    risk_per_trade = {
        name: round(config.base_risk_per_trade * weights[name] * n * scale_factor, 2)
        for name in combination
    }

    return ScaledResult(
        combination    = combination,
        method         = method,
        weights        = weights,
        critical_day   = critical_day,
        worst_day_loss = worst_day_loss,
        scale_factor   = scale_factor,
        risk_per_trade = risk_per_trade,
        portfolio_df   = scaled_df,
    )
