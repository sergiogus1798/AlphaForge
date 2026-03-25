"""
portfolio.py — Combines multiple strategies into an aggregate portfolio.
"""

import pandas as pd
from alphaforge.metrics import compute_metrics


def build_equity_curve(df: pd.DataFrame, time_col: str = "Close time") -> pd.Series:
    """
    Build a cumulative equity curve from a trades DataFrame,
    sorted by close time.
    """
    sorted_df = df.sort_values(time_col)
    return sorted_df["Profit/Loss"].cumsum().reset_index(drop=True)


def combine_strategies(strategies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all strategy DataFrames into one combined DataFrame.
    Each row retains its 'strategy' tag.
    """
    if not strategies:
        raise ValueError("No strategies to combine.")
    return pd.concat(strategies.values(), ignore_index=True)


def portfolio_equity_curve(combined: pd.DataFrame) -> pd.Series:
    """
    Produce the combined portfolio equity curve by sorting all trades
    chronologically and accumulating PnL.
    """
    return build_equity_curve(combined)


def portfolio_metrics(strategies: dict[str, pd.DataFrame], *, initial_capital: float = 10_000) -> dict:
    """Compute aggregate metrics across all strategies combined. Monetary values in USD."""
    combined = combine_strategies(strategies)
    return {
        "strategies_count": len(strategies),
        **compute_metrics(combined, initial_capital=initial_capital),
    }


def correlation_matrix(strategies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute the correlation matrix of per-trade PnL series across strategies.
    Useful for understanding diversification between strategies.
    """
    series = {}
    for name, df in strategies.items():
        series[name] = df["Profit/Loss"].reset_index(drop=True)
    pnl_df = pd.DataFrame(series)
    return pnl_df.corr()
