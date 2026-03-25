"""
metrics.py — Per-strategy performance metrics.

All public functions accept a DataFrame of trades with at least:
    - 'Profit/Loss'  (float)
    - 'Open time'    (datetime)
    - 'Close time'   (datetime)

compute_metrics() is the main entry point — returns a dict with all 19 metrics.
metrics_table()   wraps multiple strategies into a single comparison DataFrame.
"""

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sorted_pnl(df: pd.DataFrame) -> pd.Series:
    """Trade PnL sorted chronologically by close time."""
    return df.sort_values("Close time")["Profit/Loss"].reset_index(drop=True)


def _equity_curve(pnl: pd.Series) -> pd.Series:
    return pnl.cumsum()


def _date_range_years(df: pd.DataFrame) -> float:
    """Total duration of the strategy in fractional years."""
    start = df["Open time"].min()
    end = df["Close time"].max()
    return (end - start).total_seconds() / (365.25 * 24 * 3600)


def _wins(pnl: pd.Series) -> pd.Series:
    return pnl[pnl > 0]


def _losses(pnl: pd.Series) -> pd.Series:
    return pnl[pnl < 0]


def _max_consecutive(mask: pd.Series) -> int:
    """Longest consecutive streak of True values in a boolean Series."""
    max_streak = current = 0
    for val in mask:
        if val:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


# ── Individual metric functions ───────────────────────────────────────────────

def total_profit(df: pd.DataFrame) -> float:
    return df["Profit/Loss"].sum()


def num_trades(df: pd.DataFrame) -> int:
    return len(df)


def gross_profit(df: pd.DataFrame) -> float:
    return df["Profit/Loss"].clip(lower=0).sum()


def gross_loss(df: pd.DataFrame) -> float:
    """Returned as a positive number representing total loss magnitude."""
    return abs(df["Profit/Loss"].clip(upper=0).sum())


def average_win(df: pd.DataFrame) -> float:
    wins = _wins(df["Profit/Loss"])
    return wins.mean() if len(wins) > 0 else 0.0


def average_loss(df: pd.DataFrame) -> float:
    """Returned as a positive number."""
    losses = _losses(df["Profit/Loss"])
    return abs(losses.mean()) if len(losses) > 0 else 0.0


def winning_percentage(df: pd.DataFrame) -> float:
    """Percentage of trades that are winners."""
    wins = (df["Profit/Loss"] > 0).sum()
    return (wins / len(df) * 100) if len(df) > 0 else 0.0


def profit_factor(df: pd.DataFrame) -> float:
    gp = gross_profit(df)
    gl = gross_loss(df)
    return (gp / gl) if gl != 0 else float("inf")


def average_trade(df: pd.DataFrame) -> float:
    """Average PnL per trade."""
    return df["Profit/Loss"].mean() if len(df) > 0 else 0.0


def max_drawdown(df: pd.DataFrame) -> float:
    """
    Maximum drawdown in currency units (positive = loss magnitude).
    Computed on the chronologically-sorted equity curve.
    """
    pnl = _sorted_pnl(df)
    equity = _equity_curve(pnl)
    peak = equity.cummax()
    return (peak - equity).max()


def pct_drawdown(df: pd.DataFrame, initial_capital: float = 10_000) -> float:
    """
    Maximum drawdown as a percentage of the peak equity at the moment
    the drawdown occurs (initial capital + cumulative PnL at that peak).

    This matches StrategyQuant X's % Drawdown calculation.
    Note: a small residual difference (~0.1%) may remain because SQX
    also accounts for open trade equity, which is not available in the CSV.
    """
    pnl = _sorted_pnl(df)
    equity = _equity_curve(pnl)
    cummax = equity.cummax()
    dd_series = cummax - equity
    max_dd_idx = dd_series.idxmax()
    peak_equity_at_dd = initial_capital + cummax[max_dd_idx]
    dd = dd_series[max_dd_idx]
    return (dd / peak_equity_at_dd * 100) if peak_equity_at_dd != 0 else float("nan")


def return_dd_ratio(df: pd.DataFrame) -> float:
    """Total profit divided by max drawdown."""
    dd = max_drawdown(df)
    return (total_profit(df) / dd) if dd != 0 else float("inf")


def sharpe_ratio(df: pd.DataFrame, *, risk_free_rate: float = 0.01) -> float:
    """
    Annualised Sharpe ratio using daily PnL as the return series.

    Formula: (mean_daily_pnl - rf_daily) / std_daily_pnl * sqrt(252)

    risk_free_rate is an annual decimal (default 1% = 0.01), converted to a
    daily dollar adjustment using initial_capital=$10,000 and 252 trading days.
    Business days (Mon-Fri) with no closed trades count as zero-return days.
    """
    if "Close time" not in df.columns:
        return float("nan")

    daily = (
        df.groupby(df["Close time"].dt.date)["Profit/Loss"]
        .sum()
        .reindex(
            pd.bdate_range(
                df["Open time"].min().date(),
                df["Close time"].max().date(),
            ).date,
            fill_value=0.0,
        )
    )

    std = daily.std()
    if std == 0 or np.isnan(std):
        return float("nan")

    rf_daily = 10_000 * risk_free_rate / 252
    return ((daily.mean() - rf_daily) / std) * np.sqrt(252)


def yearly_avg_pct_return(df: pd.DataFrame, initial_capital: float = 10_000) -> float:
    """Average annual return as a percentage of initial capital."""
    years = _date_range_years(df)
    if years == 0:
        return float("nan")
    return (total_profit(df) / initial_capital / years) * 100


def cagr(df: pd.DataFrame, initial_capital: float = 10_000) -> float:
    """Compound Annual Growth Rate as a percentage."""
    years = _date_range_years(df)
    if years == 0:
        return float("nan")
    final = initial_capital + total_profit(df)
    if final <= 0 or initial_capital <= 0:
        return float("nan")
    return ((final / initial_capital) ** (1 / years) - 1) * 100


def annual_pct_max_dd(df: pd.DataFrame, initial_capital: float = 10_000) -> float:
    """Yearly avg % return divided by % drawdown — reward-to-risk efficiency."""
    yr = yearly_avg_pct_return(df, initial_capital)
    pdd = pct_drawdown(df, initial_capital)
    return (yr / pdd) if pdd != 0 else float("inf")


def r_expectancy(df: pd.DataFrame) -> float:
    """
    R Expectancy — expected return per trade expressed in units of average loss (R).

    Formula: (win_rate * avg_win / avg_loss) - loss_rate
    """
    pnl = df["Profit/Loss"]
    win_rate = (pnl > 0).mean()
    loss_rate = 1 - win_rate
    avg_w = average_win(df)
    avg_l = average_loss(df)
    if avg_l == 0:
        return float("nan")
    return (win_rate * avg_w / avg_l) - loss_rate


def max_consecutive_wins(df: pd.DataFrame) -> int:
    pnl = _sorted_pnl(df)
    return _max_consecutive(pnl > 0)


def max_consecutive_losses(df: pd.DataFrame) -> int:
    pnl = _sorted_pnl(df)
    return _max_consecutive(pnl < 0)


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, *, initial_capital: float = 10_000) -> dict:
    """
    Compute all 19 metrics for a single strategy DataFrame.

    Args:
        df:               Clean trades DataFrame from the loader.
        initial_capital:  Starting account balance in USD (default: $10,000).
                          Used for percentage-based and CAGR metrics.

    Returns:
        Dict with all metrics. Monetary values in USD, rates as %.
    """
    return {
        # Summary
        "total_profit":          round(total_profit(df), 2),
        "yearly_avg_pct_return": round(yearly_avg_pct_return(df, initial_capital), 2),
        "cagr":                  round(cagr(df, initial_capital), 2),
        # Cards
        "num_trades":            num_trades(df),
        "sharpe_ratio":          round(sharpe_ratio(df), 4),
        "profit_factor":         round(profit_factor(df), 2),
        "return_dd_ratio":       round(return_dd_ratio(df), 2),
        "winning_percentage":    round(winning_percentage(df), 2),
        "drawdown":              round(max_drawdown(df), 2),
        "pct_drawdown":          round(pct_drawdown(df, initial_capital), 2),
        "average_trade":         round(average_trade(df), 2),
        "annual_pct_max_dd":     round(annual_pct_max_dd(df, initial_capital), 2),
        "r_expectancy":          round(r_expectancy(df), 4),
        # Trades
        "gross_profit":          round(gross_profit(df), 2),
        "gross_loss":            round(gross_loss(df), 2),
        "average_win":           round(average_win(df), 2),
        "average_loss":          round(average_loss(df), 2),
        "max_consecutive_wins":  max_consecutive_wins(df),
        "max_consecutive_losses": max_consecutive_losses(df),
    }


def metrics_table(
    strategies: dict[str, pd.DataFrame],
    *,
    initial_capital: float = 10_000,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per strategy and all 19 metrics as columns.
    Monetary values in USD. Pass initial_capital (USD) to override the default $10,000.
    """
    rows = []
    for name, df in strategies.items():
        row = {"strategy": name}
        row.update(compute_metrics(df, initial_capital=initial_capital))
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")
