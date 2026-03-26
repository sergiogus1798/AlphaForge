"""
mae_stress.py — MAE stress test for validated portfolios (Step 5).

Replaces each trade's closed P&L with its Maximum Adverse Excursion (MAE),
then re-checks whether the daily loss limit and total drawdown limit survive.

This answers: "If every trade hit its worst floating loss simultaneously,
would this portfolio still be within prop firm limits?"

The scale factor from Step 3 is preserved — we only swap the P&L values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.validator import ValidPortfolio


@dataclass
class StressResult:
    """Results of the MAE stress test on a single portfolio."""
    passed               : bool
    worst_day_loss_mae   : float    # worst daily loss using MAE values ($)
    critical_day_mae     : date     # date of that worst day
    max_drawdown_usd_mae : float    # max drawdown under MAE scenario ($)
    max_drawdown_pct_mae : float    # max drawdown under MAE scenario (%)
    trades_with_mae      : int      # how many trades had a valid MAE column
    trades_total         : int      # total trades in portfolio
    mae_coverage_pct     : float    # % of trades where MAE was available


def _find_mae_column(df: pd.DataFrame) -> str | None:
    """Return the name of the MAE column, or None if not present."""
    for col in df.columns:
        if "MAE" in col.upper() and "MFE" not in col.upper():
            return col
    return None


def _build_mae_portfolio(
    portfolio_df: pd.DataFrame,
    scale_factor: float,
) -> tuple[pd.DataFrame, int, int]:
    """
    Build a copy of the portfolio DataFrame with Profit/Loss replaced by
    scaled MAE values (worst-case floating loss).

    Trades without a valid MAE keep their original scaled P&L as fallback.

    Returns:
        (mae_df, trades_with_mae, trades_total)
    """
    df = portfolio_df.copy()
    mae_col = _find_mae_column(df)
    total = len(df)
    with_mae = 0

    if mae_col is None:
        # No MAE column at all — stress test is not possible
        return df, 0, total

    valid_mae = df[mae_col].notna() & (df[mae_col] != 0)
    with_mae = int(valid_mae.sum())

    # MAE is the adverse excursion — ensure it's treated as a loss (negative)
    # Scale by the same scale_factor applied during Step 3
    df.loc[valid_mae, "Profit/Loss"] = -df.loc[valid_mae, mae_col].abs() * scale_factor

    return df, with_mae, total


def _daily_pnl_series(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby(df["Close time"].dt.date)["Profit/Loss"]
        .sum()
        .sort_index()
    )


def _max_drawdown(df: pd.DataFrame, initial_capital: float) -> tuple[float, float]:
    daily = _daily_pnl_series(df)
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    equity = daily.reindex(date_range, fill_value=0.0).cumsum()
    total_equity = initial_capital + equity
    peak = total_equity.cummax()
    dd_usd = (peak - total_equity).max()
    dd_pct = (dd_usd / peak[(peak - total_equity).idxmax()]) * 100
    return float(dd_usd), float(dd_pct)


def run_mae_stress(
    portfolio: ValidPortfolio,
    config: PortfolioConfig,
) -> StressResult:
    """
    Run the MAE stress test on a validated portfolio.

    Args:
        portfolio : ValidPortfolio from the generation pipeline
        config    : PortfolioConfig with daily and total DD limits

    Returns:
        StressResult — always returned even if MAE data is unavailable
        (passed=True with a warning note in mae_coverage_pct=0)
    """
    mae_df, with_mae, total = _build_mae_portfolio(
        portfolio.portfolio_df,
        portfolio.scale_factor,
    )

    daily = _daily_pnl_series(mae_df)
    critical_day = daily.idxmin()
    worst_day    = float(daily[critical_day])

    dd_usd, dd_pct = _max_drawdown(mae_df, config.account_balance)

    daily_ok = worst_day >= -config.daily_loss_limit_usd
    total_ok  = dd_usd <= config.total_drawdown_limit_usd
    passed    = daily_ok and total_ok

    coverage = (with_mae / total * 100) if total > 0 else 0.0

    return StressResult(
        passed               = passed,
        worst_day_loss_mae   = worst_day,
        critical_day_mae     = critical_day,
        max_drawdown_usd_mae = dd_usd,
        max_drawdown_pct_mae = dd_pct,
        trades_with_mae      = with_mae,
        trades_total         = total,
        mae_coverage_pct     = coverage,
    )


def stress_test_all(
    combinations,           # list[CombinationResult]
    config: PortfolioConfig,
    verbose: bool = True,
) -> None:
    """
    Run MAE stress test on every ValidPortfolio across all CombinationResults.
    Results are attached as vp.stress (StressResult).
    Informational only — no portfolios are removed.
    """
    from tqdm import tqdm
    from alphaforge.portfolio.generator.combo_result import METHODS

    total = sum(
        1 for cr in combinations
        for vp in cr.portfolios.values()
        if vp is not None
    )
    passed_count = 0

    with tqdm(total=total, desc="  MAE stress test", unit="portfolio", disable=not verbose) as bar:
        for cr in combinations:
            for method in METHODS:
                vp = cr.portfolios.get(method)
                if vp is None:
                    continue
                result = run_mae_stress(vp, config)
                vp.stress = result
                if result.passed:
                    passed_count += 1
                bar.update(1)

    if verbose:
        print(f"\n  MAE stress: {passed_count}/{total} portfolios would pass worst-case limits "
              f"(informational — no portfolios removed).\n")
