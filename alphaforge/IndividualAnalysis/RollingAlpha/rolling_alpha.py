"""
rolling_alpha.py — Rolling OLS alpha/beta decomposition.

For each month-end date, fits OLS over a trailing lookback window:
    r_strategy = α + β · r_asset

Returns a DataFrame with one row per month containing:
  - alpha       daily alpha (intercept), in decimal
  - alpha_ann   alpha annualised (× 252)
  - beta        OLS slope
  - alpha_tstat t-statistic on alpha
  - r_squared   OLS R²
  - n_obs       observations used

The step is monthly (≈22 trading days) so computation stays cheap.
The lookback defaults to 252 days (1 year) — enough for stable estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

_MIN_OBS = 60   # minimum observations required to fit OLS


def compute_rolling_alpha(
    aligned:  pd.DataFrame,
    lookback: int = 252,
    step:     int = 22,
) -> pd.DataFrame:
    """
    Compute rolling OLS alpha and beta at monthly intervals.

    Args:
        aligned:  DataFrame with columns 'strategy' and 'asset' (daily returns).
        lookback: Trailing window size in trading days (default 252 = 1 year).
        step:     Step between windows in trading days (default 22 ≈ 1 month).

    Returns:
        DataFrame indexed by date with columns:
          alpha, alpha_ann, beta, alpha_tstat, r_squared, n_obs
    """
    data = aligned[["strategy", "asset"]].dropna()
    idx  = data.index

    records = []
    # Walk forward by step, ending at each month-end observation
    positions = range(lookback, len(idx) + 1, step)
    if len(idx) not in positions:
        positions = list(positions) + [len(idx)]

    for end in positions:
        start = max(0, end - lookback)
        chunk = data.iloc[start:end]

        if len(chunk) < _MIN_OBS:
            continue

        y = chunk["strategy"].values
        x = chunk["asset"].values
        X = sm.add_constant(x)

        try:
            res   = sm.OLS(y, X).fit()
            alpha = float(res.params[0])
            beta  = float(res.params[1])
            at    = float(res.tvalues[0])
            r2    = float(res.rsquared)
        except Exception:
            continue

        records.append({
            "date":        idx[end - 1],
            "alpha":       alpha,
            "alpha_ann":   alpha * 252,
            "beta":        beta,
            "alpha_tstat": at,
            "r_squared":   r2,
            "n_obs":       len(chunk),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")
    return df
