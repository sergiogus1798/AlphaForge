"""
wf.py — Walk-forward equity computation for non-trivial weighting methods.

For each combination + method (min_variance, risk_parity, hrp):
  1. Generate non-overlapping OOS windows (anchored to Jan 1st, step = wf_oos_years).
     IS for each window = wf_is_years of lookback before the OOS start.
  2. For each window: compute weights from IS daily P&L, apply to OOS trades.
  3. Scale OOS P&L using the same scale_factor as the static method (for comparability).
  4. Stitch all OOS windows -> one continuous equity curve (cumsum, starts from 0).

Equal weight is excluded — its WF curve is identical to the static curve.
Returns None if no OOS windows can be generated.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.weighting import (
    build_daily_pnl,
    min_variance,
    risk_parity,
    hrp,
    equal_weight,
)
from alphaforge.metrics.metrics import compute_metrics

WF_METHODS = ("min_variance", "risk_parity", "hrp")


# ── Window generation ─────────────────────────────────────────────────────────

def _generate_windows(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    is_years: int,
    oos_years: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate (is_start, oos_start, oos_end) triplets.

    - oos_start lands on Jan 1st, steps forward by oos_years each iteration.
    - First oos_start: the first Jan 1st >= min_date + is_years.
    - oos_end: oos_start + oos_years, clipped to max_date.
    - Windows stop when oos_start >= max_date (no OOS data available).
    """
    is_offset  = pd.DateOffset(years=is_years)
    oos_offset = pd.DateOffset(years=oos_years)

    first_possible = min_date + is_offset
    # Ceil to next Jan 1st
    if first_possible.month == 1 and first_possible.day == 1:
        first_jan = pd.Timestamp(first_possible.year, 1, 1)
    else:
        first_jan = pd.Timestamp(first_possible.year + 1, 1, 1)

    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    oos_start = first_jan
    while oos_start < max_date:
        is_start = oos_start - is_offset
        oos_end  = min(oos_start + oos_offset, max_date)
        windows.append((is_start, oos_start, oos_end))
        oos_start = oos_start + oos_offset

    return windows


# ── Weight computation for one IS window ─────────────────────────────────────

def _is_weights(
    names: list[str],
    strategies: dict,
    is_start: pd.Timestamp,
    oos_start: pd.Timestamp,
    method: str,
) -> dict[str, float]:
    """
    Compute weights from IS trades [is_start, oos_start).

    Falls back to equal weight if fewer than 2 strategies have IS data.
    """
    is_dfs: dict[str, pd.DataFrame] = {}
    for name in names:
        df = strategies[name]
        mask = (df["Close time"] >= is_start) & (df["Close time"] < oos_start)
        is_df = df.loc[mask]
        if len(is_df) > 0:
            is_dfs[name] = is_df

    is_names = list(is_dfs.keys())
    if len(is_names) < 2:
        return equal_weight(names)

    pnl = build_daily_pnl(is_dfs, is_names)

    if method == "min_variance":
        partial_w = min_variance(pnl)
    elif method == "risk_parity":
        partial_w = risk_parity(pnl)
    elif method == "hrp":
        partial_w = hrp(pnl)
    else:
        return equal_weight(names)

    # Strategies absent from IS get weight 0; renormalise
    weights = {n: partial_w.get(n, 0.0) for n in names}
    total = sum(weights.values())
    if total <= 0:
        return equal_weight(names)
    return {n: v / total for n, v in weights.items()}


# ── Main function ─────────────────────────────────────────────────────────────

def compute_wf_equity(
    combination: tuple[str, ...],
    strategies: dict,
    config: PortfolioConfig,
    method: str,
    scale_factor: float,
    static_weights: dict[str, float] | None = None,
) -> tuple[pd.Series | None, list[dict]]:
    """
    Compute a walk-forward equity curve and weight history for one method.

    The curve covers the FULL data history:
      - IS period (before first OOS window): static method weights applied
        to all trades, so the curve starts from day 1 of the data.
      - OOS windows: WF weights (trained on IS data before each window).

    This gives a continuous equity from start to finish, making metrics
    directly comparable to the static curve.

    Args:
        combination    : strategy names
        strategies     : name -> DataFrame (with Close time + Profit/Loss columns)
        config         : PortfolioConfig (reads wf_is_years, wf_oos_years)
        method         : "min_variance", "risk_parity", or "hrp"
        scale_factor   : scale factor from the corresponding static ValidPortfolio
        static_weights : weights from the static portfolio for this method
                         (used for the pre-OOS IS period); falls back to equal weight

    Returns:
        (equity, portfolio_df, weight_history)
        equity        : pd.Series — daily cumulative P&L (starts at 0), or None
        portfolio_df  : pd.DataFrame — combined trades (IS + OOS) with scaled P&L,
                        suitable for compute_metrics(); or None
        weight_history: list of dicts with keys:
                          "label"   : "2019–2022"
                          "oos_start", "oos_end": pd.Timestamp
                          "weights" : dict[strategy_name -> float]
    """
    names = list(combination)
    n     = len(names)

    all_times = pd.concat([strategies[nm]["Close time"] for nm in names])
    min_date  = all_times.min()
    max_date  = all_times.max()

    windows = _generate_windows(min_date, max_date, config.wf_is_years, config.wf_oos_years)
    if not windows:
        return None, None, []

    first_oos_start = windows[0][1]

    # Static weights for the pre-OOS period; fall back to equal if not provided
    pre_weights = static_weights if static_weights else equal_weight(names)

    all_frames: list[pd.DataFrame] = []
    weight_history: list[dict] = []

    # ── Pre-OOS IS period: use static weights ────────────────────────────────
    for nm in names:
        df   = strategies[nm]
        mask = df["Close time"] < first_oos_start
        chunk = df.loc[mask].copy()
        if chunk.empty:
            continue
        chunk["Profit/Loss"] = chunk["Profit/Loss"] * pre_weights[nm] * n * scale_factor
        all_frames.append(chunk)

    # ── OOS windows: use WF weights with per-window scale_factor ────────────
    for is_start, oos_start, oos_end in windows:
        weights = _is_weights(names, strategies, is_start, oos_start, method)

        label = f"{oos_start.year}–{oos_end.year}"
        weight_history.append({
            "label":     label,
            "oos_start": oos_start,
            "oos_end":   oos_end,
            "weights":   weights,
        })

        # Compute scale_factor for these WF weights from the IS daily P&L
        # so the OOS period respects the same daily loss limit as the static portfolio.
        is_daily: dict[str, pd.Series] = {}
        for nm in names:
            df   = strategies[nm]
            mask = (df["Close time"] >= is_start) & (df["Close time"] < oos_start)
            is_df = df.loc[mask]
            if not is_df.empty:
                daily_nm = (
                    is_df.groupby(is_df["Close time"].dt.date)["Profit/Loss"].sum()
                    * weights[nm] * n
                )
                is_daily[nm] = daily_nm

        if is_daily:
            combined_is_daily = pd.concat(is_daily.values()).groupby(level=0).sum()
            worst_is_day = combined_is_daily.min()
            win_scale = (config.daily_loss_limit_usd / abs(worst_is_day)
                         if worst_is_day < 0 else scale_factor)
        else:
            win_scale = scale_factor

        for nm in names:
            df    = strategies[nm]
            mask  = (df["Close time"] >= oos_start) & (df["Close time"] < oos_end)
            chunk = df.loc[mask].copy()
            if chunk.empty:
                continue
            chunk["Profit/Loss"] = chunk["Profit/Loss"] * weights[nm] * n * win_scale
            all_frames.append(chunk)

    if not all_frames:
        return None, None, weight_history

    combined = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values("Close time")
    )
    daily = combined.groupby(combined["Close time"].dt.date)["Profit/Loss"].sum()
    if daily.empty:
        return None, None, weight_history

    dr     = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    equity = daily.reindex(dr, fill_value=0.0).cumsum()
    return equity, combined, weight_history


# ── Batch computation for all combinations ────────────────────────────────────

def compute_all_wf_equities(
    combinations,   # list[CombinationResult]
    strategies: dict,
    config: PortfolioConfig,
    verbose: bool = True,
) -> None:
    """
    Compute walk-forward equity and weight history for every combination x WF
    method in-place.

    Stores results in:
      cr.wf_equity   : dict[method -> pd.Series | None]
      cr.wf_metrics  : dict[method -> dict]          (same keys as vp.metrics)
      cr.wf_weights  : dict[method -> list[dict]]    (window weight history)

    Equal weight is skipped (its WF equals its static curve).
    """
    with tqdm(
        combinations,
        desc="  Walk-forward equity",
        unit="combo",
        disable=not verbose,
    ) as bar:
        for cr in bar:
            wf_equity:  dict = {}
            wf_metrics: dict = {}
            wf_weights: dict = {}
            for method in WF_METHODS:
                vp = cr.portfolios.get(method)
                if vp is None:
                    wf_equity[method]  = None
                    wf_metrics[method] = {}
                    wf_weights[method] = []
                    continue
                equity, portfolio_df, history = compute_wf_equity(
                    cr.combination, strategies, config,
                    method, vp.scale_factor,
                    static_weights=vp.weights,
                )
                wf_equity[method]  = equity
                wf_weights[method] = history
                if portfolio_df is not None:
                    wf_metrics[method] = compute_metrics(
                        portfolio_df, initial_capital=config.account_balance
                    )
                else:
                    wf_metrics[method] = {}
            cr.wf_equity  = wf_equity
            cr.wf_metrics = wf_metrics
            cr.wf_weights = wf_weights
