"""
monte_carlo.py — Monte Carlo robustness tests for trading strategies.

Five independent tests, each producing a DataFrame of simulation results:

  Test 1  run_bootstrap             Sample trades with replacement (bootstrapping)
  Test 2  run_reshuffle             Shuffle trade order, keep same trades
  Test 3  run_time_block_bootstrap  Bootstrap chronological 6-month blocks (sample with replacement)
  Test 4  run_trade_block_bootstrap Sample consecutive trade blocks with replacement
  Test 5  run_best_trade_removal    Remove top X% trades by PnL, test remainder

All tests support random trade skipping (default 5% probability).
Each simulation records: final_equity, max_drawdown, return_pct, num_trades.

Usage:
    from alphaforge.IndividualAnalysis.MonteCarlo import run_all_tests
    results = run_all_tests(df, n_simulations=10_000)
    # results["bootstrap"], results["reshuffle"], ...
"""

import math

import numpy as np
import pandas as pd

from alphaforge.metrics import compute_metrics


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _apply_skip(pnl: np.ndarray, skip_prob: float, rng) -> np.ndarray:
    """Randomly drop trades with probability skip_prob."""
    if skip_prob <= 0:
        return pnl
    return pnl[rng.random(len(pnl)) >= skip_prob]


def _equity_stats(pnl: np.ndarray, initial_capital: float = 10_000) -> dict:
    """Compute key stats from a PnL sequence."""
    if len(pnl) == 0:
        return {"final_equity": initial_capital, "max_drawdown": 0.0,
                "return_pct": 0.0, "ret_dd": float("nan"), "num_trades": 0}
    equity     = initial_capital + np.cumsum(pnl)
    peak       = np.maximum.accumulate(equity)
    dd         = float((peak - equity).max())
    final      = float(equity[-1])
    ret        = (final - initial_capital) / initial_capital * 100
    ret_dd     = ret / (dd / initial_capital * 100) if dd > 0 else float("nan")
    return {
        "final_equity": round(final, 2),
        "max_drawdown": round(dd, 2),
        "return_pct":   round(ret, 4),
        "ret_dd":       round(ret_dd, 4),
        "num_trades":   len(pnl),
    }


# ── Test 1 — Trade Bootstrapping ───────────────────────────────────────────────

def run_bootstrap(
    df: pd.DataFrame,
    n_simulations: int = 10_000,
    skip_trade_probability: float = 0.05,
    initial_capital: float = 10_000,
    seed: int = None,
    progress_fn=None,
) -> pd.DataFrame:
    """
    Test 1: Sample trades with replacement until reaching original trade count.

    Tests whether strategy performance depends on specific trades.
    The same trade may appear multiple times; others may be absent entirely.
    """
    rng  = np.random.default_rng(seed)
    pnl  = df["Profit/Loss"].dropna().values
    n    = len(pnl)
    tick = max(1, n_simulations // 200)
    results = []
    for i in range(n_simulations):
        sampled = rng.choice(pnl, size=n, replace=True)
        sampled = _apply_skip(sampled, skip_trade_probability, rng)
        results.append(_equity_stats(sampled, initial_capital))
        if progress_fn and (i + 1) % tick == 0:
            progress_fn(i + 1, n_simulations)
    return pd.DataFrame(results)


# ── Test 2 — Trade Reshuffling ─────────────────────────────────────────────────

def run_reshuffle(
    df: pd.DataFrame,
    n_simulations: int = 10_000,
    skip_trade_probability: float = 0.05,
    initial_capital: float = 10_000,
    seed: int = None,
    progress_fn=None,
) -> pd.DataFrame:
    """
    Test 2: Randomly shuffle trade order while keeping the exact same trades.

    Tests path dependency — especially how sensitive drawdowns are to the
    sequence in which trades occurred.
    """
    rng  = np.random.default_rng(seed)
    pnl  = df["Profit/Loss"].dropna().values
    tick = max(1, n_simulations // 200)
    results = []
    for i in range(n_simulations):
        shuffled = rng.permutation(pnl)
        shuffled = _apply_skip(shuffled, skip_trade_probability, rng)
        results.append(_equity_stats(shuffled, initial_capital))
        if progress_fn and (i + 1) % tick == 0:
            progress_fn(i + 1, n_simulations)
    return pd.DataFrame(results)


# ── Test 3 — Time Block Bootstrapping ─────────────────────────────────────────

def run_time_block_bootstrap(
    df: pd.DataFrame,
    n_simulations: int = 10_000,
    time_block_size: str = "6ME",
    skip_trade_probability: float = 0.05,
    initial_capital: float = 10_000,
    seed: int = None,
    progress_fn=None,
) -> pd.DataFrame:
    """
    Test 3: Split trades into fixed-length time blocks, then sample blocks
    with replacement (bootstrap).

    Each 6-month window becomes one block (variable trade count). Each
    simulation draws len(blocks) blocks with replacement, so some regimes
    appear multiple times while others are absent — unlike mere shuffling,
    the final equity and trade count genuinely vary across simulations.

    Args:
        time_block_size: pandas period alias for block size (default '6ME' = 6 months).
    """
    rng = np.random.default_rng(seed)
    s   = df.sort_values("Close time").set_index("Close time")
    blocks = [g["Profit/Loss"].dropna().values
              for _, g in s.groupby(pd.Grouper(freq=time_block_size))
              if len(g) > 0]
    if not blocks:
        return pd.DataFrame()
    n_blocks = len(blocks)
    tick     = max(1, n_simulations // 200)
    results  = []
    for i in range(n_simulations):
        indices = rng.integers(0, n_blocks, size=n_blocks)   # sample WITH replacement
        pnl     = np.concatenate([blocks[j] for j in indices])
        pnl     = _apply_skip(pnl, skip_trade_probability, rng)
        results.append(_equity_stats(pnl, initial_capital))
        if progress_fn and (i + 1) % tick == 0:
            progress_fn(i + 1, n_simulations)
    return pd.DataFrame(results)


# ── Test 4 — Trade Block Bootstrapping ────────────────────────────────────────

def run_trade_block_bootstrap(
    df: pd.DataFrame,
    n_simulations: int = 10_000,
    block_size: int = 10,
    skip_trade_probability: float = 0.05,
    initial_capital: float = 10_000,
    seed: int = None,
    progress_fn=None,
) -> pd.DataFrame:
    """
    Test 4: Sample consecutive blocks of trades with replacement.

    Preserves local trade dependencies: volatility clustering, correlated
    signals, winning/losing streaks. block_size controls the dependency horizon.
    """
    rng = np.random.default_rng(seed)
    pnl = df.sort_values("Close time")["Profit/Loss"].dropna().values
    n   = len(pnl)

    blocks          = [pnl[i:i + block_size] for i in range(0, n, block_size)]
    n_blocks_needed = math.ceil(n / block_size)
    tick            = max(1, n_simulations // 200)

    results = []
    for i in range(n_simulations):
        indices = rng.integers(0, len(blocks), size=n_blocks_needed)
        sampled = np.concatenate([blocks[i] for i in indices])[:n]
        sampled = _apply_skip(sampled, skip_trade_probability, rng)
        results.append(_equity_stats(sampled, initial_capital))
        if progress_fn and (i + 1) % tick == 0:
            progress_fn(i + 1, n_simulations)
    return pd.DataFrame(results)


# ── Test 5 — Best Trade Removal ────────────────────────────────────────────────

def run_best_trade_removal(
    df: pd.DataFrame,
    n_simulations: int = 10_000,
    removal_pct: float = 0.05,
    skip_trade_probability: float = 0.05,
    initial_capital: float = 10_000,
    seed: int = None,
    progress_fn=None,
) -> pd.DataFrame:
    """
    Test 5: Remove the top removal_pct of trades by PnL, then test the remainder.

    Checks whether strategy profitability relies on a handful of extreme winning
    trades. Runs reshuffling on the reduced trade set across all simulations.
    """
    rng     = np.random.default_rng(seed)
    pnl     = df["Profit/Loss"].dropna().values
    cutoff  = int(len(pnl) * (1 - removal_pct))
    reduced = np.sort(pnl)[:cutoff]
    tick    = max(1, n_simulations // 200)

    results = []
    for i in range(n_simulations):
        shuffled = rng.permutation(reduced)
        shuffled = _apply_skip(shuffled, skip_trade_probability, rng)
        results.append(_equity_stats(shuffled, initial_capital))
        if progress_fn and (i + 1) % tick == 0:
            progress_fn(i + 1, n_simulations)
    return pd.DataFrame(results)


# ── Equity curve sequences ─────────────────────────────────────────────────────

def get_equity_curves(
    df: pd.DataFrame,
    test: str = "reshuffle",
    n_curves: int = 100,
    skip_trade_probability: float = 0.0,
    initial_capital: float = 10_000,
    seed: int = 42,
) -> list:
    """
    Run n_curves simulations and return cumulative PnL sequences for plotting.

    Returns a list of lists, each being a cumulative PnL sequence (starts at 0,
    not initial_capital). skip_trade_probability defaults to 0 so the curves
    reflect the full trade set without random omissions.
    """
    rng     = np.random.default_rng(seed)
    sorted_ = df.sort_values("Close time").reset_index(drop=True)
    pnl     = sorted_["Profit/Loss"].dropna().values
    n       = len(pnl)

    # Pre-compute test-specific structures
    if test == "time_block":
        s      = sorted_.set_index("Close time")
        blocks = [g["Profit/Loss"].dropna().values
                  for _, g in s.groupby(pd.Grouper(freq="6ME")) if len(g) > 0]
        if not blocks:
            blocks = [pnl]
        pre = blocks
    elif test == "trade_block":
        pre = [pnl[i:i + 10] for i in range(0, n, 10)]
    elif test == "best_trade_removal":
        pre = np.sort(pnl)[:int(n * 0.95)]
    else:
        pre = None

    curves = []
    for _ in range(n_curves):
        if test == "reshuffle":
            seq = rng.permutation(pnl)
        elif test == "bootstrap":
            seq = rng.choice(pnl, size=n, replace=True)
        elif test == "time_block":
            idx = rng.integers(0, len(pre), size=len(pre))   # sample WITH replacement
            seq = np.concatenate([pre[j] for j in idx])
        elif test == "trade_block":
            nb  = math.ceil(n / 10)
            idx = rng.integers(0, len(pre), size=nb)
            seq = np.concatenate([pre[i] for i in idx])[:n]
        elif test == "best_trade_removal":
            seq = rng.permutation(pre)
        else:
            seq = pnl.copy()
        seq = _apply_skip(seq, skip_trade_probability, rng)
        curves.append(np.cumsum(seq).tolist())
    return curves


# ── Run all tests ──────────────────────────────────────────────────────────────

def run_all_tests(
    df: pd.DataFrame,
    n_simulations: int = 10_000,
    skip_trade_probability: float = 0.05,
    time_block_size: str = "6ME",
    trade_block_size: int = 10,
    best_trade_removal_pct: float = 0.05,
    initial_capital: float = 10_000,
    seed: int = None,
) -> dict:
    """
    Run all 5 Monte Carlo robustness tests and return results as a dict.

    Each value is a DataFrame with columns:
        final_equity, max_drawdown, return_pct, ret_dd, num_trades

    Args:
        df:                      Trades DataFrame.
        n_simulations:           Simulations per test (default 10 000).
        skip_trade_probability:  Probability of skipping each trade (default 0.05).
        time_block_size:         Pandas period alias for time block test (default '6ME').
        trade_block_size:        Trades per block for block bootstrap (default 10).
        best_trade_removal_pct:  Fraction of top trades to remove (default 0.05).
        initial_capital:         Starting equity in USD (default 10 000).
        seed:                    Random seed for reproducibility.
    """
    return {
        "bootstrap":          run_bootstrap(
            df, n_simulations, skip_trade_probability, initial_capital, seed),
        "reshuffle":          run_reshuffle(
            df, n_simulations, skip_trade_probability, initial_capital, seed),
        "time_block":         run_time_block_bootstrap(
            df, n_simulations, time_block_size, skip_trade_probability, initial_capital, seed),
        "trade_block":        run_trade_block_bootstrap(
            df, n_simulations, trade_block_size, skip_trade_probability, initial_capital, seed),
        "best_trade_removal": run_best_trade_removal(
            df, n_simulations, best_trade_removal_pct, skip_trade_probability, initial_capital, seed),
    }


# ── Summary helper ─────────────────────────────────────────────────────────────

def simulation_summary(sim_results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise a simulation result DataFrame with full distribution stats
    and confidence interval bounds.

    Returns a DataFrame indexed by metric with columns:
        mean, std, min,
        ci90_lo (p5),  ci90_hi (p95),
        ci95_lo (p2.5), ci95_hi (p97.5),
        ci99_lo (p0.5), ci99_hi (p99.5),
        p25, p50, p75, max
    """
    pcts    = [0.005, 0.01, 0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975, 0.99, 0.995]
    summary = sim_results.describe(percentiles=pcts).T
    summary = summary.rename(columns={
        "0.5%":  "ci99_lo",
        "1%":    "ci98_lo",
        "2.5%":  "ci95_lo",
        "5%":    "ci90_lo",
        "25%":   "p25",
        "50%":   "p50",
        "75%":   "p75",
        "95%":   "ci90_hi",
        "97.5%": "ci95_hi",
        "99%":   "ci98_hi",
        "99.5%": "ci99_hi",
    })
    # Reorder columns for readability
    cols = ["mean", "std", "min",
            "ci90_lo", "ci90_hi",
            "ci95_lo", "ci95_hi",
            "ci98_lo", "ci98_hi",
            "ci99_lo", "ci99_hi",
            "p25", "p50", "p75", "max"]
    existing = [c for c in cols if c in summary.columns]
    return summary[existing]


# ── Rolling Monte Carlo ────────────────────────────────────────────────────────

_TEST_FN = {
    "bootstrap":          run_bootstrap,
    "reshuffle":          run_reshuffle,
    "time_block":         run_time_block_bootstrap,
    "trade_block":        run_trade_block_bootstrap,
    "best_trade_removal": run_best_trade_removal,
}


def run_rolling(
    df: pd.DataFrame,
    n_periods: int = 4,
    test: str = "reshuffle",
    n_simulations: int = 10_000,
    skip_trade_probability: float = 0.05,
    initial_capital: float = 10_000,
    seed: int = None,
    progress_fn=None,
    # test-specific overrides (forwarded to the underlying function)
    time_block_size: str = "6ME",
    block_size: int = 10,
    removal_pct: float = 0.05,
) -> dict:
    """
    Run Monte Carlo on each chronological period independently.

    Splits trades into n_periods equal chunks by trade count, then runs the
    chosen MC test on each period separately. Use this to check whether MC
    results are consistent across time — if one period dominates, the overall
    stats are misleading.

    Args:
        df:                      Trades DataFrame.
        n_periods:               Number of chronological periods to split into.
        test:                    Which MC test to run per period. One of:
                                 'bootstrap', 'reshuffle', 'time_block',
                                 'trade_block', 'best_trade_removal'.
        n_simulations:           Simulations per period (default 10 000).
        skip_trade_probability:  Trade skip probability (default 0.05).
        initial_capital:         Starting equity per period in USD (default 10 000).
                                 Each period starts fresh from this baseline so
                                 returns are comparable regardless of period order.
        seed:                    Random seed for reproducibility.
        progress_fn:             Optional callable(done, total) for progress tracking.

    Returns:
        dict keyed by period label (e.g. "P1 2020-01 to 2021-06 | 245 trades"),
        each value a simulation DataFrame with columns:
            final_equity, max_drawdown, return_pct, ret_dd, num_trades
    """
    if test not in _TEST_FN:
        raise ValueError(f"Unknown test '{test}'. Choose from: {list(_TEST_FN)}")

    fn      = _TEST_FN[test]
    sorted_ = df.sort_values("Close time").reset_index(drop=True)

    # Split by equal calendar-time intervals, not equal trade count
    t_min  = sorted_["Close time"].min()
    t_max  = sorted_["Close time"].max()
    edges  = pd.date_range(t_min, t_max, periods=n_periods + 1)
    chunks = []
    for i in range(n_periods):
        lo, hi = edges[i], edges[i + 1]
        if i < n_periods - 1:
            mask = (sorted_["Close time"] >= lo) & (sorted_["Close time"] < hi)
        else:
            mask = (sorted_["Close time"] >= lo) & (sorted_["Close time"] <= hi)
        chunks.append(sorted_[mask].reset_index(drop=True))

    total   = n_simulations * n_periods

    results   = {}
    done_base = 0
    for i, chunk in enumerate(chunks):
        chunk = chunk.dropna(subset=["Profit/Loss"])
        if len(chunk) < 5:
            done_base += n_simulations
            continue
        t0    = chunk["Close time"].min().strftime("%Y-%m")
        t1    = chunk["Close time"].max().strftime("%Y-%m")
        label = f"P{i+1} {t0} to {t1} | {len(chunk)} trades"

        def _make_pf(base, tot, parent):
            if parent is None:
                return None
            def _pf(done, _):
                parent(base + done, tot)
            return _pf

        extra = {}
        if test == "time_block":
            extra["time_block_size"] = time_block_size
        elif test == "trade_block":
            extra["block_size"] = block_size
        elif test == "best_trade_removal":
            extra["removal_pct"] = removal_pct

        results[label] = fn(
            chunk,
            n_simulations,
            skip_trade_probability=skip_trade_probability,
            initial_capital=initial_capital,
            seed=seed,
            progress_fn=_make_pf(done_base, total, progress_fn),
            **extra,
        )
        done_base += n_simulations
    return results


def rolling_summary(
    rolling_results: dict,
    metrics: list = None,
) -> pd.DataFrame:
    """
    Build a comparison table of MC distribution stats across all periods.

    Returns a MultiIndex DataFrame: rows = (period, metric),
    columns = mean, std, ci90_lo, ci90_hi, ci95_lo, ci95_hi, ci99_lo, ci99_hi, p50.

    Args:
        rolling_results: Output of run_rolling().
        metrics:         Which metrics to include. Defaults to
                         ['return_pct', 'max_drawdown', 'ret_dd', 'final_equity'].
    """
    if metrics is None:
        metrics = ["return_pct", "max_drawdown", "ret_dd", "final_equity"]

    rows = []
    for period_label, sim_df in rolling_results.items():
        summary = simulation_summary(sim_df)
        for metric in metrics:
            if metric not in summary.index:
                continue
            row = {"period": period_label, "metric": metric}
            row.update(summary.loc[metric].to_dict())
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index(["period", "metric"])
    keep = [c for c in ["mean", "std", "ci90_lo", "ci90_hi",
                         "ci95_lo", "ci95_hi", "ci99_lo", "ci99_hi", "p50"]
            if c in out.columns]
    return out[keep]


# ── Portfolio stats extraction ────────────────────────────────────────────────

def extract_portfolio_stats(
    sim_results,
    ci_levels=(90, 95, 98, 99),
    metrics=("return_pct", "max_drawdown", "ret_dd", "final_equity"),
) -> dict:
    """
    Extract CI bounds from MC results for portfolio construction.

    Accepts either:
    - a DataFrame  (output of run_* / run_all_tests[key])  → single-period stats
    - a dict       (output of run_rolling)                  → per-period + aggregate

    Returns a dict shaped like::

        {
            "overall": {
                "return_pct":   {"mean": …, "median": …, "ci90_lo": …, "ci90_hi": …, …},
                "max_drawdown": {…},
                "ret_dd":       {…},
                "final_equity": {…},
            },
            "by_period": {          # only present when rolling dict is passed
                "P1 2020-01 …": {"return_pct": {…}, …},
                …
            }
        }

    The CI percentile keys follow the pattern ci<level>_lo / ci<level>_hi.
    Supported levels: 90, 95, 98, 99 (require simulation_summary with 98% support).

    Why both overall and by_period?
    - "overall" → single number for portfolio optimisation (Sharpe proxy, etc.)
    - "by_period" → check consistency; a strategy that looks good overall but has
      one catastrophic period should be down-weighted.
    """

    def _extract_one(df: pd.DataFrame) -> dict:
        if df is None or len(df) == 0:
            return {}
        summ = simulation_summary(df)
        out  = {}
        for metric in metrics:
            if metric not in summ.index:
                continue
            s   = summ.loc[metric]
            row = {
                "mean":   float(s.get("mean", np.nan)),
                "median": float(s.get("p50",  np.nan)),
            }
            for ci in ci_levels:
                lo_key = f"ci{ci}_lo"
                hi_key = f"ci{ci}_hi"
                row[lo_key] = float(s.get(lo_key, np.nan))
                row[hi_key] = float(s.get(hi_key, np.nan))
            out[metric] = row
        return out

    if isinstance(sim_results, pd.DataFrame):
        return {"overall": _extract_one(sim_results)}

    # Rolling dict
    all_dfs = list(sim_results.values())
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        overall  = _extract_one(combined)
    else:
        overall  = {}

    by_period = {label: _extract_one(df) for label, df in sim_results.items()}
    return {"overall": overall, "by_period": by_period}


# ── Legacy API ─────────────────────────────────────────────────────────────────

def original_metrics(df: pd.DataFrame) -> dict:
    """Compute baseline metrics from the real trade sequence."""
    return compute_metrics(df)


def run_simulation(df: pd.DataFrame, n_simulations: int = 1000,
                   seed: int = None) -> pd.DataFrame:
    """Legacy: reshuffling simulation (Test 2) returning equity stats."""
    return run_reshuffle(df, n_simulations=n_simulations, seed=seed)


def compare_to_simulation(df: pd.DataFrame, n_simulations: int = 1000,
                          seed: int = None) -> dict:
    """Legacy: full Monte Carlo analysis using reshuffling."""
    baseline  = original_metrics(df)
    simulated = run_reshuffle(df, n_simulations=n_simulations, seed=seed)
    return {
        "original":  baseline,
        "simulated": simulated,
        "summary":   simulation_summary(simulated),
    }
