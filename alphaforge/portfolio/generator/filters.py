"""
filters.py — Pairwise correlation and conflict filters for Step 1.

Two stages applied to every pair of strategies in a combination:

  STATIC (fast, run first):
    1. Pearson correlation    (monthly P&L, linear)
    2. Spearman correlation   (monthly P&L, rank/monotonic)
    3. Co-loss frequency      (fraction of months both lose simultaneously)
    4. Same-asset conflict    (no trades on same asset on same day / same bar)

  ROLLING (slower, run only on combinations that passed static):
    5. Rolling Pearson        (all 12-month windows must be below threshold)
    6. Rolling Spearman       (all 12-month windows must be below threshold)
       └─ stricter threshold applied to windows falling within the recent period

A combination is rejected if ANY pair fails ANY filter.
Rejection counts per filter are tracked and reported at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from alphaforge.portfolio.config import PortfolioConfig


# ── Monthly P&L helpers ───────────────────────────────────────────────────────

def _monthly_pnl(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate trade P&L into monthly buckets using Close time.
    Returns a Series indexed by Period (YYYY-MM), missing months filled with 0.
    """
    s = df.set_index("Close time")["Profit/Loss"].copy()
    s.index = s.index.to_period("M")
    return s.groupby(s.index).sum()


def _align_monthly(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Align two monthly P&L series to their shared date range.
    Missing months within the range are filled with 0.
    """
    all_periods = a.index.union(b.index)
    a = a.reindex(all_periods, fill_value=0.0)
    b = b.reindex(all_periods, fill_value=0.0)
    return a, b


def _rolling_correlations(
    a: pd.Series,
    b: pd.Series,
    window_months: int,
) -> pd.DataFrame:
    """
    Compute rolling Pearson and Spearman correlations on aligned monthly P&L.

    Slides a window of `window_months` months one month at a time.
    Returns a DataFrame with columns [pearson, spearman] indexed by the
    window's end period. Windows with fewer than 6 non-zero observations
    in either series are skipped (NaN).
    """
    a, b = _align_monthly(a, b)
    n = len(a)
    records = []

    for end in range(window_months - 1, n):
        start = end - window_months + 1
        wa = a.iloc[start : end + 1].values
        wb = b.iloc[start : end + 1].values

        # Skip windows where both series are effectively flat (all zeros)
        if (wa == 0).all() or (wb == 0).all():
            continue

        p_corr, _ = pearsonr(wa, wb)
        s_corr, _ = spearmanr(wa, wb)
        records.append({
            "end_period": a.index[end],
            "pearson": p_corr,
            "spearman": s_corr,
        })

    if not records:
        return pd.DataFrame(columns=["end_period", "pearson", "spearman"])

    return pd.DataFrame(records).set_index("end_period")


# ── Static filter functions ───────────────────────────────────────────────────

def pearson_filter(
    a_monthly: pd.Series,
    b_monthly: pd.Series,
    threshold: float,
) -> bool:
    """True (PASS) if full-period |Pearson correlation| <= threshold."""
    a, b = _align_monthly(a_monthly, b_monthly)
    if len(a) < 6:
        return True
    corr, _ = pearsonr(a.values, b.values)
    return abs(corr) <= threshold


def spearman_filter(
    a_monthly: pd.Series,
    b_monthly: pd.Series,
    threshold: float,
) -> bool:
    """True (PASS) if full-period |Spearman correlation| <= threshold."""
    a, b = _align_monthly(a_monthly, b_monthly)
    if len(a) < 6:
        return True
    corr, _ = spearmanr(a.values, b.values)
    return abs(corr) <= threshold


def tail_correlation_filter(
    a_monthly: pd.Series,
    b_monthly: pd.Series,
    tail_percentile: float,
    threshold: float,
) -> bool:
    """
    True (PASS) if correlation during tail months stays within threshold.

    Tail months are defined as months where EITHER strategy falls below its
    own `tail_percentile` quantile of P&L — i.e. one of them is having a
    bad month. This directly measures how correlated the losses are during
    stress periods, which matters most for prop firm drawdown limits.

    Returns True (pass) if fewer than 4 tail months exist (insufficient data).
    """
    a, b = _align_monthly(a_monthly, b_monthly)
    if len(a) < 6:
        return True

    a_thresh = a.quantile(tail_percentile)
    b_thresh = b.quantile(tail_percentile)
    tail_mask = (a <= a_thresh) | (b <= b_thresh)
    tail_a = a[tail_mask]
    tail_b = b[tail_mask]

    if len(tail_a) < 4:
        return True

    corr, _ = pearsonr(tail_a.values, tail_b.values)
    return abs(corr) <= threshold


def co_loss_filter(
    a_monthly: pd.Series,
    b_monthly: pd.Series,
    threshold: float,
) -> bool:
    """
    True (PASS) if co-loss frequency <= threshold.
    Co-loss = fraction of months where both strategies have negative P&L.
    Purely observed — no theoretical assumptions.
    """
    a, b = _align_monthly(a_monthly, b_monthly)
    if len(a) < 6:
        return True
    co_loss_freq = ((a < 0) & (b < 0)).sum() / len(a)
    return co_loss_freq <= threshold


def same_asset_conflict_filter(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    window_hours: float = 8.0,
) -> bool:
    """
    True (PASS) if no two trades from different strategies open on the same
    asset within `window_hours` of each other.

    For each symbol traded by both strategies, checks every pair of open times
    (one from each strategy) using a sorted binary-search scan.

    window_hours = 0.0 → filter disabled (always passes).
    """
    import numpy as np

    if window_hours <= 0:
        return True

    window_ns = np.timedelta64(int(window_hours * 3_600 * 1_000_000_000), "ns")

    symbols_a = set(df_a["Symbol"].unique())
    symbols_b = set(df_b["Symbol"].unique())

    for symbol in symbols_a & symbols_b:
        ta_arr = df_a.loc[df_a["Symbol"] == symbol, "Open time"].values
        tb_arr = df_b.loc[df_b["Symbol"] == symbol, "Open time"].values

        if ta_arr.size == 0 or tb_arr.size == 0:
            continue

        ta_arr = np.sort(ta_arr)
        tb_arr = np.sort(tb_arr)

        for ta in ta_arr:
            lo = np.searchsorted(tb_arr, ta - window_ns)
            if lo < len(tb_arr) and tb_arr[lo] <= ta + window_ns:
                return False  # conflict within window

    return True


# ── Rolling filter functions ──────────────────────────────────────────────────

def rolling_correlation_filter(
    a_monthly: pd.Series,
    b_monthly: pd.Series,
    config: PortfolioConfig,
) -> bool:
    """
    True (PASS) if rolling Pearson AND Spearman stay within thresholds across
    all windows.

    Rules:
      - Every window: |pearson| <= max_rolling_corr AND |spearman| <= max_rolling_corr
      - Windows whose end period falls within the last `recent_years` years:
        |pearson| <= max_rolling_corr_recent AND |spearman| <= max_rolling_corr_recent

    Windows with insufficient data are skipped.
    Returns True if no windows exist (not enough history).
    """
    roll = _rolling_correlations(a_monthly, b_monthly, config.rolling_window_months)

    if roll.empty:
        return True

    # Determine the recency cutoff as a Period
    latest = roll.index.max()
    recent_cutoff = latest - config.recent_years * 12  # approximate in months

    for period, row in roll.iterrows():
        threshold = (
            config.max_rolling_corr_recent
            if period > recent_cutoff
            else config.max_rolling_corr
        )
        if abs(row["pearson"]) > threshold or abs(row["spearman"]) > threshold:
            return False

    return True


# ── Rejection tracking ────────────────────────────────────────────────────────

@dataclass
class FilterStats:
    """Tracks how many combinations were rejected by each filter stage."""
    total: int = 0
    passed_static: int = 0
    passed_dd: int = 0
    passed_rolling: int = 0
    rejected_pearson: int = 0
    rejected_spearman: int = 0
    rejected_co_loss: int = 0
    rejected_tail_corr: int = 0
    rejected_same_asset: int = 0
    rejected_dd: int = 0
    rejected_rolling: int = 0

    def report(self) -> None:
        rej_static  = self.total            - self.passed_static
        rej_dd      = self.passed_static    - self.passed_dd
        rej_rolling = self.passed_dd        - self.passed_rolling
        print(f"\n  ── Filter Report ──────────────────────────────")
        print(f"  Combinations evaluated : {self.total}")
        print(f"  Passed static filters  : {self.passed_static}  ({rej_static} rejected)")
        print(f"    ↳ Pearson            : {self.rejected_pearson} rejected")
        print(f"    ↳ Spearman           : {self.rejected_spearman} rejected")
        print(f"    ↳ Co-loss frequency  : {self.rejected_co_loss} rejected")
        print(f"    ↳ Tail correlation   : {self.rejected_tail_corr} rejected")
        print(f"    ↳ Same-asset conflict: {self.rejected_same_asset} rejected")
        print(f"  Passed DD check        : {self.passed_dd}  ({rej_dd} rejected)")
        print(f"  Passed rolling filters : {self.passed_rolling}  ({rej_rolling} rejected)")
        print(f"  ───────────────────────────────────────────────\n")


# ── Combination-level filter ──────────────────────────────────────────────────

def _apply_static_filters(
    combination: tuple[str, ...],
    strategies: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    monthly_cache: dict[str, pd.Series],
    stats: FilterStats,
    universe=None,   # optional Universe for O(1) matrix lookups
) -> bool:
    """
    Apply the 4 static filters to every pair. Returns True if all pass.
    If a Universe is supplied, all checks are pure index lookups (fast).
    Otherwise falls back to computing from raw data (slow but correct).
    """
    names = list(combination)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]

            if universe is not None:
                # O(1) matrix lookup — pre-computed in Universe
                if abs(universe.pearson(a_name, b_name)) > config.max_pearson_corr:
                    stats.rejected_pearson += 1
                    return False
                if abs(universe.spearman(a_name, b_name)) > config.max_spearman_corr:
                    stats.rejected_spearman += 1
                    return False
                if config.enable_co_loss_filter and universe.co_loss(a_name, b_name) > config.max_co_loss_freq:
                    stats.rejected_co_loss += 1
                    return False
                if config.enable_same_asset_filter and universe.has_overlap(a_name, b_name):
                    stats.rejected_same_asset += 1
                    return False
                if config.enable_tail_corr_filter:
                    a_m = monthly_cache[a_name]
                    b_m = monthly_cache[b_name]
                    if not tail_correlation_filter(a_m, b_m, config.tail_percentile, config.max_tail_corr):
                        stats.rejected_tail_corr += 1
                        return False
            else:
                # Fallback: compute from raw monthly series
                a_m = monthly_cache[a_name]
                b_m = monthly_cache[b_name]

                if not pearson_filter(a_m, b_m, config.max_pearson_corr):
                    stats.rejected_pearson += 1
                    return False
                if not spearman_filter(a_m, b_m, config.max_spearman_corr):
                    stats.rejected_spearman += 1
                    return False
                if config.enable_co_loss_filter and not co_loss_filter(a_m, b_m, config.max_co_loss_freq):
                    stats.rejected_co_loss += 1
                    return False
                if config.enable_tail_corr_filter and not tail_correlation_filter(a_m, b_m, config.tail_percentile, config.max_tail_corr):
                    stats.rejected_tail_corr += 1
                    return False
                if config.enable_same_asset_filter and not same_asset_conflict_filter(
                    strategies[a_name], strategies[b_name], config.same_asset_window_hours
                ):
                    stats.rejected_same_asset += 1
                    return False

    return True


def _apply_rolling_filters(
    combination: tuple[str, ...],
    config: PortfolioConfig,
    monthly_cache: dict[str, pd.Series],
    stats: FilterStats,
) -> bool:
    """Apply rolling correlation filter to every pair. Returns True if all pass."""
    names = list(combination)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not rolling_correlation_filter(
                monthly_cache[names[i]],
                monthly_cache[names[j]],
                config,
            ):
                stats.rejected_rolling += 1
                return False

    return True


def filter_static_only(
    combinations: list[tuple[str, ...]],
    strategies: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    verbose: bool = True,
    universe=None,
) -> tuple[list[tuple[str, ...]], dict[str, pd.Series], FilterStats]:
    """
    Run only the static filters and return survivors, the monthly cache, and stats.

    If a Universe is supplied, uses pre-computed N×N matrices (fast).
    Otherwise computes correlations on the fly (slower, used as fallback).

    Returns:
        (static_passed_combinations, monthly_cache, FilterStats)
    """
    from tqdm import tqdm

    monthly_cache: dict[str, pd.Series] = {
        name: _monthly_pnl(df) for name, df in strategies.items()
    }

    stats = FilterStats(total=len(combinations))
    static_passed = []

    with tqdm(combinations, desc="  Static filters", unit="combo", disable=not verbose) as bar:
        for combo in bar:
            if _apply_static_filters(combo, strategies, config, monthly_cache, stats, universe=universe):
                static_passed.append(combo)
            bar.set_postfix(passed=len(static_passed), tried=bar.n)

    stats.passed_static = len(static_passed)
    if verbose:
        print(f"\n  Static filters: {len(static_passed)}/{len(combinations)} passed "
              f"(Pearson: {stats.rejected_pearson}, Spearman: {stats.rejected_spearman}, "
              f"Co-loss: {stats.rejected_co_loss}, Tail: {stats.rejected_tail_corr}, "
              f"Same-asset: {stats.rejected_same_asset} rejected)\n")

    return static_passed, monthly_cache, stats


def filter_rolling_only(
    combinations: list[tuple[str, ...]],
    monthly_cache: dict[str, pd.Series],
    config: PortfolioConfig,
    verbose: bool = True,
    stats: FilterStats | None = None,
) -> tuple[list[tuple[str, ...]], FilterStats]:
    """
    Run only the rolling filters on combinations that have already passed
    static filters and the DD check.

    Args:
        combinations  : combinations to test
        monthly_cache : pre-computed monthly P&L (from filter_static_only)
        config        : PortfolioConfig
        verbose       : print progress
        stats         : existing FilterStats to update (creates new one if None)

    Returns:
        (passed_combinations, FilterStats)
    """
    from tqdm import tqdm

    if stats is None:
        stats = FilterStats(
            total=len(combinations),
            passed_static=len(combinations),
            passed_dd=len(combinations),
        )

    final_passed: list[tuple[str, ...]] = []

    with tqdm(combinations, desc="  Rolling filters", unit="combo", disable=not verbose) as bar:
        for combo in bar:
            if _apply_rolling_filters(combo, config, monthly_cache, stats):
                final_passed.append(combo)
            bar.set_postfix(passed=len(final_passed), tried=bar.n)

    stats.passed_rolling = len(final_passed)

    if verbose:
        rej = len(combinations) - len(final_passed)
        print(f"\n  Rolling filters: {len(final_passed)}/{len(combinations)} passed "
              f"({rej} rejected).\n")

    return final_passed, stats


def filter_combinations(
    combinations: list[tuple[str, ...]],
    strategies: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    verbose: bool = True,
    universe=None,
) -> tuple[list[tuple[str, ...]], FilterStats]:
    """
    Run static + rolling filters over a list of combinations.
    (Convenience wrapper used by test scripts; pipeline uses the split functions.)

    Returns:
        (passed_combinations, FilterStats)
    """
    static_passed, monthly_cache, stats = filter_static_only(
        combinations, strategies, config, verbose=verbose, universe=universe,
    )
    # Mark DD fields as N/A (not checked here)
    stats.passed_dd = len(static_passed)

    final_passed, stats = filter_rolling_only(
        static_passed, monthly_cache, config, verbose=verbose, stats=stats,
    )

    if verbose:
        stats.report()

    return final_passed, stats


# ── Diagnostic plotting ───────────────────────────────────────────────────────

def plot_rolling_correlations(
    combinations: list[tuple[str, ...]],
    monthly_cache: dict[str, pd.Series],
    config: PortfolioConfig,
    max_combos: int = 3,
    max_pairs_detail: int = 12,
) -> None:
    """
    Plot rolling Pearson and Spearman correlations for each pair in the given
    combinations. Useful for inspecting why combinations failed the rolling filter.

    For portfolios with <= max_pairs_detail pairs: one subplot per pair (detail view).
    For portfolios with > max_pairs_detail pairs: heatmap summary of max |correlation|
    per pair across all rolling windows (compact view), plus detail for top 12 pairs.

    Args:
        combinations      : list of combinations to inspect (typically static survivors)
        monthly_cache     : pre-computed monthly P&L per strategy
        config            : PortfolioConfig (thresholds + window size)
        max_combos        : limit how many combinations to plot (default 3)
        max_pairs_detail  : if n_pairs > this, switch to heatmap + top-N detail (default 12)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    BG       = "#0f1117"
    AX_BG    = "#1a1a2e"
    PEARSON  = "#00d4ff"
    SPEARMAN = "#f0a500"
    THRESH   = "#e74c3c"
    THRESH_R = "#ff6b6b"
    GRID     = "#2a2a4a"
    TEXT     = "#e0e0e0"

    def _style_ax(ax):
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5)

    def _plot_pair(ax, a_name, b_name, recent_cutoff):
        roll = _rolling_correlations(
            monthly_cache[a_name], monthly_cache[b_name], config.rolling_window_months
        )
        _style_ax(ax)
        if roll.empty:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    color=TEXT, ha="center")
        else:
            x = [p.to_timestamp() for p in roll.index]
            ax.plot(x, roll["pearson"],  color=PEARSON,  linewidth=1.5, label="Pearson")
            ax.plot(x, roll["spearman"], color=SPEARMAN, linewidth=1.5, label="Spearman", linestyle="--")
            ax.axhline(config.max_rolling_corr, color=THRESH, linewidth=1,
                       linestyle=":", label=f"±{config.max_rolling_corr}")
            ax.axhline(-config.max_rolling_corr, color=THRESH, linewidth=1, linestyle=":")
            recent_ts = recent_cutoff.to_timestamp()
            ax.axhline(config.max_rolling_corr_recent, color=THRESH_R, linewidth=1,
                       linestyle="-.", label=f"±{config.max_rolling_corr_recent} (recent)")
            ax.axhline(-config.max_rolling_corr_recent, color=THRESH_R, linewidth=1, linestyle="-.")
            ax.axvspan(recent_ts, x[-1], alpha=0.08, color=THRESH_R)
            for period, row in roll.iterrows():
                thresh = config.max_rolling_corr_recent if period > recent_cutoff else config.max_rolling_corr
                if abs(row["pearson"]) > thresh or abs(row["spearman"]) > thresh:
                    ax.axvline(period.to_timestamp(), color=THRESH, alpha=0.25, linewidth=1)
        a_short = a_name.split("/")[-1]
        b_short = b_name.split("/")[-1]
        ax.set_title(f"{a_short}  ×  {b_short}", color=TEXT, fontsize=9, pad=4)
        ax.set_ylabel("Correlation", color=TEXT, fontsize=8)
        ax.legend(fontsize=7, facecolor=AX_BG, labelcolor=TEXT, loc="upper left")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color=GRID, linewidth=0.8)

    for combo in combinations[:max_combos]:
        names   = list(combo)
        pairs   = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        n_pairs = len(pairs)

        all_periods = pd.concat([monthly_cache[n] for n in names], axis=0).index
        latest = all_periods.max()
        recent_cutoff = latest - config.recent_years * 12

        title_base = (f"Rolling {config.rolling_window_months}m Correlations  |  "
                      + " · ".join(n.split("/")[-1] for n in names))

        if n_pairs <= max_pairs_detail:
            # ── Detail view: one subplot per pair ────────────────────────────
            n_cols = min(2, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(14, 4 * n_rows),
                                     facecolor=BG, squeeze=False)
            fig.suptitle(title_base, color=TEXT, fontsize=11)

            for idx, (a_name, b_name) in enumerate(pairs):
                ax = axes[idx // n_cols][idx % n_cols]
                _plot_pair(ax, a_name, b_name, recent_cutoff)

            # Hide empty axes
            for idx in range(n_pairs, n_rows * n_cols):
                axes[idx // n_cols][idx % n_cols].set_visible(False)

            plt.tight_layout()
            plt.show()

        else:
            # ── Compact view: heatmap of max |corr| + detail for top pairs ──

            # Build max |corr| matrix
            short_names = [n.split("/")[-1] for n in names]
            n = len(names)
            mat = np.zeros((n, n))
            pair_max: list[tuple[float, str, str]] = []

            for i, a_name in enumerate(names):
                for j, b_name in enumerate(names):
                    if i == j:
                        mat[i, j] = 1.0
                    elif j > i:
                        roll = _rolling_correlations(
                            monthly_cache[a_name], monthly_cache[b_name],
                            config.rolling_window_months
                        )
                        if roll.empty:
                            v = 0.0
                        else:
                            v = float(roll[["pearson", "spearman"]].abs().max().max())
                        mat[i, j] = mat[j, i] = v
                        pair_max.append((v, a_name, b_name))

            fig_h, ax_h = plt.subplots(figsize=(max(8, n * 0.6 + 2), max(6, n * 0.5 + 2)),
                                        facecolor=BG)
            ax_h.set_facecolor(AX_BG)
            im = ax_h.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
            ax_h.set_xticks(range(n))
            ax_h.set_yticks(range(n))
            ax_h.set_xticklabels(short_names, rotation=45, ha="right", color=TEXT, fontsize=7)
            ax_h.set_yticklabels(short_names, color=TEXT, fontsize=7)
            ax_h.set_title(f"Max |Correlation| across all windows\n{title_base}",
                           color=TEXT, fontsize=10)
            for i in range(n):
                for j in range(n):
                    ax_h.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                              fontsize=6, color="black" if mat[i,j] < 0.7 else "white")
            plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

            # Detail plots for the top max_pairs_detail pairs by max |corr|
            pair_max.sort(reverse=True, key=lambda t: t[0])
            top_pairs = [(a, b) for _, a, b in pair_max[:max_pairs_detail]]
            n_top = len(top_pairs)
            n_cols = min(2, n_top)
            n_rows = (n_top + n_cols - 1) // n_cols
            fig2, axes2 = plt.subplots(n_rows, n_cols,
                                       figsize=(14, 4 * n_rows),
                                       facecolor=BG, squeeze=False)
            fig2.suptitle(f"Top {n_top} pairs by max |correlation|\n{title_base}",
                          color=TEXT, fontsize=11)
            for idx, (a_name, b_name) in enumerate(top_pairs):
                ax = axes2[idx // n_cols][idx % n_cols]
                _plot_pair(ax, a_name, b_name, recent_cutoff)
            for idx in range(n_top, n_rows * n_cols):
                axes2[idx // n_cols][idx % n_cols].set_visible(False)
            plt.tight_layout()
            plt.show()
