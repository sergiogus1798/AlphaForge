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
    """True (PASS) if full-period Pearson correlation <= threshold."""
    a, b = _align_monthly(a_monthly, b_monthly)
    if len(a) < 6:
        return True
    corr, _ = pearsonr(a.values, b.values)
    return corr <= threshold


def spearman_filter(
    a_monthly: pd.Series,
    b_monthly: pd.Series,
    threshold: float,
) -> bool:
    """True (PASS) if full-period Spearman correlation <= threshold."""
    a, b = _align_monthly(a_monthly, b_monthly)
    if len(a) < 6:
        return True
    corr, _ = spearmanr(a.values, b.values)
    return corr <= threshold


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
    same_day: bool = True,
) -> bool:
    """
    True (PASS) if no trades open on the same asset at the same time.

    same_day=True  → conflict = same asset, same calendar day  (default)
    same_day=False → conflict = same asset, same bar (exact datetime match)
    """
    if same_day:
        keys_a = set(zip(df_a["Symbol"], df_a["Open time"].dt.date))
        keys_b = set(zip(df_b["Symbol"], df_b["Open time"].dt.date))
    else:
        keys_a = set(zip(df_a["Symbol"], df_a["Open time"]))
        keys_b = set(zip(df_b["Symbol"], df_b["Open time"]))

    return len(keys_a & keys_b) == 0


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
      - Every window: pearson <= max_rolling_corr AND spearman <= max_rolling_corr
      - Windows whose end period falls within the last `recent_years` years:
        pearson <= max_rolling_corr_recent AND spearman <= max_rolling_corr_recent

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
        if row["pearson"] > threshold or row["spearman"] > threshold:
            return False

    return True


# ── Rejection tracking ────────────────────────────────────────────────────────

@dataclass
class FilterStats:
    """Tracks how many combinations were rejected by each filter."""
    total: int = 0
    passed_static: int = 0
    passed_rolling: int = 0
    rejected_pearson: int = 0
    rejected_spearman: int = 0
    rejected_co_loss: int = 0
    rejected_same_asset: int = 0
    rejected_rolling: int = 0

    def report(self) -> None:
        rejected_static = self.total - self.passed_static
        rejected_rolling = self.passed_static - self.passed_rolling
        print(f"\n  ── Filter Report ──────────────────────────────")
        print(f"  Combinations evaluated : {self.total}")
        print(f"  Passed static filters  : {self.passed_static}  ({rejected_static} rejected)")
        print(f"    ↳ Pearson            : {self.rejected_pearson} rejected")
        print(f"    ↳ Spearman           : {self.rejected_spearman} rejected")
        print(f"    ↳ Co-loss frequency  : {self.rejected_co_loss} rejected")
        print(f"    ↳ Same-asset conflict: {self.rejected_same_asset} rejected")
        print(f"  Passed rolling filters : {self.passed_rolling}  ({rejected_rolling} rejected)")
        print(f"  ───────────────────────────────────────────────\n")


# ── Combination-level filter ──────────────────────────────────────────────────

def _apply_static_filters(
    combination: tuple[str, ...],
    strategies: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    monthly_cache: dict[str, pd.Series],
    stats: FilterStats,
) -> bool:
    """Apply the 4 static filters to every pair. Returns True if all pass."""
    names = list(combination)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_m = monthly_cache[a_name]
            b_m = monthly_cache[b_name]

            if not pearson_filter(a_m, b_m, config.max_pearson_corr):
                stats.rejected_pearson += 1
                return False

            if not spearman_filter(a_m, b_m, config.max_spearman_corr):
                stats.rejected_spearman += 1
                return False

            if not co_loss_filter(a_m, b_m, config.max_co_loss_freq):
                stats.rejected_co_loss += 1
                return False

            if not same_asset_conflict_filter(
                strategies[a_name], strategies[b_name], config.same_asset_same_day
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
) -> tuple[list[tuple[str, ...]], dict[str, pd.Series]]:
    """
    Run only the static filters and return survivors + the monthly cache.
    Useful for diagnostics before committing to the rolling filter stage.

    Returns:
        (static_passed_combinations, monthly_cache)
    """
    from tqdm import tqdm

    monthly_cache: dict[str, pd.Series] = {
        name: _monthly_pnl(df) for name, df in strategies.items()
    }

    stats = FilterStats(total=len(combinations))
    static_passed = []

    with tqdm(combinations, desc="  Static filters", unit="combo", disable=not verbose) as bar:
        for combo in bar:
            if _apply_static_filters(combo, strategies, config, monthly_cache, stats):
                static_passed.append(combo)
            bar.set_postfix(passed=len(static_passed), tried=bar.n)

    stats.passed_static = len(static_passed)
    if verbose:
        print(f"\n  Static filters: {len(static_passed)}/{len(combinations)} passed "
              f"(Pearson: {stats.rejected_pearson}, Spearman: {stats.rejected_spearman}, "
              f"Co-loss: {stats.rejected_co_loss}, Same-asset: {stats.rejected_same_asset} rejected)\n")

    return static_passed, monthly_cache


def filter_combinations(
    combinations: list[tuple[str, ...]],
    strategies: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    verbose: bool = True,
) -> tuple[list[tuple[str, ...]], FilterStats]:
    """
    Run all filters over a list of combinations.

    Static filters run first (fast). Rolling filters only run on combinations
    that survived the static stage.

    Returns:
        (passed_combinations, FilterStats)
        FilterStats contains per-filter rejection counts for diagnostics.
    """
    from tqdm import tqdm

    monthly_cache: dict[str, pd.Series] = {
        name: _monthly_pnl(df) for name, df in strategies.items()
    }

    stats = FilterStats(total=len(combinations))
    static_passed = []
    final_passed = []

    with tqdm(
        combinations,
        desc="  Static filters",
        unit="combo",
        disable=not verbose,
    ) as bar:
        for combo in bar:
            if _apply_static_filters(combo, strategies, config, monthly_cache, stats):
                static_passed.append(combo)
            bar.set_postfix(passed=len(static_passed), tried=bar.n)

    stats.passed_static = len(static_passed)

    with tqdm(
        static_passed,
        desc="  Rolling filters",
        unit="combo",
        disable=not verbose,
    ) as bar:
        for combo in bar:
            if _apply_rolling_filters(combo, config, monthly_cache, stats):
                final_passed.append(combo)
            bar.set_postfix(passed=len(final_passed), tried=bar.n)

    stats.passed_rolling = len(final_passed)

    if verbose:
        stats.report()

    return final_passed, stats


# ── Diagnostic plotting ───────────────────────────────────────────────────────

def plot_rolling_correlations(
    combinations: list[tuple[str, ...]],
    monthly_cache: dict[str, pd.Series],
    config: PortfolioConfig,
    max_combos: int = 3,
) -> None:
    """
    Plot rolling Pearson and Spearman correlations for each pair in the given
    combinations. Useful for inspecting why combinations failed the rolling filter.

    One figure per combination. Each subplot = one strategy pair.
    Threshold lines are drawn so you can see exactly which windows breached them.

    Args:
        combinations  : list of combinations to inspect (typically static survivors)
        monthly_cache : pre-computed monthly P&L per strategy
        config        : PortfolioConfig (thresholds + window size)
        max_combos    : limit how many combinations to plot (default 3)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    BG       = "#0f1117"
    AX_BG    = "#1a1a2e"
    PEARSON  = "#00d4ff"
    SPEARMAN = "#f0a500"
    THRESH   = "#e74c3c"
    THRESH_R = "#ff6b6b"
    GRID     = "#2a2a4a"
    TEXT     = "#e0e0e0"

    for combo in combinations[:max_combos]:
        names  = list(combo)
        pairs  = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        n_pairs = len(pairs)

        fig, axes = plt.subplots(
            n_pairs, 1,
            figsize=(14, 4 * n_pairs),
            facecolor=BG,
            squeeze=False,
        )
        fig.suptitle(
            f"Rolling {config.rolling_window_months}m Correlations\n"
            + " | ".join(n.split("/")[-1] for n in names),
            color=TEXT, fontsize=11, y=1.01,
        )

        # Determine recency cutoff once (use the latest period across all pairs)
        all_periods = pd.concat(
            [monthly_cache[n] for n in names], axis=0
        ).index
        latest = all_periods.max()
        recent_cutoff = latest - config.recent_years * 12

        for ax, (a_name, b_name) in zip(axes[:, 0], pairs):
            roll = _rolling_correlations(
                monthly_cache[a_name], monthly_cache[b_name], config.rolling_window_months
            )

            ax.set_facecolor(AX_BG)
            ax.tick_params(colors=TEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)
            ax.grid(True, color=GRID, linewidth=0.5)

            if roll.empty:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                        color=TEXT, ha="center")
            else:
                x = [p.to_timestamp() for p in roll.index]
                ax.plot(x, roll["pearson"],  color=PEARSON,  linewidth=1.5, label="Pearson")
                ax.plot(x, roll["spearman"], color=SPEARMAN, linewidth=1.5, label="Spearman", linestyle="--")

                # Global threshold line
                ax.axhline(config.max_rolling_corr, color=THRESH, linewidth=1,
                           linestyle=":", label=f"Threshold ({config.max_rolling_corr})")

                # Recent threshold + shaded region
                recent_ts = recent_cutoff.to_timestamp()
                ax.axhline(config.max_rolling_corr_recent, color=THRESH_R, linewidth=1,
                           linestyle="-.", label=f"Recent threshold ({config.max_rolling_corr_recent})")
                ax.axvspan(recent_ts, x[-1], alpha=0.08, color=THRESH_R, label=f"Recent ({config.recent_years}y)")

                # Highlight breaches
                for period, row in roll.iterrows():
                    thresh = config.max_rolling_corr_recent if period > recent_cutoff else config.max_rolling_corr
                    if row["pearson"] > thresh or row["spearman"] > thresh:
                        ax.axvline(period.to_timestamp(), color=THRESH, alpha=0.25, linewidth=1)

            a_short = a_name.split("/")[-1]
            b_short = b_name.split("/")[-1]
            ax.set_title(f"{a_short}  ×  {b_short}", color=TEXT, fontsize=9, pad=4)
            ax.set_ylabel("Correlation", color=TEXT, fontsize=8)
            ax.legend(fontsize=7, facecolor=AX_BG, labelcolor=TEXT, loc="upper left")
            ax.set_ylim(-1, 1)
            ax.axhline(0, color=GRID, linewidth=0.8)

        plt.tight_layout()
        plt.show()
