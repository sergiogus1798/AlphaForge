"""
regime_tests.py — Test 8: Drawdown Clustering + HMM Regime Detection.

Answers two related questions:
  A. Does your strategy's alpha survive across different market regimes
     (Bull / Bear / High-Volatility), or is it a regime-specific artefact?
  B. Do your drawdowns cluster non-randomly in time, or are they spread
     evenly (as a genuine edge should be)?

Pipeline:
  1. fit_hmm(n_states)        Fit Gaussian HMM to price returns → regime label per day.
  2. alpha_by_regime()        OLS α, β, Sharpe, win-rate per regime.
  3. drawdown_periods()       Find all equity-curve drawdown episodes.
  4. runs_test()              Wald-Wolfowitz runs test on the loss/gain sequence.
  5. loss_autocorrelation()   Ljung-Box + lag-1 ACF on the signed daily returns.
  6. regime_drawdown_overlap() Fraction of each drawdown that falls in each regime.
  7. run_all()                Run full pipeline → RegimeReport.

Usage:
    from alphaforge.IndividualAnalysis.RegimeAnalysis import RegimeAnalysis

    analysis = RegimeAnalysis(aligned, price_df,
                              strategy_name="AUDJPY/Strategy 7",
                              pair="AUDJPY")
    report   = analysis.run_all()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from hmmlearn.hmm import GaussianHMM


# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_REGIME_OBS = 30   # minimum days in a regime to run OLS
_MIN_DD_DEPTH   = 0.5  # minimum drawdown depth (%) to record as a period


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class RegimeStats:
    """Alpha/beta/Sharpe statistics for one HMM regime."""
    label:        str           # "Bull" / "Bear" / "High-Vol" / "Regime N"
    n_days:       int           # total market days in this regime
    n_active:     int           # strategy trading days in this regime
    mean_ret:     float         # mean daily price return in this regime
    vol:          float         # std dev of daily price return
    alpha:        float         # daily alpha (OLS intercept)
    alpha_tstat:  float         # t-stat on alpha
    beta:         float         # OLS slope
    r_squared:    float         # OLS R²
    info_ratio:   float         # annualised Information Ratio vs asset in regime
    win_rate:     float         # fraction of active days with positive strategy return
    verdict:      str           # "ALPHA" / "BETA" / "INCONCLUSIVE" / "INSUFFICIENT"


@dataclass
class DrawdownPeriod:
    """One peak-to-trough-to-recovery episode."""
    peak_date:     pd.Timestamp
    trough_date:   pd.Timestamp
    recovery_date: Optional[pd.Timestamp]   # None if not yet recovered
    depth_pct:     float                    # drawdown depth as % of peak equity
    duration_days: int                      # calendar days from peak to trough
    dominant_regime: Optional[str]          # regime with most days during drawdown


@dataclass
class RunsTestResult:
    """Wald-Wolfowitz runs test on the sign of daily strategy returns."""
    n_positive:   int
    n_negative:   int
    n_runs:       int
    expected_runs: float
    z_stat:       float
    p_value:      float
    verdict:      str   # "RANDOM" / "CLUSTERED" / "REGULAR"
    notes:        list[str] = field(default_factory=list)


@dataclass
class LossACFResult:
    """Ljung-Box + lag-1 ACF on daily strategy returns."""
    lag1_acf:         float
    ljung_box_stat:   float
    ljung_box_pval:   float
    verdict:          str    # "INDEPENDENT" / "CLUSTERED" / "MEAN_REVERTING"
    notes:            list[str] = field(default_factory=list)


@dataclass
class RegimeReport:
    """Full output of RegimeAnalysis.run_all()."""
    n_states:           int
    regime_labels:      pd.Series           # daily regime index (int), index = date
    regime_names:       List[str]           # e.g. ["Bull", "Bear", "High-Vol"]
    regime_stats:       List[RegimeStats]
    drawdown_periods:   List[DrawdownPeriod]
    runs_test:          RunsTestResult
    loss_acf:           LossACFResult
    # convenience accessors
    equity_curve:       pd.Series           # cumulative strategy equity ($ from initial capital)
    price_returns:      pd.Series           # raw daily price returns used for HMM


# ── Main class ────────────────────────────────────────────────────────────────

class RegimeAnalysis:
    """
    Regime detection and drawdown clustering analysis.

    Args:
        aligned:         DataFrame with columns 'strategy' and 'asset' (daily returns),
                         same format as produced by AlphaDetector.build_returns().
        price_df:        Raw OHLCV DataFrame indexed by date (used to compute
                         price returns for HMM fitting).
        strategy_name:   Human-readable label for the strategy.
        pair:            Instrument identifier (e.g. "AUDJPY").
        initial_capital: Starting equity in dollars (default 10,000).
    """

    def __init__(
        self,
        aligned:         pd.DataFrame,
        price_df:        pd.DataFrame,
        strategy_name:   str   = "Strategy",
        pair:            str   = "",
        initial_capital: float = 10_000.0,
    ):
        self.aligned          = aligned
        self.price_df         = price_df
        self.strategy_name    = strategy_name
        self.pair             = pair
        self.initial_capital  = initial_capital

        # Build price-return series aligned to the strategy date range
        self._price_returns = self._build_price_returns()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all(self, n_states: int = 3, smooth_window: int = 21) -> RegimeReport:
        """Run the full regime + drawdown analysis and return a RegimeReport.

        Args:
            n_states:      Number of HMM hidden states (default 3).
            smooth_window: Rolling-mode window (days) applied to raw HMM state
                           assignments to suppress rapid flickering.  Set to 1
                           to disable smoothing.  Default 21 (~1 month).
        """
        print(f"[RegimeAnalysis] Fitting {n_states}-state HMM (smooth={smooth_window}d)...")
        labels, names = self.fit_hmm(n_states, smooth_window=smooth_window)
        self._print_hmm_diagnostics(names)

        print("[RegimeAnalysis] Computing alpha by regime...")
        r_stats = self.alpha_by_regime(labels, names)

        equity = self._equity_curve()

        print("[RegimeAnalysis] Finding drawdown periods...")
        dds = self.drawdown_periods(equity, labels, names)

        print("[RegimeAnalysis] Running drawdown clustering tests...")
        rt  = self.runs_test()
        lac = self.loss_autocorrelation()

        print("[RegimeAnalysis] Done.")
        return RegimeReport(
            n_states         = n_states,
            regime_labels    = labels,
            regime_names     = names,
            regime_stats     = r_stats,
            drawdown_periods = dds,
            runs_test        = rt,
            loss_acf         = lac,
            equity_curve     = equity,
            price_returns    = self._price_returns,
        )

    def fit_hmm(
        self,
        n_states:      int = 3,
        smooth_window: int = 21,
    ) -> tuple[pd.Series, list[str]]:
        """
        Fit a Gaussian HMM to a tanh-transformed MA-crossover signal and assign
        daily regime labels.

        Feature:
          tanh_cross = tanh( (Close − MA30) / MA30 / _TANH_SCALE )

        Why tanh-compressed MA crossover?
          - The MA50/MA200 death-cross (MA50 < MA200) is the standard technical
            indicator for sustained downtrends.  Both sharp crashes (GFC) and slow
            multi-year declines (AUDJPY 2014-2020) produce a negative signal.
          - Raw percentage differences have wild outliers (GFC → −27%) that dominate
            the HMM's Bear state, pulling its mean far into crash territory.  Moderate
            declines (−1 to −2%) then look like "Ranging" rather than "Bear".
          - Applying tanh with scale ≈ 0.005 compresses both mild and extreme bears
            into the same [−1, −0.3] region, so the HMM can group all declining
            markets into one state regardless of severity.
          - The feature is inherently persistent (MAs change slowly), so
            state transitions are infrequent without needing extra smoothing.

        Best-of-30 random seeds avoids HMM local minima.

        Regimes are sorted by mean tanh_cross:
          - highest mean → "Bull"    (golden cross, price rising)
          - lowest mean  → "Bear"    (death cross, price declining)
          - middle       → "Ranging" (near-zero cross, sideways)

        Args:
            n_states:      Number of hidden states (default 3).
            smooth_window: Minimum regime duration (trading days) enforced after
                           Viterbi decoding to remove residual blips.

        Returns:
            labels: pd.Series (int) indexed by date, values in [0, n_states-1]
            names:  list of human-readable regime names, one per state index
        """
        _TANH_SCALE = 0.02

        prices   = self.price_df["Close"].sort_index()
        ma30     = prices.rolling(30).mean()
        raw      = ((prices - ma30) / ma30).rename("close_vs_ma30")
        signal   = np.tanh(raw / _TANH_SCALE)
        signal.name = "tanh_close_ma30"

        features = signal.dropna().to_frame()
        X        = features.values

        # ── Fit HMM — best of 30 seeds to avoid local minima ─────────────────
        import warnings as _w
        best_score = -np.inf
        best_model = None
        for seed in range(30):
            m = GaussianHMM(
                n_components    = n_states,
                covariance_type = "full",
                n_iter          = 300,
                random_state    = seed,
            )
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                m.fit(X)
            sc = m.score(X)
            if sc > best_score:
                best_score = sc
                best_model = m
        model      = best_model
        raw_labels = model.predict(X)

        # ── Map feature labels → daily via forward-fill ────────────────────────
        label_daily  = pd.Series(raw_labels, index=features.index, name="regime")
        label_series = (
            label_daily
            .reindex(self.aligned.index, method="ffill")
            .bfill()
            .astype(int)
        )

        # ── Assign state labels ────────────────────────────────────────────────
        # For 3 states we want:
        #   label 0 = Bull    → state with HIGHEST mean weekly return
        #   label 1 = Bear    → state with LOWEST  mean weekly return (crash)
        #   label 2 = Ranging → the middle state (near-zero, lower vol)
        #
        # Sorting purely descending would put the crash state last ("Ranging"),
        # which makes the deep bear markets appear amber and the thin sideways
        # blips appear red — inverted.  Instead we pin Bull=highest, Bear=lowest,
        # and everything else in the middle.
        means = model.means_[:, 0]   # first feature = weekly return
        sorted_asc = np.argsort(means)   # ascending: [lowest, ..., highest]

        remap: dict[int, int] = {}
        if n_states == 1:
            remap[int(sorted_asc[0])] = 0
        elif n_states == 2:
            remap[int(sorted_asc[-1])] = 0  # Bull  = highest
            remap[int(sorted_asc[0])]  = 1  # Bear  = lowest
        else:
            remap[int(sorted_asc[-1])] = 0  # Bull    = highest mean
            remap[int(sorted_asc[0])]  = 1  # Bear    = lowest  mean (crash)
            for new_idx, old_idx in enumerate(sorted_asc[1:-1], start=2):
                remap[int(old_idx)] = new_idx   # Ranging = middle

        label_series = label_series.map(remap)

        # ── Minimum-segment enforcer: remove residual blips ───────────────────
        if smooth_window > 1:
            label_series = _enforce_min_duration(label_series, smooth_window)

        # ── Build human-readable names ─────────────────────────────────────────
        if n_states == 2:
            names = ["Bull", "Bear"]
        elif n_states == 3:
            names = ["Bull", "Bear", "Ranging"]
        else:
            names = ["Bull", "Bear"] + [f"Ranging {i}" for i in range(1, n_states - 1)]

        # ── Store model for diagnostics ────────────────────────────────────────
        self._hmm_model      = model
        self._hmm_features   = features
        self._hmm_state_order = sorted_asc   # ascending order used for remap

        return label_series, names

    def alpha_by_regime(
        self,
        labels: pd.Series,
        names:  list[str],
    ) -> list[RegimeStats]:
        """
        Run OLS regression (strategy ~ asset) separately for each regime.

        Returns a list of RegimeStats, one per regime.
        """
        results = []
        n_states = len(names)

        for state_idx in range(n_states):
            mask = labels == state_idx
            sub  = self.aligned.loc[mask]

            y = sub["strategy"]
            x = sub["asset"]

            n_days   = int(mask.sum())
            n_active = int((y.abs() > 1e-10).sum())

            if n_days < _MIN_REGIME_OBS:
                results.append(RegimeStats(
                    label       = names[state_idx],
                    n_days      = n_days,
                    n_active    = n_active,
                    mean_ret    = float(x.mean()),
                    vol         = float(x.std()),
                    alpha       = np.nan,
                    alpha_tstat = np.nan,
                    beta        = np.nan,
                    r_squared   = np.nan,
                    info_ratio  = np.nan,
                    win_rate    = np.nan,
                    verdict     = "INSUFFICIENT",
                ))
                continue

            # OLS
            X_ols = sm.add_constant(x.values)
            try:
                ols   = sm.OLS(y.values, X_ols).fit()
                alpha = float(ols.params[0])
                beta  = float(ols.params[1])
                at    = float(ols.tvalues[0])
                r2    = float(ols.rsquared)
            except Exception:
                alpha = beta = at = r2 = np.nan

            # Information Ratio vs asset (annualised)
            # IR = mean(r_strategy − r_asset) / std(r_strategy − r_asset) * √252
            # Directly measures how much excess return the strategy earns above
            # the asset per unit of tracking error — regime-by-regime.
            active_mask = y.abs() > 1e-10
            excess = (y - x)[active_mask]
            if len(excess) >= 5 and excess.std() > 1e-12:
                info_ratio = float(excess.mean() / excess.std() * np.sqrt(252))
            else:
                info_ratio = np.nan

            win_rate = float((y > 0).sum() / n_active) if n_active > 0 else np.nan

            verdict = _alpha_verdict(alpha, at)

            results.append(RegimeStats(
                label       = names[state_idx],
                n_days      = n_days,
                n_active    = n_active,
                mean_ret    = float(x.mean()),
                vol         = float(x.std()),
                alpha       = alpha,
                alpha_tstat = at,
                beta        = beta,
                r_squared   = r2,
                info_ratio  = info_ratio,
                win_rate    = win_rate,
                verdict     = verdict,
            ))

        return results

    def drawdown_periods(
        self,
        equity: pd.Series,
        labels: pd.Series,
        names:  list[str],
    ) -> list[DrawdownPeriod]:
        """
        Identify all drawdown episodes from the equity curve.

        A drawdown starts when equity falls below its running peak and ends
        when it recovers back to (or above) that peak.

        Returns list of DrawdownPeriod, sorted by start date.
        """
        running_peak  = equity.cummax()
        dd_pct        = (equity - running_peak) / running_peak  # always <= 0

        periods = []
        in_dd   = False
        peak_date = trough_date = None
        peak_val  = trough_val  = None

        for date, val in equity.items():
            peak = running_peak.loc[date]

            if not in_dd:
                if val < peak * (1 - _MIN_DD_DEPTH / 100.0):
                    # Start of a drawdown
                    in_dd      = True
                    peak_date  = equity[equity == peak].last_valid_index()
                    trough_date = date
                    trough_val  = val
                    peak_val    = peak
            else:
                if val < trough_val:
                    trough_date = date
                    trough_val  = val
                if val >= peak_val:
                    # Recovery
                    recovery_date = date
                    depth = (peak_val - trough_val) / peak_val * 100
                    dur   = (trough_date - peak_date).days

                    dom = _dominant_regime(labels, names, peak_date, trough_date)
                    periods.append(DrawdownPeriod(
                        peak_date     = peak_date,
                        trough_date   = trough_date,
                        recovery_date = recovery_date,
                        depth_pct     = depth,
                        duration_days = dur,
                        dominant_regime = dom,
                    ))
                    in_dd = False

        # Handle ongoing drawdown (no recovery yet)
        if in_dd and peak_date is not None:
            depth = (peak_val - trough_val) / peak_val * 100
            dur   = (trough_date - peak_date).days
            dom   = _dominant_regime(labels, names, peak_date, trough_date)
            periods.append(DrawdownPeriod(
                peak_date     = peak_date,
                trough_date   = trough_date,
                recovery_date = None,
                depth_pct     = depth,
                duration_days = dur,
                dominant_regime = dom,
            ))

        return periods

    def runs_test(self) -> RunsTestResult:
        """
        Wald-Wolfowitz runs test on the sign of daily strategy returns.

        A 'run' is a maximal sequence of consecutive positive or negative
        returns.  Too few runs → clustering; too many → mean-reversion.

        Only active trading days (abs(return) > 1e-10) are used.
        """
        y      = self.aligned["strategy"]
        active = y[y.abs() > 1e-10]

        if len(active) < 20:
            return RunsTestResult(
                n_positive=0, n_negative=0, n_runs=0,
                expected_runs=0, z_stat=np.nan, p_value=np.nan,
                verdict="INSUFFICIENT",
                notes=["Fewer than 20 active trading days — test skipped."],
            )

        signs = np.sign(active.values)
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())
        n     = n_pos + n_neg

        # Count runs
        runs  = 1 + int(np.sum(signs[1:] != signs[:-1]))

        # Expected runs and variance under H0 (random)
        exp   = (2 * n_pos * n_neg) / n + 1
        var   = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
        if var <= 0:
            z = np.nan
            p = np.nan
        else:
            z = (runs - exp) / np.sqrt(var)
            p = float(2 * stats.norm.sf(abs(z)))   # two-tailed

        if np.isnan(z):
            verdict = "INCONCLUSIVE"
            notes   = []
        elif z < -2.0 and p < 0.05:
            verdict = "CLUSTERED"
            notes   = [
                "Significantly fewer runs than expected — losses cluster in time.",
                "This could indicate regime-dependency or streak risk.",
            ]
        elif z > 2.0 and p < 0.05:
            verdict = "REGULAR"
            notes   = [
                "Significantly more runs than expected — returns alternate sign frequently.",
                "Consistent with a mean-reverting strategy.",
            ]
        else:
            verdict = "RANDOM"
            notes   = ["No evidence of clustering — losses appear randomly distributed."]

        return RunsTestResult(
            n_positive    = n_pos,
            n_negative    = n_neg,
            n_runs        = runs,
            expected_runs = exp,
            z_stat        = float(z) if not np.isnan(z) else np.nan,
            p_value       = p if p is not None else np.nan,
            verdict       = verdict,
            notes         = notes,
        )

    def loss_autocorrelation(self) -> LossACFResult:
        """
        Ljung-Box test + lag-1 ACF on daily strategy returns (all active days).

        A significant positive lag-1 ACF means bad days tend to follow bad days
        (loss clustering).  Negative lag-1 ACF means mean-reversion.
        """
        y      = self.aligned["strategy"]
        active = y[y.abs() > 1e-10]

        if len(active) < 20:
            return LossACFResult(
                lag1_acf=np.nan, ljung_box_stat=np.nan, ljung_box_pval=np.nan,
                verdict="INSUFFICIENT",
                notes=["Fewer than 20 active trading days — test skipped."],
            )

        arr = active.values
        # Lag-1 ACF
        lag1 = float(pd.Series(arr).autocorr(lag=1))

        # Ljung-Box on lags 1-10
        lb = acorr_ljungbox(arr, lags=[10], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[-1])
        lb_pval = float(lb["lb_pvalue"].iloc[-1])

        if lb_pval < 0.05 and lag1 > 0.05:
            verdict = "CLUSTERED"
            notes   = [
                f"Lag-1 ACF = {lag1:+.3f}: positive autocorrelation — losses cluster.",
                f"Ljung-Box p = {lb_pval:.3f}: serial dependence confirmed.",
            ]
        elif lb_pval < 0.05 and lag1 < -0.05:
            verdict = "MEAN_REVERTING"
            notes   = [
                f"Lag-1 ACF = {lag1:+.3f}: negative autocorrelation — returns mean-revert.",
                f"Ljung-Box p = {lb_pval:.3f}: serial dependence confirmed.",
            ]
        else:
            verdict = "INDEPENDENT"
            notes   = [
                f"Lag-1 ACF = {lag1:+.3f}. Ljung-Box p = {lb_pval:.3f}.",
                "No significant serial dependence in strategy returns.",
            ]

        return LossACFResult(
            lag1_acf       = lag1,
            ljung_box_stat = lb_stat,
            ljung_box_pval = lb_pval,
            verdict        = verdict,
            notes          = notes,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _print_hmm_diagnostics(self, names: list[str]) -> None:
        """Print fitted state parameters and transition matrix to the terminal."""
        model  = self._hmm_model
        order  = self._hmm_state_order   # ascending mean — remap maps lowest→Bear, highest→Bull
        n      = len(names)

        # Rebuild remap to find which raw state index maps to each name
        # remap: raw_old → new_label.  Invert to get new_label → raw_old.
        if n == 1:
            raw_for_name = [int(order[0])]
        elif n == 2:
            raw_for_name = [int(order[-1]), int(order[0])]
        else:
            raw_for_name = [int(order[-1]), int(order[0])] + [int(order[i]) for i in range(1, n - 1)]

        _TANH_SCALE = 0.005
        print()
        print(f"  {'State':<10}  {'tanh_mean':>10}  {'tanh_std':>9}  {'~Close/MA30':>12}  {'exp.dur':>8}")
        print("  " + "-" * 58)
        for i, nm in enumerate(names):
            raw = raw_for_name[i]
            m   = model.means_[raw]
            cv  = np.sqrt(np.diag(model.covars_[raw]))
            self_p  = model.transmat_[raw, raw]
            exp_dur = 1.0 / (1.0 - self_p + 1e-9)
            approx_pct = float(np.arctanh(np.clip(m[0], -0.9999, 0.9999))) * _TANH_SCALE * 100
            print(f"  {nm:<10}  {m[0]:>+10.4f}  {cv[0]:>9.4f}  {approx_pct:>+11.2f}%  {exp_dur:>6.1f}d")

        print()
        print(f"  Transition matrix  (row=from, col=to)")
        header = "  " + " " * 10 + "".join(f"  ->{nm:<10}" for nm in names)
        print(header)
        for i, nm_i in enumerate(names):
            raw_i = raw_for_name[i]
            row   = [model.transmat_[raw_i, raw_for_name[j]] for j in range(n)]
            print(f"  {nm_i:<10}  " + "  ".join(f"{v:>12.4f}" for v in row))
        print()

    def _build_price_returns(self) -> pd.Series:
        """Compute daily log-returns of the Close price, aligned to strategy dates."""
        close = self.price_df["Close"].sort_index()
        ret   = np.log(close / close.shift(1)).dropna()
        ret.name = "price_return"
        return ret

    def _equity_curve(self) -> pd.Series:
        """Build a daily equity curve from strategy returns and initial capital."""
        r = self.aligned["strategy"]
        equity = self.initial_capital * (1 + r).cumprod()
        equity.name = "equity"
        return equity


# ── Module-level helpers ──────────────────────────────────────────────────────

def _enforce_min_duration(labels: pd.Series, min_days: int) -> pd.Series:
    """
    Iteratively absorb any regime segment shorter than min_days into its
    longer neighbour (left or right).  Absorbing into the longer neighbour
    prevents the cascade where small segments keep merging into one giant block.

    Runs repeatedly until no segment shorter than min_days remains.
    """
    arr = labels.values.copy().astype(int)
    n   = len(arr)

    changed = True
    while changed:
        changed = False
        # Build list of (start, end, state) segments
        segs = []
        i = 0
        while i < n:
            j = i + 1
            while j < n and arr[j] == arr[i]:
                j += 1
            segs.append((i, j, int(arr[i])))
            i = j

        for k, (start, end, state) in enumerate(segs):
            run_len = end - start
            if run_len < min_days:
                left_len  = (segs[k - 1][1] - segs[k - 1][0]) if k > 0            else 0
                right_len = (segs[k + 1][1] - segs[k + 1][0]) if k < len(segs)-1 else 0

                if left_len == 0 and right_len == 0:
                    continue   # only segment — nothing to absorb into
                elif left_len >= right_len:
                    new_state = segs[k - 1][2]
                else:
                    new_state = segs[k + 1][2]

                arr[start:end] = new_state
                changed = True
                break   # restart scan after any change

    return pd.Series(arr, index=labels.index, name=labels.name)


def _alpha_verdict(alpha: float, tstat: float) -> str:
    if np.isnan(alpha) or np.isnan(tstat):
        return "INCONCLUSIVE"
    if abs(tstat) >= 2.0:
        return "ALPHA" if alpha > 0 else "NEG_ALPHA"
    return "INCONCLUSIVE"


def _dominant_regime(
    labels:     pd.Series,
    names:      list[str],
    start_date: pd.Timestamp,
    end_date:   pd.Timestamp,
) -> Optional[str]:
    """Return the name of the regime that covers the most days in [start, end]."""
    sub = labels.loc[start_date:end_date]
    if sub.empty:
        return None
    counts = sub.value_counts()
    dom_idx = int(counts.idxmax())
    return names[dom_idx] if dom_idx < len(names) else f"Regime {dom_idx}"
