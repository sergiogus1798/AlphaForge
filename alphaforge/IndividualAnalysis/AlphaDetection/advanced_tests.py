"""
advanced_tests.py — Tests 1-4 and 6 for alpha validity beyond the core OLS regression.

Tests implemented:
  1. Autocorrelation   — Are strategy residuals serially correlated?
                         (autocorrelated residuals → alpha may be spurious)
  2. Distribution      — Do residuals look like a Gaussian or hide tail risk?
                         (extreme skew / kurtosis → hidden short-vol profile)
  3. Nonlinear Beta    — Is there a quadratic market dependency?
                         (significant β₂ → payoff is concave = short gamma)
  4. Asymmetric Beta   — Does the strategy behave differently in up vs down markets?
                         (high β_down/β_up ratio → crash sensitivity)
  6. Placebo/Permutation — Is the alpha statistically real vs random luck?
                         (individual shuffle + block shuffle → empirical p-value)

Rolling analysis:
  rolling_metrics(window=252)  — compute how each metric evolved over time.

Usage:
    from alphaforge.IndividualAnalysis.AlphaDetection.advanced_tests import AdvancedTests

    tester  = AdvancedTests(aligned_df, core_result)
    results = tester.run_all()

    rolling_df = tester.rolling_metrics(window=252)   # 1-year rolling window
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


# ── Minimum observations ──────────────────────────────────────────────────────

_MIN_OBS = 60


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AutocorrelationResult:
    """
    Residual autocorrelation analysis.

    Checks whether today's alpha stream residual can be predicted from
    yesterday's — if yes, the signal is not fully exploited or worse,
    the 'alpha' is a regime artifact.
    """
    acf_values:       np.ndarray          # ACF at lags 1..max_lags
    lags:             np.ndarray          # [1, 2, ..., max_lags]
    significance_band: float              # ±2/√n  (95% CI for white noise)
    ljung_box_stat:   float               # Ljung-Box Q-statistic (up to lag 10)
    ljung_box_pval:   float               # p-value — small = autocorrelation present
    n_significant:    int                 # how many lags exceed the significance band
    newey_west_tstat: float               # NW-HAC t-stat on alpha (corrected for AC)
    newey_west_alpha: float               # alpha estimate under NW standard errors
    verdict:          str                 # "CLEAN" / "AUTOCORRELATED" / "INCONCLUSIVE"
    notes:            list[str] = field(default_factory=list)


@dataclass
class DistributionResult:
    """
    Return distribution shape analysis.

    Measures whether the strategy's alpha stream has fat tails or skew
    that would be invisible to a linear regression.
    """
    skewness:        float     # negative = left-tail (hidden downside)
    excess_kurtosis: float     # >3 = fat tails; >6 = extreme fat tails
    shapiro_stat:    float     # Shapiro-Wilk test statistic
    shapiro_pval:    float     # p-value — small = NOT normal
    dagostino_stat:  float     # D'Agostino K² test statistic
    dagostino_pval:  float     # p-value — small = NOT normal
    pct_95:          float     # 95th percentile daily return
    pct_05:          float     # 5th percentile daily return (left tail)
    verdict:         str       # "NORMAL" / "FAT_TAILS" / "SKEWED" / "DANGEROUS"
    notes:           list[str] = field(default_factory=list)


@dataclass
class NonlinearBetaResult:
    """
    Quadratic (nonlinear) beta test.

    Adds R_asset² as a second regressor.  A significant β₂ means the
    strategy's returns depend on the *magnitude* of the market move,
    not just its direction.

    β₂ < 0 → concave payoff → short gamma / short vol (risky in crises)
    β₂ > 0 → convex payoff  → trend-following / long gamma (desirable)
    """
    beta_linear:    float    # β₁ (linear market sensitivity)
    beta_quadratic: float    # β₂ (curvature term)
    beta2_tstat:    float    # t-stat for β₂
    beta2_pval:     float    # p-value for β₂
    r2_linear:      float    # R² of the simple linear model
    r2_quadratic:   float    # R² of the quadratic model
    r2_gain:        float    # Δ R² from adding β₂
    alpha_nonlin:   float    # alpha under the quadratic model
    verdict:        str      # "CONVEX" / "CONCAVE" / "LINEAR" / "INCONCLUSIVE"
    notes:          list[str] = field(default_factory=list)


@dataclass
class AsymmetricBetaResult:
    """
    Asymmetric beta (up-market vs down-market sensitivity).

    Splits regression days into up-market (R_asset > 0) and down-market
    (R_asset < 0) and runs separate OLS on each half.

    β_down / β_up ratio > 1.5 → strategy is more exposed in falling markets
    (crash sensitivity — a classic hidden risk).
    """
    beta_up:         float    # beta when market was rising
    beta_down:       float    # beta when market was falling
    alpha_up:        float    # alpha in up-market days
    alpha_down:      float    # alpha in down-market days
    tstat_up:        float    # t-stat on beta_up
    tstat_down:      float    # t-stat on beta_down
    n_up:            int      # number of up-market days
    n_down:          int      # number of down-market days
    asymmetry_ratio: float    # abs(β_down) / max(abs(β_up), 1e-6)
    verdict:         str      # "SYMMETRIC" / "CRASH_SENSITIVE" / "INCONCLUSIVE"
    notes:           list[str] = field(default_factory=list)


@dataclass
class PlaceboResult:
    """
    Permutation / placebo test (Test 6).

    Answers: 'Could this alpha have appeared by chance?'

    Shuffles strategy returns 1000 times (breaking any real relationship
    with the market) and re-fits OLS each time. The resulting null
    distribution of alpha tells us what alpha looks like under the null
    hypothesis of no skill.

    Two shuffle variants:
      Individual : completely random — destroys all serial structure
      Block      : shuffles in ~20-day blocks — preserves short-range
                   autocorrelation, giving a more conservative test

    Empirical p-value = fraction of null alphas ≥ observed alpha.
    """
    observed_alpha:         float        # actual OLS alpha from the full dataset
    null_alphas_individual: np.ndarray   # null distribution (individual shuffle)
    null_alphas_block:      np.ndarray   # null distribution (block shuffle)
    pval_individual:        float        # empirical p-value (individual)
    pval_block:             float        # empirical p-value (block) — more conservative
    n_permutations:         int
    block_size:             int
    verdict:                str          # "REAL" / "BORDERLINE" / "SPURIOUS"
    notes:                  list[str] = field(default_factory=list)


@dataclass
class AdvancedReport:
    """Container for all advanced test results."""
    autocorrelation: Optional[AutocorrelationResult] = None
    distribution:    Optional[DistributionResult]    = None
    nonlinear_beta:  Optional[NonlinearBetaResult]   = None
    asymmetric_beta: Optional[AsymmetricBetaResult]  = None
    placebo:         Optional[PlaceboResult]          = None


# ── Main class ────────────────────────────────────────────────────────────────

class AdvancedTests:
    """
    Runs the advanced alpha-validity tests on a strategy's return series.

    Args:
        aligned:  DataFrame with columns ["strategy", "asset"] — the output
                  of AlphaDetector.build_returns().
        core:     RegressionResult from AlphaDetector.run_core_regression() —
                  used to extract the beta-neutral residuals.
        max_lags: Maximum lag for the ACF test (default 20).
    """

    def __init__(
        self,
        aligned,          # pd.DataFrame  — from AlphaDetector.build_returns()
        core,             # RegressionResult — from AlphaDetector.run_core_regression()
        max_lags: int = 20,
    ):
        self.aligned  = aligned.copy()
        self.core     = core
        self.max_lags = max_lags

        # Alpha stream: residuals + alpha intercept (beta-neutral daily return)
        self._stream: Optional[pd.Series] = None

    # ── Public entry point ────────────────────────────────────────────────────

    def run_all(
        self,
        include_placebo: bool = True,
        n_permutations:  int  = 1000,
        block_size:      int  = 20,
    ) -> AdvancedReport:
        """Run all tests and return an AdvancedReport."""
        report = AdvancedReport(
            autocorrelation = self.run_autocorrelation(),
            distribution    = self.run_distribution(),
            nonlinear_beta  = self.run_nonlinear_beta(),
            asymmetric_beta = self.run_asymmetric_beta(),
        )
        if include_placebo:
            report.placebo = self.run_placebo(n_permutations, block_size)
        return report

    # ── Test 1: Autocorrelation ───────────────────────────────────────────────

    def run_autocorrelation(self) -> AutocorrelationResult:
        """
        Test for serial correlation in the beta-neutral alpha stream.

        Steps:
          1. Compute ACF at lags 1..max_lags
          2. Run Ljung-Box Q-test (lags 1-10)
          3. Re-run core OLS with Newey-West HAC standard errors
             (corrects t-stat if residuals are autocorrelated)
        """
        stream = self._get_stream()

        if len(stream) < _MIN_OBS:
            return AutocorrelationResult(
                acf_values=np.array([]), lags=np.array([]),
                significance_band=float("nan"),
                ljung_box_stat=float("nan"), ljung_box_pval=float("nan"),
                n_significant=0,
                newey_west_tstat=float("nan"), newey_west_alpha=float("nan"),
                verdict="INCONCLUSIVE", notes=["Not enough observations"],
            )

        n    = len(stream)
        band = 2.0 / np.sqrt(n)
        arr  = stream.values

        # Fast ACF via numpy corrcoef
        acf_vals = np.array([
            float(np.corrcoef(arr[:-k], arr[k:])[0, 1])
            for k in range(1, self.max_lags + 1)
        ])
        lags  = np.arange(1, self.max_lags + 1)
        n_sig = int(np.sum(np.abs(acf_vals) > band))

        # Ljung-Box test on first 10 lags
        lb_result = acorr_ljungbox(arr, lags=[10], return_df=True)
        lb_stat   = float(lb_result["lb_stat"].iloc[-1])
        lb_pval   = float(lb_result["lb_pvalue"].iloc[-1])

        # Newey-West corrected alpha
        y  = self.aligned["strategy"]
        X  = sm.add_constant(self.aligned[["asset"]])
        df = pd.concat([y, X], axis=1).dropna()
        nw = sm.OLS(df.iloc[:, 0], df.iloc[:, 1:]).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )
        nw_alpha = float(nw.params["const"])
        nw_tstat = float(nw.tvalues["const"])

        notes = []
        if lb_pval < 0.05 or n_sig >= 3:
            verdict = "AUTOCORRELATED"
            notes.append(
                f"Ljung-Box p={lb_pval:.3f} — residuals are serially correlated"
            )
            notes.append(
                f"NW-corrected t(α)={nw_tstat:+.2f} vs naive t(α)="
                f"{self.core.alpha_tstat:+.2f}"
            )
        elif n_sig == 0 and lb_pval > 0.20:
            verdict = "CLEAN"
            notes.append("No significant autocorrelation detected")
        else:
            verdict = "INCONCLUSIVE"
            notes.append(f"{n_sig} lags exceed the 95% band — mild autocorrelation")

        return AutocorrelationResult(
            acf_values=acf_vals, lags=lags,
            significance_band=band,
            ljung_box_stat=lb_stat, ljung_box_pval=lb_pval,
            n_significant=n_sig,
            newey_west_tstat=round(nw_tstat, 3),
            newey_west_alpha=round(nw_alpha, 6),
            verdict=verdict, notes=notes,
        )

    # ── Test 2: Distribution ──────────────────────────────────────────────────

    def run_distribution(self) -> DistributionResult:
        """
        Analyse the shape of the alpha stream return distribution.

        A normal-looking distribution is desirable (no hidden tail risk).
        Negative skew + high kurtosis = short-volatility profile in disguise.
        """
        stream = self._get_stream()

        if len(stream) < _MIN_OBS:
            return DistributionResult(
                skewness=float("nan"), excess_kurtosis=float("nan"),
                shapiro_stat=float("nan"), shapiro_pval=float("nan"),
                dagostino_stat=float("nan"), dagostino_pval=float("nan"),
                pct_95=float("nan"), pct_05=float("nan"),
                verdict="INCONCLUSIVE", notes=["Not enough observations"],
            )

        # Filter to active trading days only (strategy return != 0).
        # Zero-return days (strategy flat) inflate kurtosis and distort
        # skewness when a strategy only trades a few times per week.
        active_mask    = self.aligned["strategy"].reindex(stream.index).abs() > 1e-10
        arr_active     = stream[active_mask].values
        arr_all        = stream.values
        n_active       = int(active_mask.sum())

        arr = arr_active if n_active >= _MIN_OBS else arr_all

        skew = float(stats.skew(arr))
        kurt = float(stats.kurtosis(arr))          # excess kurtosis (normal=0)
        p05  = float(np.percentile(arr, 5))
        p95  = float(np.percentile(arr, 95))

        sw_stat,  sw_pval  = stats.shapiro(arr[:5000])
        dag_stat, dag_pval = stats.normaltest(arr)

        notes     = []
        dangerous = (skew < -1.0) and (kurt > 3.0)
        fat_tails = kurt > 3.0
        skewed    = abs(skew) > 1.0

        if dangerous:
            verdict = "DANGEROUS"
            notes.append(
                f"Negative skew ({skew:.2f}) + fat tails (kurt={kurt:.2f}) "
                "→ hidden short-vol profile"
            )
        elif fat_tails:
            verdict = "FAT_TAILS"
            notes.append(f"Excess kurtosis={kurt:.2f} → returns have fat tails")
        elif skewed:
            verdict = "SKEWED"
            col = "left (downside)" if skew < 0 else "right (upside)"
            notes.append(f"Skew={skew:.2f} → distribution is {col}-skewed")
        else:
            verdict = "NORMAL"
            notes.append("Distribution is approximately Gaussian — no hidden tail risk")

        notes.append(
            f"Active-day filter: {n_active:,} of {len(arr_all):,} days had a "
            "non-zero strategy return — shape metrics computed on active days only"
        )
        if sw_pval < 0.01:
            notes.append(f"Shapiro-Wilk p={sw_pval:.3f} → strongly rejects normality")

        return DistributionResult(
            skewness=round(skew, 4), excess_kurtosis=round(kurt, 4),
            shapiro_stat=round(float(sw_stat), 4),
            shapiro_pval=round(float(sw_pval), 4),
            dagostino_stat=round(float(dag_stat), 4),
            dagostino_pval=round(float(dag_pval), 4),
            pct_95=round(p95, 6), pct_05=round(p05, 6),
            verdict=verdict, notes=notes,
        )

    # ── Test 3: Nonlinear Beta ────────────────────────────────────────────────

    def run_nonlinear_beta(self) -> NonlinearBetaResult:
        """
        Quadratic OLS:  R_strategy = α + β₁·R_asset + β₂·R_asset² + ε

        A significant negative β₂ means the strategy is implicitly short
        options (short gamma / short vol) — it makes small steady gains but
        is vulnerable to large market moves in either direction.
        """
        data = self.aligned.dropna()

        if len(data) < _MIN_OBS:
            return NonlinearBetaResult(
                beta_linear=float("nan"), beta_quadratic=float("nan"),
                beta2_tstat=float("nan"), beta2_pval=float("nan"),
                r2_linear=float("nan"), r2_quadratic=float("nan"),
                r2_gain=float("nan"), alpha_nonlin=float("nan"),
                verdict="INCONCLUSIVE", notes=["Not enough observations"],
            )

        y  = data["strategy"]
        x  = data["asset"]
        x2 = x ** 2

        X_lin  = sm.add_constant(x.to_frame())
        m_lin  = sm.OLS(y, X_lin).fit()
        r2_lin = float(m_lin.rsquared)

        X_quad  = sm.add_constant(pd.concat([x, x2.rename("asset_sq")], axis=1))
        m_quad  = sm.OLS(y, X_quad).fit()
        r2_quad = float(m_quad.rsquared)

        beta1   = float(m_quad.params["asset"])
        beta2   = float(m_quad.params["asset_sq"])
        t2      = float(m_quad.tvalues["asset_sq"])
        p2      = float(m_quad.pvalues["asset_sq"])
        alpha_q = float(m_quad.params["const"])
        r2_gain = r2_quad - r2_lin

        notes = []
        if abs(t2) < 1.5 or abs(beta2) < 1e-6:
            verdict = "LINEAR"
            notes.append(
                f"β₂={beta2:.4f}  t={t2:+.2f}  p={p2:.3f} — "
                "no significant nonlinear market exposure"
            )
        elif beta2 < 0:
            verdict = "CONCAVE"
            notes.append(
                f"β₂={beta2:.4f}  t={t2:+.2f}  p={p2:.3f} — "
                "concave payoff: profits from low-vol, vulnerable to large moves"
            )
        else:
            verdict = "CONVEX"
            notes.append(
                f"β₂={beta2:.4f}  t={t2:+.2f}  p={p2:.3f} — "
                "convex payoff: trend-following or long-gamma characteristic"
            )

        if r2_gain > 0.02:
            notes.append(
                f"Adding β₂ improves R² by {r2_gain*100:.2f}pp — "
                "nonlinearity explains a meaningful fraction of returns"
            )

        return NonlinearBetaResult(
            beta_linear=round(beta1, 4), beta_quadratic=round(beta2, 6),
            beta2_tstat=round(t2, 3), beta2_pval=round(p2, 4),
            r2_linear=round(r2_lin, 4), r2_quadratic=round(r2_quad, 4),
            r2_gain=round(r2_gain, 4), alpha_nonlin=round(alpha_q, 6),
            verdict=verdict, notes=notes,
        )

    # ── Test 4: Asymmetric Beta ───────────────────────────────────────────────

    def run_asymmetric_beta(self) -> AsymmetricBetaResult:
        """
        Split regression on up-market vs down-market days.

        Runs two separate OLS regressions:
          - Only days where R_asset > 0  (market was rising)
          - Only days where R_asset <= 0 (market was falling)

        If β_down >> β_up, the strategy is crash-sensitive.
        """
        data = self.aligned.dropna()

        if len(data) < _MIN_OBS:
            return AsymmetricBetaResult(
                beta_up=float("nan"), beta_down=float("nan"),
                alpha_up=float("nan"), alpha_down=float("nan"),
                tstat_up=float("nan"), tstat_down=float("nan"),
                n_up=0, n_down=0,
                asymmetry_ratio=float("nan"),
                verdict="INCONCLUSIVE", notes=["Not enough observations"],
            )

        mask_up   = data["asset"] > 0
        up_data   = data[mask_up]
        down_data = data[~mask_up]

        def _ols_split(sub):
            if len(sub) < 30:
                return (float("nan"),) * 4
            y_ = sub["strategy"]
            X_ = sm.add_constant(sub[["asset"]])
            m  = sm.OLS(y_, X_).fit()
            return (
                float(m.params["const"]),
                float(m.params["asset"]),
                float(m.tvalues["const"]),
                float(m.tvalues["asset"]),
            )

        a_up,   b_up,   ta_up,   tb_up   = _ols_split(up_data)
        a_down, b_down, ta_down, tb_down = _ols_split(down_data)

        n_up   = int(mask_up.sum())
        n_down = int((~mask_up).sum())

        ratio = (
            abs(b_down) / max(abs(b_up), 1e-6)
            if not (np.isnan(b_up) or np.isnan(b_down))
            else float("nan")
        )

        notes = []
        if np.isnan(ratio):
            verdict = "INCONCLUSIVE"
            notes.append("Insufficient data in one market direction")
        elif ratio > 1.5:
            verdict = "CRASH_SENSITIVE"
            notes.append(
                f"β_down={b_down:+.3f}  β_up={b_up:+.3f}  ratio={ratio:.2f} — "
                "strategy is significantly more exposed in falling markets"
            )
        elif ratio > 1.2:
            verdict = "INCONCLUSIVE"
            notes.append(
                f"Mild asymmetry: β_down/β_up={ratio:.2f} — borderline crash sensitivity"
            )
        else:
            verdict = "SYMMETRIC"
            notes.append(
                f"β_down={b_down:+.3f}  β_up={b_up:+.3f}  ratio={ratio:.2f} — "
                "symmetric market exposure in up and down periods"
            )

        def _s(v): return round(v, 4) if not np.isnan(v) else float("nan")
        def _s6(v): return round(v, 6) if not np.isnan(v) else float("nan")

        return AsymmetricBetaResult(
            beta_up=_s(b_up), beta_down=_s(b_down),
            alpha_up=_s6(a_up), alpha_down=_s6(a_down),
            tstat_up=_s(tb_up), tstat_down=_s(tb_down),
            n_up=n_up, n_down=n_down,
            asymmetry_ratio=round(float(ratio), 3) if not np.isnan(ratio) else float("nan"),
            verdict=verdict, notes=notes,
        )

    # ── Test 6: Placebo / Permutation ─────────────────────────────────────────

    def run_placebo(
        self,
        n_permutations: int = 1000,
        block_size:     int = 20,
    ) -> PlaceboResult:
        """
        Permutation test: is the observed alpha statistically real?

        Shuffles the strategy return series N times and re-estimates OLS
        alpha in each case.  Two shuffle methods:

          Individual (coin-flip): completely random permutation of daily
            returns — destroys all serial structure.

          Block shuffle: shuffles in contiguous ~20-day blocks — preserves
            the short-range autocorrelation structure that the strategy
            naturally has, giving a more conservative (harder-to-beat) null.

        Empirical p-value = fraction of null alphas ≥ observed alpha.
        Uses numpy for speed (1000 regressions in ~1-2 seconds).
        """
        data = self.aligned.dropna()

        if len(data) < _MIN_OBS:
            return PlaceboResult(
                observed_alpha=float("nan"),
                null_alphas_individual=np.array([]),
                null_alphas_block=np.array([]),
                pval_individual=float("nan"),
                pval_block=float("nan"),
                n_permutations=n_permutations,
                block_size=block_size,
                verdict="INCONCLUSIVE",
                notes=["Not enough observations"],
            )

        y = data["strategy"].values
        x = data["asset"].values
        X = np.column_stack([np.ones(len(x)), x])

        # Observed alpha (numpy fast OLS)
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        observed_alpha = float(b[0])

        # Residual permutation: shuffle ε, not y.
        # Shuffling raw y detaches returns from their market context — a good
        # strategy day (+3% on a +2% market) can get reshuffled onto a −2%
        # market day, inflating the null alpha unfairly against the strategy.
        # Instead we permute only the residuals (the timing-skill component):
        #   y_null = β̂·x + ε_shuffled   (null: α = 0, β unchanged)
        # This keeps each market day paired with a plausible residual while
        # testing only whether the TIMING of skill was random.
        residuals = y - X @ b   # ε̂ = y − α̂ − β̂·x
        x_col     = x           # asset returns (no intercept column needed for reconstruction)

        rng = np.random.default_rng(42)
        null_individual = np.empty(n_permutations)
        null_block      = np.empty(n_permutations)

        for i in range(n_permutations):
            # Individual shuffle of residuals
            eps_ind = rng.permutation(residuals)
            y_ind   = b[1] * x_col + eps_ind   # β̂·x + ε_perm  (α_null = 0)
            b_ind, _, _, _ = np.linalg.lstsq(X, y_ind, rcond=None)
            null_individual[i] = b_ind[0]

            # Block shuffle of residuals
            eps_blk = _block_shuffle(residuals, block_size, rng)
            y_blk   = b[1] * x_col + eps_blk
            b_blk, _, _, _ = np.linalg.lstsq(X, y_blk, rcond=None)
            null_block[i] = b_blk[0]

        pval_ind = float((null_individual >= observed_alpha).mean())
        pval_blk = float((null_block      >= observed_alpha).mean())

        notes = []
        # Verdict based on the more conservative block p-value
        if pval_blk < 0.05:
            verdict = "REAL"
            notes.append(
                f"Block p={pval_blk:.3f} — alpha survives conservative permutation test"
            )
        elif pval_blk < 0.15:
            verdict = "BORDERLINE"
            notes.append(
                f"Block p={pval_blk:.3f} — alpha is marginally significant; "
                "use more data or reduce parameters"
            )
        else:
            verdict = "SPURIOUS"
            notes.append(
                f"Block p={pval_blk:.3f} — alpha could easily have appeared "
                "by chance; likely data-mined or regime-specific"
            )

        if abs(pval_ind - pval_blk) > 0.10:
            notes.append(
                f"Large gap between individual p={pval_ind:.3f} and block p={pval_blk:.3f} "
                "— strategy returns have autocorrelation that matters for inference"
            )

        return PlaceboResult(
            observed_alpha=round(observed_alpha, 6),
            null_alphas_individual=null_individual,
            null_alphas_block=null_block,
            pval_individual=round(pval_ind, 4),
            pval_block=round(pval_blk, 4),
            n_permutations=n_permutations,
            block_size=block_size,
            verdict=verdict,
            notes=notes,
        )

    # ── Rolling metrics ───────────────────────────────────────────────────────

    def rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """
        Compute rolling versions of the four test key metrics.

        At each date t, fits the analysis on the preceding `window` trading days.
        Uses numpy for speed — this runs in a few seconds on 20 years of data.

        Args:
            window: number of trading days per rolling window (default 252 = 1 year).

        Returns:
            pd.DataFrame indexed by date with columns:
                alpha           — OLS intercept on the trailing window (rolling alpha, Test 6)
                lag1_acf        — lag-1 autocorrelation of the alpha stream (Test 1)
                skewness        — skewness of the alpha stream (Test 2)
                excess_kurtosis — excess kurtosis of the alpha stream (Test 2)
                beta2           — quadratic beta coefficient, negative = concave (Test 3)
                r2_gain         — R² improvement from adding β₂ (Test 3)
                beta_up         — market beta on up-market days (Test 4)
                beta_down       — market beta on down-market days (Test 4)
                asymmetry_ratio — |β_down| / |β_up| (Test 4)
        """
        data   = self.aligned.dropna()
        stream = self._get_stream()

        # Align stream to data index (should already be aligned, but be safe)
        stream = stream.reindex(data.index)

        n = len(data)
        if n < window + 10:
            return pd.DataFrame(columns=[
                "alpha",
                "lag1_acf", "skewness", "excess_kurtosis",
                "beta2", "r2_gain",
                "beta_up", "beta_down", "asymmetry_ratio",
            ])

        rows = []
        for i in range(window, n + 1):
            chunk_data   = data.iloc[i - window : i]
            chunk_stream = stream.iloc[i - window : i].dropna()

            x    = chunk_data["asset"].values
            y    = chunk_data["strategy"].values
            date = data.index[i - 1]

            # ── Lag-1 ACF and distribution of alpha stream ─────────────────
            s = chunk_stream.values
            if len(s) > 5:
                lag1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
                # Shape metrics: active trading days only (strategy != 0)
                strat_active = chunk_data["strategy"].reindex(chunk_stream.index).abs() > 1e-10
                n_active = int(strat_active.sum())
                if n_active == 0:
                    # No trades in this window — shape stats are meaningless
                    sk = krt = np.nan
                elif n_active >= 5:
                    s_act = s[strat_active.values]
                    sk  = float(stats.skew(s_act))
                    krt = float(stats.kurtosis(s_act))
                else:
                    # Too few active days — use full stream as fallback
                    sk  = float(stats.skew(s))
                    krt = float(stats.kurtosis(s))
            else:
                lag1 = sk = krt = np.nan

            # ── Linear alpha + quadratic β₂ and ΔR² (Tests 3 & 6) ────────────
            alpha_val = np.nan
            if len(x) >= 20:
                X_lin  = np.column_stack([np.ones(len(x)), x])
                X_quad = np.column_stack([np.ones(len(x)), x, x ** 2])

                b_lin,  _, _, _ = np.linalg.lstsq(X_lin,  y, rcond=None)
                b_quad, _, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)

                alpha_val = float(b_lin[0])   # rolling OLS intercept

                y_hat_lin  = X_lin  @ b_lin
                y_hat_quad = X_quad @ b_quad
                ss_tot     = np.sum((y - y.mean()) ** 2)

                if ss_tot > 0:
                    r2_lin  = 1 - np.sum((y - y_hat_lin)  ** 2) / ss_tot
                    r2_quad = 1 - np.sum((y - y_hat_quad) ** 2) / ss_tot
                    r2_gain = float(r2_quad - r2_lin)
                else:
                    r2_gain = np.nan

                b2 = float(b_quad[2])
            else:
                b2 = r2_gain = np.nan

            # ── Asymmetric betas (Test 4) ──────────────────────────────────
            mask_up   = x > 0
            mask_down = ~mask_up
            if mask_up.sum() >= 10 and mask_down.sum() >= 10:
                X_up = np.column_stack([np.ones(mask_up.sum()),   x[mask_up]])
                X_dn = np.column_stack([np.ones(mask_down.sum()), x[mask_down]])
                b_up_r, _, _, _ = np.linalg.lstsq(X_up, y[mask_up],   rcond=None)
                b_dn_r, _, _, _ = np.linalg.lstsq(X_dn, y[mask_down], rcond=None)
                beta_up_val   = float(b_up_r[1])
                beta_down_val = float(b_dn_r[1])
                ratio         = float(abs(beta_down_val) / max(abs(beta_up_val), 1e-6))
            else:
                beta_up_val = beta_down_val = ratio = np.nan

            rows.append({
                "date":            date,
                "alpha":           alpha_val,
                "lag1_acf":        lag1,
                "skewness":        sk,
                "excess_kurtosis": krt,
                "beta2":           b2,
                "r2_gain":         r2_gain,
                "beta_up":         beta_up_val,
                "beta_down":       beta_down_val,
                "asymmetry_ratio": ratio,
            })

        return pd.DataFrame(rows).set_index("date")

    # ── Rolling placebo periods ───────────────────────────────────────────────

    def rolling_placebo_periods(
        self,
        window:         int = 252,
        n_permutations: int = 5000,
    ) -> list[dict]:
        """
        Run a permutation test on each consecutive non-overlapping window.

        For a window of W days, the full time series is sliced into
        floor(N / W) non-overlapping chunks.  For each chunk an individual
        shuffle permutation test is run and the null alpha distribution
        collected.

        Returns a list of dicts, one per window:
            start           — first date of the window
            end             — last date of the window
            observed_alpha  — OLS intercept on that window
            null_alphas     — array of N permuted alphas (individual shuffle)
            pval            — empirical p-value (fraction of null >= observed)
        """
        data = self.aligned.dropna()
        n    = len(data)
        if n < window:
            return []

        # Count total active periods first for the progress bar
        n_active = sum(
            1 for s in range(0, n - window + 1, window)
            if np.abs(data["strategy"].values[s : s + window]).sum() >= 1e-12
        )

        periods   = []
        rng       = np.random.default_rng(42)
        start     = 0
        done      = 0

        print(f"[Rolling Placebo]  {n_active} periods  x  {n_permutations:,} permutations each")

        while start + window <= n:
            chunk = data.iloc[start : start + window]
            y     = chunk["strategy"].values
            x     = chunk["asset"].values
            X     = np.column_stack([np.ones(len(x)), x])

            if np.abs(y).sum() < 1e-12:
                start += window
                continue

            b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            obs_alpha  = float(b[0])

            # Residual permutation (same logic as static placebo)
            residuals = y - X @ b

            null = np.empty(n_permutations)
            for i in range(n_permutations):
                eps_p        = rng.permutation(residuals)
                y_p          = b[1] * x + eps_p
                b_p, _, _, _ = np.linalg.lstsq(X, y_p, rcond=None)
                null[i]      = b_p[0]

            pval = float((null >= obs_alpha).mean())
            periods.append({
                "start":          chunk.index[0],
                "end":            chunk.index[-1],
                "observed_alpha": obs_alpha,
                "null_alphas":    null,
                "pval":           pval,
            })

            done += 1
            bar      = "#" * done + "-" * (n_active - done)
            pval_tag = "REAL" if pval < 0.05 else ("BORDER" if pval < 0.15 else "SPURIOUS")
            print(
                f"  [{bar}]  {done}/{n_active}  "
                f"{chunk.index[0].strftime('%Y-%m')} to {chunk.index[-1].strftime('%Y-%m')}  "
                f"alpha={obs_alpha*100:+.4f}%  p={pval:.3f}  [{pval_tag}]"
            )
            start += window

        print(f"[Rolling Placebo]  done.\n")
        return periods

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_stream(self) -> pd.Series:
        """Return the beta-neutral alpha stream (residuals + alpha intercept)."""
        if self._stream is None:
            self._stream = (self.core.residuals + self.core.alpha).dropna()
        return self._stream


# ── Module-level helper ───────────────────────────────────────────────────────

def _block_shuffle(arr: np.ndarray, block_size: int, rng) -> np.ndarray:
    """
    Shuffle `arr` in contiguous blocks of `block_size`.
    Preserves short-range autocorrelation within each block.
    """
    n        = len(arr)
    n_blocks = int(np.ceil(n / block_size))
    blocks   = [arr[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    indices  = rng.permutation(n_blocks)
    shuffled = np.concatenate([blocks[j] for j in indices])
    return shuffled[:n]
