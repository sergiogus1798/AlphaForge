"""
factor_tests.py — Test 9: Multi-Factor Exposure

Checks whether a strategy's apparent alpha is actually hidden beta to well-known
systematic risk factors:

  Factor 1 — Momentum (TSMOM):
      Signal = sign of cumulative asset return over t-252 to t-22 (12-1 month).
      Factor return = signal × r_asset_t.
      Interpretation: does the strategy load on trend-following?

  Factor 2 — Carry Proxy:
      Signal = sign of mean daily return over past 63 calendar days.
      Factor return = signal × r_asset_t.
      Interpretation: does the strategy load on carry / drift direction?

  Factor 3 — Vol-Selling:
      Factor return = -(σ_t_21d − σ_{t-22_21d}) / σ_long
      Normalised negative change in realised vol.
      Interpretation: does the strategy profit by being short volatility?

Pipeline:
  1. compute_factors()          — build MOM, VOL_SELL from price_df
  2. naive_ols()                — benchmark: r_strat ~ α + β·r_asset
  3. single_factor_ols()        — r_strat ~ α + β·factor  (2 separate regressions)
  4. multi_factor_ols()         — r_strat ~ α + β₁·MOM + β₂·VOL_SELL
  5. run()                      — full pipeline → FactorReport

Usage:
    from alphaforge.IndividualAnalysis.FactorExposure import FactorExposure

    tester = FactorExposure(aligned, price_df, strategy_name="AUDJPY/Strategy 11.12.166")
    report = tester.run()
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_OBS        = 60
_MOM_LONG       = 252   # days for momentum look-back
_MOM_SHORT      = 22    # days to skip at the front of momentum window (skip-1m)
_CARRY_WINDOW   = 63    # days for carry proxy drift estimate
_VOL_WINDOW     = 21    # days for realised-vol estimate
_VOL_STEP       = 22    # lag between current and previous vol estimate


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class SingleFactorResult:
    """Alpha/beta decomposition against a single factor."""
    factor_name:              str
    beta:                     float    # loading on the factor
    beta_tstat:               float
    beta_pval:                float
    beta_se:                  float    # standard error of beta
    alpha:                    float    # daily alpha controlling for this factor
    alpha_tstat:              float
    alpha_ann:                float    # annualised alpha (× 252)
    r_squared:                float
    alpha_reduction_pct:      float    # % of naive alpha eaten by this factor
    verdict:                  str      # "EXPOSED" / "CLEAN" / "WEAK"
    notes:                    list[str] = field(default_factory=list)


@dataclass
class MultiFactorResult:
    """Alpha/beta decomposition against all three factors simultaneously."""
    betas:                    dict     # {factor_name: beta}
    beta_tstats:              dict     # {factor_name: tstat}
    beta_pvals:               dict     # {factor_name: pval}
    beta_ses:                 dict     # {factor_name: se}
    alpha:                    float    # daily alpha after all three factors
    alpha_tstat:              float
    alpha_ann:                float    # annualised
    r_squared:                float
    residuals:                pd.Series
    alpha_reduction_pct:      float    # % of naive alpha eaten by all factors
    verdict:                  str
    notes:                    list[str] = field(default_factory=list)


@dataclass
class FactorReport:
    """Full factor exposure report for a strategy."""
    strategy_name:    str
    pair:             str
    n_obs:            int
    date_start:       str
    date_end:         str

    # Benchmark
    naive_alpha:      float      # daily alpha from simple r_strat ~ α + β·r_asset
    naive_alpha_tstat: float
    naive_alpha_ann:  float
    naive_beta:       float
    naive_r_squared:  float

    # Single-factor results (one per factor)
    single_factor:    list[SingleFactorResult]

    # Multi-factor result
    multi_factor:     MultiFactorResult

    # Factor return series (for plotting)
    factor_returns:   pd.DataFrame   # columns: MOM, VOL_SELL

    verdict:          str
    notes:            list[str] = field(default_factory=list)


# ── Factor computation ────────────────────────────────────────────────────────

def _compute_factors(
    price_df:  pd.DataFrame,
    mom_long:  int = _MOM_LONG,
    mom_short: int = _MOM_SHORT,
    vol_window: int = _VOL_WINDOW,
    vol_step:   int = _VOL_STEP,
) -> pd.DataFrame:
    """
    Compute daily systematic factor returns from OHLCV price data.

    Returns a DataFrame indexed by date with columns:
        MOM       — time-series momentum factor return
        VOL_SELL  — vol-selling factor return
    """
    close  = price_df["Close"].copy()
    r_raw  = close.pct_change()

    # ── Factor 1: Momentum (TSMOM) ────────────────────────────────────────────
    cum_long   = close.shift(mom_short) / close.shift(mom_long) - 1
    mom_signal = np.sign(cum_long)
    mom_factor = (mom_signal * r_raw).rename("MOM")

    # ── Factor 2: Vol-Selling ─────────────────────────────────────────────────
    # Require a full year for the normalisation denominator to avoid early-series
    # instability (tiny vol_norm → huge division result).
    vol_curr   = r_raw.rolling(vol_window).std()
    vol_prev   = vol_curr.shift(vol_step)
    vol_norm   = vol_curr.rolling(252, min_periods=252).mean()
    vol_raw    = -(vol_curr - vol_prev) / vol_norm.clip(lower=1e-8)
    # Hard-clip at ±0.5 (±50 %/day) to guard against any residual outliers.
    vol_factor = vol_raw.clip(lower=-0.5, upper=0.5).rename("VOL_SELL")

    factors = pd.concat([mom_factor, vol_factor], axis=1)
    factors.index = pd.to_datetime(factors.index).normalize()
    return factors


# ── Rolling factor exposure ───────────────────────────────────────────────────

def compute_rolling_factor_betas(
    df:     pd.DataFrame,
    window: int = 252,
    step:   int = 22,
) -> pd.DataFrame:
    """
    Compute rolling multi-factor OLS over the aligned dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: strategy, asset, MOM, VOL_SELL.
    window : int
        Rolling lookback window in observations.
    step : int
        How many observations to advance between windows.

    Returns
    -------
    pd.DataFrame with columns:
        date, alpha, alpha_ann, alpha_tstat,
        b_MOM, b_VOL_SELL,
        naive_alpha, naive_alpha_ann, naive_alpha_tstat,
        r_squared, n_obs
    """
    import warnings
    factor_cols = ["MOM", "VOL_SELL"]
    records = []

    indices = list(range(window - 1, len(df), step))
    for end_idx in indices:
        start_idx = end_idx - window + 1
        chunk = df.iloc[start_idx: end_idx + 1].dropna()
        if len(chunk) < _MIN_OBS:
            continue

        y       = chunk["strategy"]
        r_asset = chunk["asset"]
        factors = chunk[factor_cols]

        # Multi-factor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xm = sm.add_constant(factors, has_constant="add")
            rm = sm.OLS(y, Xm).fit()

            # Naive
            Xn = sm.add_constant(r_asset.rename("asset").to_frame(), has_constant="add")
            rn = sm.OLS(y, Xn).fit()

        records.append({
            "date":              chunk.index[-1],
            "alpha":             rm.params["const"],
            "alpha_ann":         rm.params["const"] * 252,
            "alpha_tstat":       rm.tvalues["const"],
            "b_MOM":             rm.params["MOM"],
            "b_VOL_SELL":        rm.params["VOL_SELL"],
            "t_MOM":             rm.tvalues["MOM"],
            "t_VOL_SELL":        rm.tvalues["VOL_SELL"],
            "naive_alpha":       rn.params["const"],
            "naive_alpha_ann":   rn.params["const"] * 252,
            "naive_alpha_tstat": rn.tvalues["const"],
            "r_squared":         rm.rsquared,
            "n_obs":             len(chunk),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── OLS helpers ───────────────────────────────────────────────────────────────

def _ols(y: pd.Series, X: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS with a constant prepended to X."""
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit()


def _annualise(daily_alpha: float) -> float:
    return daily_alpha * 252


def _verdict_single(beta_pval: float, alpha_reduction_pct: float) -> str:
    if beta_pval < 0.05 and alpha_reduction_pct > 20:
        return "EXPOSED"
    elif beta_pval < 0.10:
        return "WEAK"
    return "CLEAN"


def _verdict_multi(alpha_tstat: float, alpha_reduction_pct: float) -> str:
    if alpha_tstat >= 2.0 and alpha_reduction_pct < 50:
        return "ALPHA SURVIVES"
    elif alpha_tstat >= 2.0:
        return "ALPHA REDUCED"
    elif alpha_reduction_pct > 50:
        return "FACTOR BETA"
    return "INCONCLUSIVE"


# ── Main class ────────────────────────────────────────────────────────────────

class FactorExposure:
    """
    Multi-Factor Exposure tester (Test 9).

    Parameters
    ----------
    aligned : pd.DataFrame
        Output of AlphaDetector.build_returns(). Must have columns
        'strategy' and 'asset'.
    price_df : pd.DataFrame
        Daily OHLCV data for the traded pair. Must have a DatetimeIndex and
        a 'Close' column.
    strategy_name : str
        Human-readable label for the strategy.
    pair : str
        Instrument identifier, e.g. "AUDJPY".
    """

    def __init__(
        self,
        aligned:       pd.DataFrame,
        price_df:      pd.DataFrame,
        strategy_name: str = "Strategy",
        pair:          str = "",
        mom_long:      int = _MOM_LONG,
        mom_short:     int = _MOM_SHORT,
        vol_window:    int = _VOL_WINDOW,
        vol_step:      int = _VOL_STEP,
    ):
        self.aligned       = aligned.copy()
        self.price_df      = price_df.copy()
        self.strategy_name = strategy_name
        self.pair          = pair
        self.mom_long      = mom_long
        self.mom_short     = mom_short
        self.vol_window    = vol_window
        self.vol_step      = vol_step

        # Normalise index
        self.aligned.index    = pd.to_datetime(self.aligned.index).normalize()
        self.price_df.index   = pd.to_datetime(self.price_df.index).normalize()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> FactorReport:
        """Run the full factor exposure pipeline and return a FactorReport."""
        factors      = _compute_factors(
            self.price_df,
            mom_long   = self.mom_long,
            mom_short  = self.mom_short,
            vol_window = self.vol_window,
            vol_step   = self.vol_step,
        )
        df           = self._merge(factors)

        if len(df) < _MIN_OBS:
            raise ValueError(
                f"Only {len(df)} aligned observations — need at least {_MIN_OBS}."
            )

        y       = df["strategy"]
        r_asset = df["asset"]

        naive      = self._naive_ols(y, r_asset)
        naive_a    = naive.params["const"]
        naive_atst = naive.tvalues["const"]

        singles = []
        for fname in ["MOM", "VOL_SELL"]:
            singles.append(self._single_factor(y, df[fname], fname, naive_a))

        multi = self._multi_factor(y, df[["MOM", "VOL_SELL"]], naive_a)

        verdict, notes = self._overall_verdict(singles, multi)

        return FactorReport(
            strategy_name    = self.strategy_name,
            pair             = self.pair,
            n_obs            = len(df),
            date_start       = str(df.index[0].date()),
            date_end         = str(df.index[-1].date()),
            naive_alpha      = naive_a,
            naive_alpha_tstat= naive_atst,
            naive_alpha_ann  = _annualise(naive_a),
            naive_beta       = naive.params.get("asset", float("nan")),
            naive_r_squared  = naive.rsquared,
            single_factor    = singles,
            multi_factor     = multi,
            factor_returns   = df[["MOM", "VOL_SELL"]],
            verdict          = verdict,
            notes            = notes,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _merge(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Align strategy returns with factor returns on common dates."""
        df = self.aligned[["strategy", "asset"]].join(factors, how="inner")
        df = df.dropna()
        return df

    def _naive_ols(self, y: pd.Series, r_asset: pd.Series):
        return _ols(y, r_asset.rename("asset").to_frame())

    def _single_factor(
        self,
        y:         pd.Series,
        factor:    pd.Series,
        fname:     str,
        naive_a:   float,
    ) -> SingleFactorResult:
        res    = _ols(y, factor.rename(fname).to_frame())
        beta   = res.params[fname]
        btstat = res.tvalues[fname]
        bpval  = res.pvalues[fname]
        bse    = res.bse[fname]
        alpha  = res.params["const"]
        atstat = res.tvalues["const"]

        reduction = (
            (naive_a - alpha) / abs(naive_a) * 100
            if abs(naive_a) > 1e-10 else 0.0
        )

        verdict = _verdict_single(bpval, reduction)
        notes   = []
        if verdict == "EXPOSED":
            notes.append(
                f"{fname} explains {reduction:.1f}% of naive alpha "
                f"(β={beta:.3f}, t={btstat:.2f})"
            )

        return SingleFactorResult(
            factor_name          = fname,
            beta                 = beta,
            beta_tstat           = btstat,
            beta_pval            = bpval,
            beta_se              = bse,
            alpha                = alpha,
            alpha_tstat          = atstat,
            alpha_ann            = _annualise(alpha),
            r_squared            = res.rsquared,
            alpha_reduction_pct  = reduction,
            verdict              = verdict,
            notes                = notes,
        )

    def _multi_factor(
        self,
        y:       pd.Series,
        factors: pd.DataFrame,
        naive_a: float,
    ) -> MultiFactorResult:
        res     = _ols(y, factors)
        fnames  = list(factors.columns)

        betas   = {f: res.params[f]  for f in fnames}
        tstats  = {f: res.tvalues[f] for f in fnames}
        pvals   = {f: res.pvalues[f] for f in fnames}
        ses     = {f: res.bse[f]     for f in fnames}

        alpha   = res.params["const"]
        atstat  = res.tvalues["const"]

        reduction = (
            (naive_a - alpha) / abs(naive_a) * 100
            if abs(naive_a) > 1e-10 else 0.0
        )

        residuals = pd.Series(res.resid, index=factors.index, name="alpha_stream")
        verdict   = _verdict_multi(atstat, reduction)

        notes = []
        sig_factors = [f for f in fnames if pvals[f] < 0.05]
        if sig_factors:
            notes.append(
                f"Significant exposure to: {', '.join(sig_factors)}"
            )
        notes.append(
            f"Multi-factor model explains {reduction:.1f}% of naive alpha. "
            f"Residual alpha t-stat = {atstat:.2f}."
        )

        return MultiFactorResult(
            betas               = betas,
            beta_tstats         = tstats,
            beta_pvals          = pvals,
            beta_ses            = ses,
            alpha               = alpha,
            alpha_tstat         = atstat,
            alpha_ann           = _annualise(alpha),
            r_squared           = res.rsquared,
            residuals           = residuals,
            alpha_reduction_pct = reduction,
            verdict             = verdict,
            notes               = notes,
        )

    def _overall_verdict(
        self,
        singles: list[SingleFactorResult],
        multi:   MultiFactorResult,
    ) -> tuple[str, list[str]]:
        notes = []
        exposed = [s for s in singles if s.verdict == "EXPOSED"]

        if exposed:
            names = [s.factor_name for s in exposed]
            notes.append(f"Strategy is significantly exposed to: {', '.join(names)}.")

        notes.extend(multi.notes)

        if multi.verdict == "ALPHA SURVIVES":
            verdict = "ALPHA SURVIVES"
            notes.append("Alpha is robust — it persists after controlling for all three factors.")
        elif multi.verdict == "ALPHA REDUCED":
            verdict = "ALPHA REDUCED"
            notes.append(
                "Alpha is materially reduced by factor exposures but remains significant."
            )
        elif multi.verdict == "FACTOR BETA":
            verdict = "FACTOR BETA"
            notes.append(
                "Most of the apparent alpha is explained by systematic factor exposures."
            )
        else:
            verdict = "INCONCLUSIVE"

        return verdict, notes


# ── Pretty-print ──────────────────────────────────────────────────────────────

def print_factor_report(report: FactorReport) -> None:
    """Print a formatted summary of the factor exposure report."""
    SEP = "=" * 62
    SEP2 = "-" * 62

    def _p(s: str = "") -> None:
        print(s.encode("ascii", "replace").decode("ascii"))

    _p()
    _p(SEP)
    _p(f"  TEST 9 -- MULTI-FACTOR EXPOSURE")
    _p(f"  Strategy : {report.strategy_name}")
    _p(f"  Pair     : {report.pair}")
    _p(f"  Period   : {report.date_start} -> {report.date_end}  ({report.n_obs} obs)")
    _p(SEP)

    _p()
    _p("  NAIVE ALPHA (vs asset only)")
    _p(f"  {SEP2}")
    _p(f"  a = {report.naive_alpha*100:.4f}%/day  "
       f"({report.naive_alpha_ann*100:.2f}%/yr)   "
       f"t = {report.naive_alpha_tstat:+.2f}   "
       f"B_asset = {report.naive_beta:.3f}   "
       f"R2 = {report.naive_r_squared:.3f}")

    _p()
    _p("  SINGLE-FACTOR REGRESSIONS")
    _p(f"  {SEP2}")
    header = f"  {'Factor':<12}  {'B':>8}  {'t(B)':>7}  {'p(B)':>6}  "
    header += f"{'a(ann%)':>8}  {'t(a)':>6}  {'a reduc%':>9}  Verdict"
    _p(header)
    _p(f"  {'-'*62}")
    for s in report.single_factor:
        _p(
            f"  {s.factor_name:<12}  {s.beta:>8.4f}  {s.beta_tstat:>7.2f}  "
            f"{s.beta_pval:>6.3f}  {s.alpha_ann*100:>8.3f}  "
            f"{s.alpha_tstat:>6.2f}  {s.alpha_reduction_pct:>9.1f}  {s.verdict}"
        )

    mf = report.multi_factor
    _p()
    _p("  MULTI-FACTOR REGRESSION  (all three factors simultaneously)")
    _p(f"  {SEP2}")
    for fname in ["MOM", "VOL_SELL"]:
        _p(
            f"  B_{fname:<10}= {mf.betas[fname]:>8.4f}   "
            f"t = {mf.beta_tstats[fname]:>6.2f}   "
            f"p = {mf.beta_pvals[fname]:.3f}"
        )
    _p(f"  {'-'*40}")
    _p(
        f"  a (multi)  = {mf.alpha*100:.4f}%/day  "
        f"({mf.alpha_ann*100:.2f}%/yr)   "
        f"t = {mf.alpha_tstat:+.2f}"
    )
    _p(f"  R2         = {mf.r_squared:.4f}")
    _p(f"  a reduction= {mf.alpha_reduction_pct:.1f}% of naive alpha")

    _p()
    _p(f"  VERDICT: {report.verdict}")
    if report.notes:
        for n in report.notes:
            _p(f"  * {n}")

    _p()
    _p(SEP)
    _p()
