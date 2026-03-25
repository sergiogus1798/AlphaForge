"""
alpha_detection.py — Alpha vs Beta decomposition for trading strategies.

For each strategy, determines whether profits come from:
  - Beta  : passive market exposure (buy-and-hold of the traded pair)
  - Alpha : genuine edge independent of market direction

Full pipeline:
  1. build_returns()           Daily strategy returns (0 when flat) + asset returns
  2. run_core_regression()     OLS: R_strategy = α + β·R_asset
  3. run_vol_regression()      Add rolling-vol change as third factor (hidden risk check)
  4. run_overnight_intraday()  Decompose into overnight vs intraday components
  5. run_stability_test()      Split early/late — alpha must persist across both
  6. run_stress_test()         Check GFC 2008, COVID 2020, rate-hike 2022
  7. residual_stats()          Sharpe + drawdown of the pure alpha stream
  8. run_all()                 Run full pipeline → AlphaReport
  9. print_summary()           Pretty-print the report

Usage:
    from alphaforge.IndividualAnalysis.AlphaDetection import AlphaDetector

    detector = AlphaDetector(trades_df, price_df, strategy_name="AUDJPY7", pair="AUDJPY")
    report   = detector.run_all()
    detector.print_summary(report)
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Decision thresholds ────────────────────────────────────────────────────────

ALPHA_TSTAT_THRESHOLD = 2.0   # t-stat above which alpha is considered significant
R2_BETA_THRESHOLD     = 0.5   # R² above which beta explains "most" of returns
MIN_OBSERVATIONS      = 60    # minimum aligned days required to run a regression


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    """Stores the output of a single OLS regression."""
    alpha:       float
    beta:        float          # first regressor coefficient (asset return)
    alpha_tstat: float
    beta_tstat:  float
    r_squared:   float
    n_obs:       int
    residuals:   pd.Series      # the alpha stream (beta-neutral returns)
    verdict:     str            # "ALPHA" / "BETA" / "INCONCLUSIVE"
    notes:       list[str] = field(default_factory=list)


@dataclass
class AlphaReport:
    """Aggregates all regression results for one strategy."""
    strategy_name:   str
    pair:            str
    core:            Optional[RegressionResult] = None
    vol_adjusted:    Optional[RegressionResult] = None
    overnight:       Optional[RegressionResult] = None
    intraday:        Optional[RegressionResult] = None
    stability_early: Optional[RegressionResult] = None
    stability_late:  Optional[RegressionResult] = None
    stress_results:  dict = field(default_factory=dict)
    residual_sharpe: float = float("nan")
    residual_max_dd: float = float("nan")

    def verdict(self) -> str:
        """
        Overall verdict based on a majority vote from:
          core regression + early stability + late stability.
        Alpha must be consistent across time to be considered real.
        """
        if self.core is None:
            return "NOT RUN"
        candidates = [self.core, self.stability_early, self.stability_late]
        votes = [r.verdict for r in candidates if r is not None]
        if votes.count("ALPHA") >= 2:
            return "ALPHA"
        if votes.count("BETA") >= 2:
            return "BETA"
        return "INCONCLUSIVE"


# ── Main class ────────────────────────────────────────────────────────────────

class AlphaDetector:
    """
    Decomposes a strategy's returns into alpha, beta, and volatility exposure.

    Args:
        trades_df:       Clean trades DataFrame (from loader.load_strategy).
                         Required columns: Open time, Close time, Type,
                         Open price, Close price, Size, Profit/Loss.
        price_df:        D1 OHLCV DataFrame indexed by date (from AssetsData).
                         Required columns: Open, Close.
        strategy_name:   Human-readable label used in reports.
        pair:            Traded instrument identifier, e.g. "AUDJPY".
        initial_capital: Starting capital used to normalise PnL → return %.
    """

    def __init__(
        self,
        trades_df:       pd.DataFrame,
        price_df:        pd.DataFrame,
        strategy_name:   str   = "Strategy",
        pair:            str   = "",
        initial_capital: float = 10_000.0,
    ):
        self.trades_df       = trades_df.copy()
        self.price_df        = price_df.copy()
        self.strategy_name   = strategy_name
        self.pair            = pair
        self.initial_capital = initial_capital

        # Normalise price index to date-only (no time component)
        if not isinstance(self.price_df.index, pd.DatetimeIndex):
            self.price_df.index = pd.to_datetime(self.price_df.index)
        self.price_df.index = self.price_df.index.normalize()

        self._aligned: Optional[pd.DataFrame] = None   # cached by build_returns()

    # ── 1. Build aligned daily returns ────────────────────────────────────────

    def build_returns(self) -> pd.DataFrame:
        """
        Build a daily aligned DataFrame with two columns:
          strategy  — daily return (closed PnL / initial_capital), 0 when flat
          asset     — close-to-close return of the traded pair

        Strategy PnL is attributed to the day the trade CLOSES.
        Days with no closing trades receive a 0 strategy return, meaning the
        strategy was flat — this is critical for a correct beta estimate.
        Including zeros (flat days) prevents inflating beta by only comparing
        days when the strategy was active.

        Returns:
            pd.DataFrame with columns ["strategy", "asset"], DatetimeIndex.
        """
        # Asset: daily close-to-close returns
        asset_ret = self.price_df["Close"].pct_change().rename("asset")

        # Strategy: aggregate closed-trade PnL by close date
        trades = self.trades_df.dropna(subset=["Close time", "Profit/Loss"]).copy()
        trades["close_date"] = trades["Close time"].dt.normalize()

        daily_pnl = (
            trades.groupby("close_date")["Profit/Loss"]
            .sum()
            .rename("strategy")
        )
        strat_ret = daily_pnl / self.initial_capital

        # Reindex to every trading day in price data — flat days → 0
        strat_ret = strat_ret.reindex(asset_ret.index, fill_value=0.0)

        data          = pd.concat([strat_ret, asset_ret], axis=1).dropna()
        self._aligned = data
        return data

    # ── 2. Core regression ────────────────────────────────────────────────────

    def run_core_regression(
        self, data: Optional[pd.DataFrame] = None
    ) -> RegressionResult:
        """
        Core CAPM-style OLS regression:
            R_strategy = α + β · R_asset + ε

        A significant positive α means the strategy generates returns
        beyond what passive market exposure would explain.

        Args:
            data: pre-built aligned DataFrame; uses cached if not provided.

        Returns:
            RegressionResult with alpha, beta, t-stats, R², residuals, verdict.
        """
        data = data if data is not None else self._get_aligned()
        return self._run_ols(data["strategy"], data[["asset"]], label="core")

    # ── 3. Volatility-adjusted regression ─────────────────────────────────────

    def run_vol_regression(
        self,
        data:   Optional[pd.DataFrame] = None,
        window: int = 20,
    ) -> RegressionResult:
        """
        Two-factor OLS:
            R_strategy = α + β₁ · R_asset + β₂ · ΔVol + ε

        ΔVol = percentage change of the 20-day rolling std of asset returns.

        Interpretation of β₂:
          β₂ < 0 → strategy is implicitly short volatility — dangerous,
                   profits may come from low-vol regimes and collapse in crises.
          β₂ > 0 → strategy benefits from volatility — more robust.
          β₂ ≈ 0 → returns are clean of volatility timing.

        Args:
            window: rolling window (days) for the vol estimate.
        """
        data = (data if data is not None else self._get_aligned()).copy()
        vol             = data["asset"].rolling(window).std()
        data["vol_chg"] = vol.pct_change()
        data            = data.dropna()
        return self._run_ols(
            data["strategy"], data[["asset", "vol_chg"]], label="vol"
        )

    # ── 4. Overnight vs intraday decomposition ─────────────────────────────────

    def run_overnight_intraday(
        self,
    ) -> tuple[RegressionResult, RegressionResult]:
        """
        Decomposes both asset and strategy returns into overnight vs intraday.

        Asset side:
          overnight = Open[t] / Close[t-1] − 1  (gap: macro news, weekend risk)
          intraday  = Close[t] / Open[t]  − 1   (session move: execution, flow)

        Strategy side:
          overnight trades = trades that opened on a previous day (held overnight)
          intraday trades  = trades that open and close on the same calendar day

        Two separate regressions answer:
          - Is the edge in holding positions overnight? (macro / event-driven)
          - Is the edge in intraday execution? (microstructure / timing)

        Returns:
            (overnight_result, intraday_result)
        """
        price      = self.price_df
        prev_close = price["Close"].shift(1)

        overnight_asset = (price["Open"] / prev_close - 1).rename("asset_overnight")
        intraday_asset  = (price["Close"] / price["Open"] - 1).rename("asset_intraday")

        trades = self.trades_df.dropna(
            subset=["Open time", "Close time", "Profit/Loss"]
        ).copy()
        trades["open_date"]  = trades["Open time"].dt.normalize()
        trades["close_date"] = trades["Close time"].dt.normalize()
        trades["same_day"]   = trades["open_date"] == trades["close_date"]

        overnight_pnl = (
            trades[~trades["same_day"]]
            .groupby("close_date")["Profit/Loss"].sum()
            / self.initial_capital
        ).rename("strategy_overnight")

        intraday_pnl = (
            trades[trades["same_day"]]
            .groupby("close_date")["Profit/Loss"].sum()
            / self.initial_capital
        ).rename("strategy_intraday")

        all_days        = overnight_asset.index
        strat_overnight = overnight_pnl.reindex(all_days, fill_value=0.0)
        strat_intraday  = intraday_pnl.reindex(all_days, fill_value=0.0)

        df_on = pd.concat([strat_overnight, overnight_asset], axis=1).dropna()
        df_id = pd.concat([strat_intraday,  intraday_asset],  axis=1).dropna()

        res_overnight = self._run_ols(
            df_on["strategy_overnight"], df_on[["asset_overnight"]], label="overnight"
        )
        res_intraday = self._run_ols(
            df_id["strategy_intraday"], df_id[["asset_intraday"]], label="intraday"
        )
        return res_overnight, res_intraday

    # ── 5. Stability test ─────────────────────────────────────────────────────

    def run_stability_test(
        self, data: Optional[pd.DataFrame] = None
    ) -> tuple[RegressionResult, RegressionResult]:
        """
        Chronological early / late split test.

        Splits the full aligned data in half and runs the core regression on
        each period independently. A real alpha must be significant in BOTH
        halves — if it only appears in one period, it is likely regime-specific
        or a result of overfitting on that sub-period.

        Returns:
            (early_result, late_result)
        """
        data  = data if data is not None else self._get_aligned()
        mid   = len(data) // 2
        early = data.iloc[:mid]
        late  = data.iloc[mid:]
        return (
            self._run_ols(early["strategy"], early[["asset"]], label="early"),
            self._run_ols(late["strategy"],  late[["asset"]],  label="late"),
        )

    # ── 6. Stress / regime tests ──────────────────────────────────────────────

    def run_stress_test(
        self,
        data:    Optional[pd.DataFrame] = None,
        regimes: Optional[dict[str, tuple[str, str]]] = None,
    ) -> dict[str, Optional[RegressionResult]]:
        """
        Run the core regression within named crisis / regime windows.

        Default regimes:
          GFC 2008       : 2007-07-01 to 2009-03-31  (credit crisis, extreme vol)
          COVID 2020     : 2020-01-01 to 2020-12-31  (liquidity shock, V-recovery)
          Rate hike 2022 : 2022-01-01 to 2022-12-31  (trend reversal, macro regime)

        A strategy with real alpha should survive all three.
        Returns None for any regime where fewer than MIN_OBSERVATIONS days overlap
        with the strategy backtest.

        Args:
            regimes: override with {"label": ("YYYY-MM-DD", "YYYY-MM-DD")}.
        """
        data = data if data is not None else self._get_aligned()

        if regimes is None:
            regimes = {
                "GFC 2008":       ("2007-07-01", "2009-03-31"),
                "COVID 2020":     ("2020-01-01", "2020-12-31"),
                "Rate hike 2022": ("2022-01-01", "2022-12-31"),
            }

        out: dict[str, Optional[RegressionResult]] = {}
        for label, (start, end) in regimes.items():
            subset = data.loc[start:end]
            if len(subset) < MIN_OBSERVATIONS:
                out[label] = None
                continue
            out[label] = self._run_ols(
                subset["strategy"], subset[["asset"]], label=label
            )
        return out

    # ── 7. Residual (alpha stream) analysis ───────────────────────────────────

    def residual_stats(
        self, core: Optional[RegressionResult] = None
    ) -> dict:
        """
        Evaluates the quality of the pure alpha stream (OLS residuals).

        The residual series is the beta-neutral strategy: what is left after
        stripping all market exposure. If it has a positive Sharpe and
        manageable drawdown, the alpha is real and persistent.
        If it collapses, the apparent alpha was just undetected beta or luck.

        Args:
            core: result from run_core_regression(); runs it if not provided.

        Returns dict:
            sharpe            — annualised Sharpe of the residual stream
            max_drawdown      — worst peak-to-trough of the residual equity curve
            cumulative_return — total return of the residual stream
        """
        if core is None:
            core = self.run_core_regression()

        # Add alpha back to residuals → beta-hedged series (mean = alpha, not 0)
        # OLS residuals = y - alpha - beta*x  (mean=0 by construction)
        # Alpha stream  = y - beta*x          (mean=alpha, what we actually want)
        resid = (core.residuals + core.alpha).dropna()
        if len(resid) < 10:
            return {
                "sharpe":            float("nan"),
                "max_drawdown":      float("nan"),
                "cumulative_return": float("nan"),
            }

        sharpe  = (
            resid.mean() / resid.std() * np.sqrt(252)
            if resid.std() > 0 else float("nan")
        )
        equity  = (1 + resid).cumprod()
        peak    = equity.cummax()
        max_dd  = float(((peak - equity) / peak).max())
        cum_ret = float(equity.iloc[-1] - 1)

        return {
            "sharpe":            round(sharpe,  3),
            "max_drawdown":      round(max_dd,  4),
            "cumulative_return": round(cum_ret,  4),
        }

    # ── 8. Full pipeline ──────────────────────────────────────────────────────

    def run_all(self) -> AlphaReport:
        """
        Executes the complete alpha detection pipeline in order:
          1. build_returns           daily alignment of strategy + asset
          2. core regression         overall alpha/beta
          3. volatility regression   hidden vol exposure
          4. overnight / intraday    where the edge lives
          5. stability split         early vs late persistence
          6. stress tests            GFC, COVID, 2022
          7. residual stats          quality of the pure alpha stream

        Returns:
            AlphaReport with all results and an overall majority-vote verdict.
        """
        data = self.build_returns()

        report = AlphaReport(strategy_name=self.strategy_name, pair=self.pair)

        if len(data) < MIN_OBSERVATIONS:
            print(
                f"[{self.strategy_name}] Only {len(data)} aligned days — "
                f"need at least {MIN_OBSERVATIONS}. Skipping."
            )
            return report

        report.core         = self.run_core_regression(data)
        report.vol_adjusted = self.run_vol_regression(data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report.overnight, report.intraday = self.run_overnight_intraday()

        report.stability_early, report.stability_late = self.run_stability_test(data)
        report.stress_results = self.run_stress_test(data)

        rs = self.residual_stats(report.core)
        report.residual_sharpe = rs["sharpe"]
        report.residual_max_dd = rs["max_drawdown"]

        return report

    # ── 9. Summary printer ────────────────────────────────────────────────────

    def print_summary(self, report: Optional[AlphaReport] = None) -> None:
        """
        Pretty-print a full AlphaReport to stdout.
        Runs the full pipeline first if report is None.
        """
        if report is None:
            report = self.run_all()

        W   = 70
        dbl = "═" * W

        def _row(label, r):
            if r is None:
                print(f"  {label:<26} — not enough data in this window")
                return
            extra = f"  ← {r.notes[0]}" if r.notes else ""
            print(
                f"  {label:<26}  α={r.alpha:+.5f}  β={r.beta:+.3f}  "
                f"t(α)={r.alpha_tstat:+.2f}  R²={r.r_squared:.3f}  "
                f"n={r.n_obs:<5}  [{r.verdict}]{extra}"
            )

        print(f"\n{dbl}")
        print(f"  ALPHA DETECTION REPORT")
        print(f"  Strategy  :  {report.strategy_name}")
        print(f"  Pair      :  {report.pair}")
        print(f"  Verdict   :  ★  {report.verdict()}  ★")
        print(dbl)

        print(f"\n  ── Core regression {'─' * 50}")
        _row("Overall", report.core)

        print(f"\n  ── Stability (chronological split) {'─' * 34}")
        _row("Early half", report.stability_early)
        _row("Late half",  report.stability_late)

        print(f"\n  ── Decomposition {'─' * 52}")
        _row("Overnight",    report.overnight)
        _row("Intraday",     report.intraday)
        _row("Vol-adjusted", report.vol_adjusted)

        print(f"\n  ── Regime / stress tests {'─' * 44}")
        for name, r in report.stress_results.items():
            _row(name, r)

        if not np.isnan(report.residual_sharpe):
            print(f"\n  ── Alpha stream (beta-neutral residuals) {'─' * 28}")
            print(f"  Sharpe (annualised)  :  {report.residual_sharpe:.3f}")
            print(f"  Max drawdown         :  {report.residual_max_dd * 100:.2f}%")

        print(dbl)

    # ── 10. Rolling alpha / beta ──────────────────────────────────────────────

    def rolling_alpha_beta(
        self,
        data:   Optional[pd.DataFrame] = None,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Compute alpha, beta, R², and alpha t-stat on a rolling window.

        At each date t, fits OLS on the preceding `window` days.
        Useful for spotting regime changes — e.g. a strategy that generated
        alpha in one period but became beta-driven later.

        Args:
            window: number of days per rolling window (default 60).

        Returns:
            DataFrame indexed by date with columns:
              alpha, beta, r_squared, alpha_tstat
        """
        data = data if data is not None else self._get_aligned()
        rows = []
        for i in range(window, len(data) + 1):
            chunk = data.iloc[i - window : i]
            r     = self._run_ols(chunk["strategy"], chunk[["asset"]], label="roll")
            rows.append({
                "date":        data.index[i - 1],
                "alpha":       r.alpha,
                "beta":        r.beta,
                "r_squared":   r.r_squared,
                "alpha_tstat": r.alpha_tstat,
            })
        return pd.DataFrame(rows).set_index("date")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_aligned(self) -> pd.DataFrame:
        """Return cached aligned data, building it if not yet done."""
        if self._aligned is None:
            self._aligned = self.build_returns()
        return self._aligned

    def _run_ols(
        self,
        y:     pd.Series,
        X_raw: pd.DataFrame,
        label: str = "",
    ) -> RegressionResult:
        """
        Internal OLS runner shared by all regression methods.
        Adds a constant, aligns y and X, fits the model, and applies
        the ALPHA / BETA / INCONCLUSIVE decision rules.
        """
        df = pd.concat([y, X_raw], axis=1).dropna()

        if len(df) < MIN_OBSERVATIONS:
            return RegressionResult(
                alpha=float("nan"),       beta=float("nan"),
                alpha_tstat=float("nan"), beta_tstat=float("nan"),
                r_squared=float("nan"),   n_obs=len(df),
                residuals=pd.Series(dtype=float, name=f"resid_{label}"),
                verdict="INCONCLUSIVE",
                notes=[f"Only {len(df)} observations (min {MIN_OBSERVATIONS})"],
            )

        y_    = df.iloc[:, 0]
        X_    = sm.add_constant(df.iloc[:, 1:])
        model = sm.OLS(y_, X_).fit()

        alpha       = float(model.params["const"])
        beta        = float(model.params.iloc[1])
        alpha_tstat = float(model.tvalues["const"])
        beta_tstat  = float(model.tvalues.iloc[1])
        r2          = float(model.rsquared)
        residuals   = model.resid.rename(f"resid_{label}")

        # ── Decision rules ──────────────────────────────────────────────────
        notes     = []
        alpha_sig = (abs(alpha_tstat) >= ALPHA_TSTAT_THRESHOLD) and (alpha > 0)
        beta_dom  = r2 >= R2_BETA_THRESHOLD

        if alpha_sig and not beta_dom:
            verdict = "ALPHA"
        elif beta_dom and not alpha_sig:
            verdict = "BETA"
            notes.append(f"R²={r2:.2f} — beta explains most of the returns")
        elif alpha_sig and beta_dom:
            verdict = "ALPHA"
            notes.append("Significant alpha but also high R² — monitor market dependency")
        else:
            verdict = "INCONCLUSIVE"
            if alpha < 0:
                notes.append("Negative alpha — strategy loses value after market adjustment")
            else:
                notes.append("Alpha not significant (|t| < 2)")

        return RegressionResult(
            alpha=round(alpha, 6),        beta=round(beta, 4),
            alpha_tstat=round(alpha_tstat, 3), beta_tstat=round(beta_tstat, 3),
            r_squared=round(r2, 4),       n_obs=len(df),
            residuals=residuals,          verdict=verdict, notes=notes,
        )
