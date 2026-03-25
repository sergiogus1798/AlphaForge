"""
regime_hmm.py — Returns-based HMM regime comparison.

Fits independent GaussianHMMs on:
  - Strategy weekly returns  → what state is my strategy's performance in?
  - Market weekly returns    → what state is the market in?

Then compares the two state sequences to answer the core question:
  "Are my strategy's regimes driven by the market, or do they have
   their own independent structure?"

  Independent regimes → genuine alpha (strategy finds its own rhythm)
  Strongly aligned    → strategy is a disguised beta vehicle

Metrics:
  - Per-state: mean, std, annualised return, % time, persistence (transition matrix diagonal)
  - Confusion matrix: P(strategy state | market state) — key diagnostic
  - Cramers V: 0 = fully independent, 1 = perfectly correlated

Usage:
    from alphaforge.IndividualAnalysis.RegimeHMM import RegimeHMM

    hmm_model = RegimeHMM(aligned, strategy_name="AUDJPY/Strategy 11.12.166")
    report = hmm_model.run(n_states=3, n_seeds=30)
"""

from __future__ import annotations

import io
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@contextmanager
def _suppress_stdout():
    """Redirect stdout to /dev/null to silence hmmlearn convergence prints."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old

try:
    from hmmlearn import hmm as hmmlearn_hmm
    _HMMLEARN = True
except ImportError:
    _HMMLEARN = False


# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_WEEKS      = 52
_DEFAULT_STATES = 3
_DEFAULT_SEEDS  = 30

_STATE_LABELS = {
    2: ["Bad", "Good"],
    3: ["Bad", "Neutral", "Good"],
    4: ["Bad", "Below Avg", "Above Avg", "Good"],
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class HMMStateStats:
    """Statistics for a single HMM state."""
    state_id:    int
    label:       str      # e.g. "Bad" / "Neutral" / "Good"
    mean_weekly: float    # mean weekly return
    std_weekly:  float    # std of weekly returns
    mean_ann:    float    # annualised mean (× 52)
    std_ann:     float    # annualised std (× √52)
    sharpe_ann:  float    # annualised Sharpe (mean/std × √52)
    n_weeks:     int
    pct_time:    float    # % of time in this state
    persistence: float    # P(stay in same state) from transition matrix


@dataclass
class RegimeHMMReport:
    """Full returns-based HMM regime comparison report."""
    strategy_name:   str
    pair:            str
    n_states:        int
    date_start:      str
    date_end:        str
    n_weeks:         int

    # Time-series
    weekly_index:     pd.DatetimeIndex
    strategy_weekly:  pd.Series      # weekly strategy returns
    market_weekly:    pd.Series      # weekly market returns
    strategy_states:  np.ndarray     # state index at each week (sorted 0=bad)
    market_states:    np.ndarray     # same for market

    # Per-state summaries
    strategy_stats:   list           # list[HMMStateStats]
    market_stats:     list           # list[HMMStateStats]
    strategy_transmat: np.ndarray    # sorted transition matrix
    market_transmat:   np.ndarray    # sorted transition matrix

    # Comparison
    confusion_matrix:      np.ndarray   # shape (n_states, n_states) — rows=mkt, cols=strat
    confusion_matrix_norm: np.ndarray   # row-normalised (sums to 1 per row)
    cramers_v:             float        # 0=independent, 1=perfectly correlated

    verdict:  str
    notes:    list = field(default_factory=list)


# ── Internal HMM fitting ──────────────────────────────────────────────────────

def _fit_hmm(returns: np.ndarray, n_states: int, n_seeds: int) -> tuple:
    """
    Fit a GaussianHMM with best-of-n_seeds initialisation.

    Returns (sorted_states, sorted_means, sorted_stds, sorted_transmat).
    States are sorted ascending by mean return (state 0 = worst).
    """
    if not _HMMLEARN:
        raise ImportError(
            "hmmlearn is required for HMM regime analysis. "
            "Install it with: pip install hmmlearn"
        )

    X = returns.reshape(-1, 1)
    best_model  = None
    best_score  = -np.inf

    for seed in range(n_seeds):
        model = hmmlearn_hmm.GaussianHMM(
            n_components    = n_states,
            covariance_type = "full",
            n_iter          = 200,
            random_state    = seed,
            tol             = 1e-4,
        )
        with warnings.catch_warnings(), _suppress_stdout():
            warnings.simplefilter("ignore")
            try:
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score  = score
                    best_model  = model
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("HMM fitting failed for all seeds.")

    # Sort states by mean return (ascending: bad → good)
    means = best_model.means_.flatten()
    order = np.argsort(means)
    state_map = {old: new for new, old in enumerate(order)}

    raw_states     = best_model.predict(X)
    sorted_states  = np.array([state_map[s] for s in raw_states], dtype=int)
    sorted_means   = means[order]
    sorted_stds    = np.sqrt(best_model.covars_.flatten()[order])
    sorted_transmat = best_model.transmat_[np.ix_(order, order)]

    return sorted_states, sorted_means, sorted_stds, sorted_transmat


def _state_stats(
    states:    np.ndarray,
    returns:   pd.Series,
    transmat:  np.ndarray,
    n_states:  int,
) -> list[HMMStateStats]:
    labels = _STATE_LABELS.get(n_states, [str(i) for i in range(n_states)])
    result = []
    for i in range(n_states):
        mask = states == i
        r    = returns.values[mask]
        mu   = float(r.mean()) if len(r) else 0.0
        sig  = float(r.std())  if len(r) > 1 else 0.0
        sh   = (mu / sig * np.sqrt(52)) if sig > 1e-10 else 0.0
        result.append(HMMStateStats(
            state_id    = i,
            label       = labels[i],
            mean_weekly = mu,
            std_weekly  = sig,
            mean_ann    = mu * 52,
            std_ann     = sig * np.sqrt(52),
            sharpe_ann  = sh,
            n_weeks     = int(mask.sum()),
            pct_time    = float(mask.mean() * 100),
            persistence = float(transmat[i, i]),
        ))
    return result


def _confusion_matrix(
    market_states:   np.ndarray,
    strategy_states: np.ndarray,
    n_states:        int,
) -> np.ndarray:
    cm = np.zeros((n_states, n_states), dtype=int)
    for m, s in zip(market_states, strategy_states):
        cm[int(m), int(s)] += 1
    return cm


def _cramers_v(cm: np.ndarray) -> float:
    """
    Cramers V — normalised association strength for categorical variables.
    0 = completely independent, 1 = perfectly correlated.
    """
    n = cm.sum()
    if n == 0:
        return 0.0
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    col_sums = cm.sum(axis=0, keepdims=True).clip(min=1)
    expected = (row_sums * col_sums / n).astype(float)
    chi2 = float(((cm - expected) ** 2 / expected.clip(min=1e-10)).sum())
    k = min(cm.shape)
    if k <= 1 or n <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))).clip(0, 1))


def _verdict(v: float, n_states: int) -> tuple[str, list[str]]:
    notes = []
    if v < 0.10:
        verdict = "INDEPENDENT"
        notes.append(
            f"Cramers V = {v:.3f}: strategy regimes are essentially independent "
            "of market regimes. Strong evidence of genuine alpha."
        )
    elif v < 0.25:
        verdict = "WEAK ALIGNMENT"
        notes.append(
            f"Cramers V = {v:.3f}: mild association between strategy and market "
            "regimes. Alpha is likely real but with some market sensitivity."
        )
    elif v < 0.45:
        verdict = "MODERATE ALIGNMENT"
        notes.append(
            f"Cramers V = {v:.3f}: strategy regimes moderately track market regimes. "
            "Some of the apparent edge may be disguised beta."
        )
    else:
        verdict = "STRONG ALIGNMENT"
        notes.append(
            f"Cramers V = {v:.3f}: strategy regimes are strongly driven by market "
            "regimes. The strategy is largely a beta vehicle."
        )
    return verdict, notes


# ── Main class ────────────────────────────────────────────────────────────────

class RegimeHMM:
    """
    Returns-based HMM regime comparison.

    Parameters
    ----------
    aligned : pd.DataFrame
        Output of AlphaDetector.build_returns(). Columns: 'strategy', 'asset'.
    strategy_name : str
    pair : str
    """

    def __init__(
        self,
        aligned:       pd.DataFrame,
        strategy_name: str = "Strategy",
        pair:          str = "",
    ):
        self.aligned       = aligned.copy()
        self.strategy_name = strategy_name
        self.pair          = pair
        self.aligned.index = pd.to_datetime(self.aligned.index).normalize()

    def run(
        self,
        n_states: int = _DEFAULT_STATES,
        n_seeds:  int = _DEFAULT_SEEDS,
    ) -> RegimeHMMReport:
        """Fit HMMs on weekly returns and compare regime structures."""

        # ── Aggregate to weekly returns ───────────────────────────────────────
        weekly = (
            self.aligned[["strategy", "asset"]]
            .resample("W")
            .sum()
            .dropna()
        )

        if len(weekly) < _MIN_WEEKS:
            raise ValueError(
                f"Only {len(weekly)} weeks of data — need at least {_MIN_WEEKS}."
            )

        strat_w = weekly["strategy"]
        mkt_w   = weekly["asset"]

        # ── Fit HMMs ──────────────────────────────────────────────────────────
        print(f"  Fitting strategy HMM ({n_states} states, {n_seeds} seeds)...")
        s_states, s_means, s_stds, s_trans = _fit_hmm(
            strat_w.values, n_states, n_seeds
        )

        print(f"  Fitting market HMM  ({n_states} states, {n_seeds} seeds)...")
        m_states, m_means, m_stds, m_trans = _fit_hmm(
            mkt_w.values, n_states, n_seeds
        )

        # ── Per-state stats ───────────────────────────────────────────────────
        s_stats = _state_stats(s_states, strat_w, s_trans, n_states)
        m_stats = _state_stats(m_states, mkt_w,   m_trans, n_states)

        # ── Comparison ────────────────────────────────────────────────────────
        cm      = _confusion_matrix(m_states, s_states, n_states)
        cm_norm = (cm / cm.sum(axis=1, keepdims=True).clip(min=1)).round(3)
        v       = _cramers_v(cm)
        verdict, notes = _verdict(v, n_states)

        return RegimeHMMReport(
            strategy_name        = self.strategy_name,
            pair                 = self.pair,
            n_states             = n_states,
            date_start           = str(weekly.index[0].date()),
            date_end             = str(weekly.index[-1].date()),
            n_weeks              = len(weekly),
            weekly_index         = weekly.index,
            strategy_weekly      = strat_w,
            market_weekly        = mkt_w,
            strategy_states      = s_states,
            market_states        = m_states,
            strategy_stats       = s_stats,
            market_stats         = m_stats,
            strategy_transmat    = s_trans,
            market_transmat      = m_trans,
            confusion_matrix     = cm,
            confusion_matrix_norm= cm_norm,
            cramers_v            = v,
            verdict              = verdict,
            notes                = notes,
        )


# ── Pretty-print ──────────────────────────────────────────────────────────────

def print_regime_hmm_report(report: RegimeHMMReport) -> None:
    SEP = "=" * 64

    def _p(s=""): print(s.encode("ascii", "replace").decode("ascii"))

    _p(); _p(SEP)
    _p("  REGIME HMM — RETURNS-BASED COMPARISON")
    _p(f"  Strategy : {report.strategy_name}")
    _p(f"  Pair     : {report.pair}")
    _p(f"  Period   : {report.date_start} -> {report.date_end}  ({report.n_weeks} weeks)")
    _p(f"  States   : {report.n_states}")
    _p(SEP)

    _p()
    _p("  STRATEGY STATES  (sorted: 0=worst ... N-1=best)")
    _p(f"  {'-'*64}")
    _p(f"  {'State':<10} {'Label':<10} {'Mean%/wk':>9} {'Ann%':>8} "
       f"{'AnnSharpe':>10} {'%Time':>7} {'Persist':>8}")
    _p(f"  {'-'*64}")
    for st in report.strategy_stats:
        _p(f"  {st.state_id:<10} {st.label:<10} {st.mean_weekly*100:>9.3f} "
           f"{st.mean_ann*100:>8.2f} {st.sharpe_ann:>10.3f} "
           f"{st.pct_time:>7.1f} {st.persistence:>8.3f}")

    _p()
    _p("  MARKET STATES")
    _p(f"  {'-'*64}")
    _p(f"  {'State':<10} {'Label':<10} {'Mean%/wk':>9} {'Ann%':>8} "
       f"{'AnnSharpe':>10} {'%Time':>7} {'Persist':>8}")
    _p(f"  {'-'*64}")
    for st in report.market_stats:
        _p(f"  {st.state_id:<10} {st.label:<10} {st.mean_weekly*100:>9.3f} "
           f"{st.mean_ann*100:>8.2f} {st.sharpe_ann:>10.3f} "
           f"{st.pct_time:>7.1f} {st.persistence:>8.3f}")

    _p()
    _p("  CONFUSION MATRIX  (rows=market state, cols=strategy state)")
    _p("  Row-normalised: P(strategy state | market state)")
    _p(f"  {'-'*40}")
    labels = _STATE_LABELS.get(report.n_states, [str(i) for i in range(report.n_states)])
    header = f"  {'Mkt \\ Strat':<14}" + "".join(f"{lb:>10}" for lb in labels)
    _p(header)
    for i, lb in enumerate(labels):
        row = f"  {lb:<14}" + "".join(f"{report.confusion_matrix_norm[i,j]:>10.3f}"
                                      for j in range(report.n_states))
        _p(row)

    _p()
    _p(f"  Cramers V   = {report.cramers_v:.4f}  (0=independent, 1=correlated)")
    _p()
    _p(f"  VERDICT: {report.verdict}")
    for n in report.notes:
        _p(f"  * {n}")
    _p(); _p(SEP); _p()
