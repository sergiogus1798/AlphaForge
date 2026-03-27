"""
run_regime.py — Launch Test 8: Regime Detection + Drawdown Clustering.

Opens one figure with four panels:
  1. Asset price with regime-coloured background
  2. Strategy equity curve with regime shading and drawdown markers
  3. Alpha / Beta / Sharpe per regime (grouped bars)
  4. Drawdown clustering scatter + runs-test / ACF verdicts

Usage:
    python run_regime.py                                  # first strategy
    python run_regime.py "XAUUSD/my sergix sample"        # specific strategy
    python run_regime.py "AUDJPY/Strategy 7.11.56" --states 2
    python run_regime.py "XAUUSD/my sergix sample" --capital 50000
"""

import sys
import warnings

from alphaforge.loader import load_folder
from alphaforge.IndividualAnalysis.AlphaDetection.data_cleaner import load_price_data
from alphaforge.IndividualAnalysis.AlphaDetection import AlphaDetector
from alphaforge.IndividualAnalysis.RegimeAnalysis import RegimeAnalysis, plot_regime_report

DEFAULT_FOLDER  = "strategies/approved"
DEFAULT_CAPITAL = 10_000.0


def _parse_args():
    key           = None
    n_states      = 3
    capital       = DEFAULT_CAPITAL
    smooth_window = 21

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--states" and i + 1 < len(args):
            n_states = int(args[i + 1]); i += 2
        elif a == "--capital" and i + 1 < len(args):
            capital = float(args[i + 1]); i += 2
        elif a == "--smooth" and i + 1 < len(args):
            smooth_window = int(args[i + 1]); i += 2
        elif not a.startswith("--"):
            key = a; i += 1
        else:
            i += 1

    return key, n_states, capital, smooth_window


def _pick_strategy(strategies: dict, key: str | None):
    if key and key in strategies:
        return key, strategies[key]
    if key:
        print(f"  Warning: '{key}' not found. Available:")
        for k in list(strategies.keys())[:10]:
            print(f"    {k}")
        print("  Falling back to first strategy.\n")
    name = next(iter(strategies))
    return name, strategies[name]


def _extract_pair(strategy_name: str) -> str:
    return strategy_name.split("/")[0].strip()


if __name__ == "__main__":
    key, n_states, capital, smooth_window = _parse_args()

    strategies = load_folder(DEFAULT_FOLDER)
    if not strategies:
        print("No strategies found in strategies/approved/. Exiting.")
        sys.exit(1)

    strategy_name, trades_df = _pick_strategy(strategies, key)
    pair = _extract_pair(strategy_name)

    print(f"  Strategy      :  {strategy_name}")
    print(f"  Pair          :  {pair}")
    print(f"  Capital       :  {capital:,.0f}")
    print(f"  HMM states    :  {n_states}")
    print(f"  Smooth window :  {smooth_window}d")
    print()

    try:
        price_df = load_price_data(pair, data_folder="AssetsData")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    detector = AlphaDetector(
        trades_df=trades_df, price_df=price_df,
        strategy_name=strategy_name, pair=pair,
        initial_capital=capital,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = detector.build_returns()

    analysis = RegimeAnalysis(
        aligned        = aligned,
        price_df       = price_df,
        strategy_name  = strategy_name,
        pair           = pair,
        initial_capital = capital,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analysis.run_all(n_states=n_states, smooth_window=smooth_window)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n  {'Regime':<12}  {'Days':>5}  {'Trades':>6}  "
          f"{'alpha/d':>8}  {'t(a)':>6}  {'IR':>7}  Verdict")
    print("  " + "-" * 65)
    for rs in report.regime_stats:
        a  = f"{rs.alpha*100:+.4f}%" if not (rs.alpha != rs.alpha) else "  n/a  "
        t  = f"{rs.alpha_tstat:+.2f}"  if not (rs.alpha_tstat != rs.alpha_tstat) else " n/a"
        sh = f"{rs.info_ratio:.3f}"    if not (rs.info_ratio != rs.info_ratio) else " n/a"
        print(f"  {rs.label:<12}  {rs.n_days:>5}  {rs.n_active:>6}  "
              f"{a:>8}  {t:>6}  {sh:>7}  [{rs.verdict}]")

    print()
    rt  = report.runs_test
    lac = report.loss_acf
    if not (rt.z_stat != rt.z_stat):
        print(f"  Runs test:  z={rt.z_stat:+.2f}  p={rt.p_value:.3f}  [{rt.verdict}]")
    print(f"  Loss ACF:   lag1={lac.lag1_acf:+.3f}  LB-p={lac.ljung_box_pval:.3f}  [{lac.verdict}]")
    print(f"  Drawdowns:  {len(report.drawdown_periods)} episodes found")
    print()

    def _reload(chosen_key: str, n_st: int, smooth: int):
        """Load a different strategy / params and return (RegimeReport, name, pair)."""
        name = chosen_key
        df   = strategies[chosen_key]
        pr   = _extract_pair(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf  = load_price_data(pr, data_folder="AssetsData")
            det = AlphaDetector(trades_df=df, price_df=pf,
                                strategy_name=name, pair=pr,
                                initial_capital=capital)
            aln = det.build_returns()
        ana = RegimeAnalysis(aligned=aln, price_df=pf,
                             strategy_name=name, pair=pr,
                             initial_capital=capital)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpt = ana.run_all(n_states=n_st, smooth_window=smooth)
        return rpt, name, pr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_regime_report(
            report,
            strategy_name  = strategy_name,
            pair           = pair,
            all_strategies = strategies,
            reload_fn      = _reload,
        )
