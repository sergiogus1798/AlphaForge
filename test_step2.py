"""
test_step2.py — Test weighting methods (Step 2) + critical day rescaling (Step 3).

Picks a portfolio of 5 strategies, applies all 4 weighting methods, and prints:
  - Table 1 : weights per strategy for each method
  - Table 2 : critical day + scale factor for each method
  - Table 3 : performance metrics BEFORE rescaling
  - Table 4 : performance metrics AFTER rescaling
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate

from alphaforge.loader.loader import load_folder
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.filters import filter_static_only
from alphaforge.portfolio.generator.sampler import sample_combinations
from alphaforge.portfolio.generator.weighting import compute_all_weights, build_weighted_portfolio
from alphaforge.portfolio.generator.scaler import rescale
from alphaforge.portfolio.generator.validator import validate
from alphaforge.portfolio.stress.mae_stress import run_mae_stress
from alphaforge.portfolio.dashboard import run_portfolio_dashboard
from alphaforge.metrics.metrics import compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────

config = PortfolioConfig(random_seed=42)

# ── Load ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  AlphaForge — Step 2 + 3: Weighting & Rescaling Test")
print("=" * 70)

strategies = load_folder("TradingData")
print(f"\n  {len(strategies)} strategies loaded.\n")

# ── Pick a portfolio (relaxed thresholds — correlation irrelevant for this test)

config_relaxed = PortfolioConfig(
    min_strategies=5, max_strategies=5,
    max_pearson_corr=0.99, max_spearman_corr=0.99,
    max_co_loss_freq=0.99, same_asset_same_day=False,
    n_portfolios=200, random_seed=42,
)
combinations = sample_combinations(strategies, config_relaxed)
static_passed, _ = filter_static_only(combinations, strategies, config_relaxed, verbose=False)
combo = static_passed[0] if static_passed else tuple(list(strategies.keys())[:5])

print(f"  Portfolio ({len(combo)} strategies):")
for name in combo:
    print(f"    • {name}")
print()

# ── Step 2: compute weights ───────────────────────────────────────────────────

all_weights = compute_all_weights(combo, strategies)
n = len(combo)
short = {name: name.split("/")[-1] for name in combo}
methods = list(all_weights.keys())
method_labels = [m.replace("_", " ").title() for m in methods]

# ── Build weighted portfolio DataFrames (pre-scale) ───────────────────────────

weighted_portfolios = {
    method: build_weighted_portfolio(combo, strategies, all_weights[method])
    for method in methods
}

# ── Step 3: rescale ───────────────────────────────────────────────────────────

scaled_results = {
    method: rescale(combo, method, all_weights[method], weighted_portfolios[method], config)
    for method in methods
}

# ── TABLE 1: Weights ──────────────────────────────────────────────────────────

weight_rows = []
for name in combo:
    row = {"Strategy": short[name]}
    for method, label in zip(methods, method_labels):
        row[label] = f"{all_weights[method][name]:.4f}"
    weight_rows.append(row)
totals = {"Strategy": "─ TOTAL"}
for label in method_labels:
    totals[label] = "1.0000"
weight_rows.append(totals)

print("  TABLE 1 — Weights per strategy")
print("  " + "─" * 66)
print("  " + tabulate(weight_rows, headers="keys", tablefmt="simple"))
print()

# ── TABLE 2: Critical day + rescaling ────────────────────────────────────────

scale_rows = []
for method, label in zip(methods, method_labels):
    sr = scaled_results[method]
    scale_rows.append({
        "Method"          : label,
        "Critical Day"    : str(sr.critical_day),
        "Worst Day ($)"   : f"${sr.worst_day_loss:,.2f}",
        "Scale Factor"    : f"{sr.scale_factor:.4f}",
        "Daily Limit ($)" : f"${config.daily_loss_limit_usd:,.0f}",
    })

# Also show final risk per trade per strategy for each method
print("  TABLE 2 — Critical day & scale factors")
print("  " + "─" * 66)
print("  " + tabulate(scale_rows, headers="keys", tablefmt="simple"))
print()

risk_rows = []
for name in combo:
    row = {"Strategy": short[name]}
    for method, label in zip(methods, method_labels):
        row[label] = f"${scaled_results[method].risk_per_trade[name]:,.2f}"
    risk_rows.append(row)

print("  TABLE 2b — Final risk per trade after rescaling ($)")
print("  " + "─" * 66)
print("  " + tabulate(risk_rows, headers="keys", tablefmt="simple"))
print()

# ── Metrics helper ────────────────────────────────────────────────────────────

METRICS = {
    "total_profit":          ("Total Profit ($)",      "${:,.0f}"),
    "yearly_avg_pct_return": ("Avg Annual Return (%)",  "{:.2f}%"),
    "cagr":                  ("CAGR (%)",               "{:.2f}%"),
    "sharpe_ratio":          ("Sharpe Ratio",            "{:.3f}"),
    "profit_factor":         ("Profit Factor",           "{:.2f}"),
    "return_dd_ratio":       ("Return/DD Ratio",         "{:.2f}"),
    "winning_percentage":    ("Win Rate (%)",            "{:.1f}%"),
    "drawdown":              ("Max Drawdown ($)",        "${:,.0f}"),
    "pct_drawdown":          ("Max Drawdown (%)",        "{:.2f}%"),
    "average_trade":         ("Avg Trade ($)",           "${:.2f}"),
    "num_trades":            ("Num Trades",              "{:.0f}"),
}

def metrics_table_rows(portfolio_map, label_map):
    rows = []
    computed = {
        method: compute_metrics(df, initial_capital=config.account_balance)
        for method, df in portfolio_map.items()
    }
    for key, (label, fmt) in METRICS.items():
        row = {"Metric": label}
        for method, mlabel in label_map.items():
            val = computed[method][key]
            try:
                row[mlabel] = fmt.format(val)
            except (ValueError, TypeError):
                row[mlabel] = str(val)
        rows.append(row)
    return rows

label_map = {m: m.replace("_", " ").title() for m in methods}

# ── TABLE 3: Metrics BEFORE rescaling ────────────────────────────────────────

print("  TABLE 3 — Performance metrics BEFORE rescaling")
print("  " + "─" * 66)
before_rows = metrics_table_rows(weighted_portfolios, label_map)
print("  " + tabulate(before_rows, headers="keys", tablefmt="simple"))
print()

# ── TABLE 4: Metrics AFTER rescaling ─────────────────────────────────────────

scaled_portfolios = {method: scaled_results[method].portfolio_df for method in methods}

print("  TABLE 4 — Performance metrics AFTER rescaling")
print("  " + "─" * 66)
after_rows = metrics_table_rows(scaled_portfolios, label_map)
print("  " + tabulate(after_rows, headers="keys", tablefmt="simple"))
print()

# ── Equity curve plot ─────────────────────────────────────────────────────────

BG      = "#0f1117"
AX_BG   = "#1a1a2e"
BEFORE  = "#4a4a7a"      # muted purple — pre-scale
AFTER   = "#00d4ff"      # cyan — post-scale
CRIT    = "#e74c3c"      # red — critical day marker
GRID    = "#2a2a4a"
TEXT    = "#e0e0e0"
DIM     = "#888888"


def _equity_curve(portfolio_df: pd.DataFrame) -> pd.Series:
    """Daily cumulative P&L indexed by date."""
    daily = (
        portfolio_df
        .groupby(portfolio_df["Close time"].dt.date)["Profit/Loss"]
        .sum()
    )
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(date_range, fill_value=0.0).cumsum()


fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
fig.suptitle("Equity Curves — All Weighting Methods  |  Before & After Rescaling",
             color=TEXT, fontsize=13, y=1.01)

for ax, method in zip(axes.flat, methods):
    sr     = scaled_results[method]
    label  = method.replace("_", " ").title()

    eq_before = _equity_curve(weighted_portfolios[method])
    eq_after  = _equity_curve(sr.portfolio_df)

    # Critical day equity value (after scaling — that's what matters for the limit)
    crit_ts    = pd.Timestamp(sr.critical_day)
    crit_val   = eq_after.get(crit_ts, None)

    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.axhline(0, color=GRID, linewidth=0.8)

    # Before (muted)
    ax.plot(eq_before.index, eq_before.values,
            color=BEFORE, linewidth=1.2, linestyle="--",
            label=f"Before  (peak ${eq_before.max():,.0f})", alpha=0.8)

    # After (bright)
    ax.plot(eq_after.index, eq_after.values,
            color=AFTER, linewidth=1.8,
            label=f"After   (peak ${eq_after.max():,.0f})")

    # Critical day red dot
    if crit_val is not None:
        ax.scatter(crit_ts, crit_val, color=CRIT, zorder=5, s=60)
        ax.annotate(
            f"  {sr.critical_day}\n  worst: ${sr.worst_day_loss:,.0f}",
            xy=(crit_ts, crit_val),
            xytext=(10, -30),
            textcoords="offset points",
            color=CRIT,
            fontsize=7,
            arrowprops=dict(arrowstyle="-", color=CRIT, lw=0.8),
        )

    # Scale factor badge
    ax.text(0.02, 0.97,
            f"scale ×{sr.scale_factor:.3f}",
            transform=ax.transAxes,
            color=DIM, fontsize=8, va="top")

    ax.set_title(label, color=TEXT, fontsize=10, pad=6)
    ax.set_ylabel("Cumulative P&L ($)", color=TEXT, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, facecolor=AX_BG, labelcolor=TEXT, loc="upper left")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )

plt.tight_layout()
plt.show()

# ── Step 4: Validate (drawdown check) ────────────────────────────────────────

print("=" * 70)
print("  STEP 4 — Drawdown Validation")
print("=" * 70)
print(f"  Limit: ${config.total_drawdown_limit_usd:,.0f}  "
      f"({config.total_drawdown_limit_pct*100:.0f}% of ${config.account_balance:,.0f})\n")

valid_portfolios = []
validation_rows  = []

for method in methods:
    sr    = scaled_results[method]
    label = method.replace("_", " ").title()
    vp    = validate(sr, config)

    if vp:
        valid_portfolios.append(vp)
        status   = "PASS"
        dd_usd   = f"${vp.max_drawdown_usd:,.0f}"
        dd_pct   = f"{vp.max_drawdown_pct:.2f}%"
    else:
        # Compute DD just for the report even though it failed
        from alphaforge.portfolio.generator.validator import _compute_drawdown
        dd_usd_val, dd_pct_val = _compute_drawdown(sr.portfolio_df, config.account_balance)
        status = "FAIL"
        dd_usd = f"${dd_usd_val:,.0f}"
        dd_pct = f"{dd_pct_val:.2f}%"

    validation_rows.append({
        "Method"          : label,
        "Max DD ($)"      : dd_usd,
        "Max DD (%)"      : dd_pct,
        f"Limit (${config.total_drawdown_limit_usd:,.0f})" : status,
    })

print("  " + tabulate(validation_rows, headers="keys", tablefmt="simple"))
print(f"\n  {len(valid_portfolios)}/{len(methods)} portfolios passed validation.\n")

# ── Step 5: MAE Stress Test ───────────────────────────────────────────────────

print("=" * 70)
print("  STEP 5 — MAE Stress Test")
print("=" * 70)
print(f"  Daily limit: ${config.daily_loss_limit_usd:,.0f}  |  "
      f"Total DD limit: ${config.total_drawdown_limit_usd:,.0f}\n")

stress_rows = []
for vp in valid_portfolios:
    label  = vp.method.replace("_", " ").title()
    stress = run_mae_stress(vp, config)
    vp.stress = stress

    stress_rows.append({
        "Method"            : label,
        "Worst Day (MAE $)" : f"${stress.worst_day_loss_mae:,.2f}",
        "Critical Day (MAE)": str(stress.critical_day_mae),
        "Max DD (MAE $)"    : f"${stress.max_drawdown_usd_mae:,.0f}",
        "Max DD (MAE %)"    : f"{stress.max_drawdown_pct_mae:.2f}%",
        "MAE Coverage"      : f"{stress.mae_coverage_pct:.0f}%",
        "Result"            : "PASS" if stress.passed else "FAIL",
    })

if stress_rows:
    print("  " + tabulate(stress_rows, headers="keys", tablefmt="simple"))
else:
    print("  No valid portfolios to stress test.")

# ── Launch Dashboard ──────────────────────────────────────────────────────────

if valid_portfolios:
    print(f"\n  Launching Portfolio Explorer with {len(valid_portfolios)} portfolio(s)...")
    run_portfolio_dashboard(valid_portfolios, config, port=8060)
else:
    print("\n  No valid portfolios — skipping dashboard.")
