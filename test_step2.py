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

strategies = load_folder("strategies/approved")
print(f"\n  {len(strategies)} strategies loaded.\n")

# ── Pick a portfolio (relaxed thresholds — correlation irrelevant for this test)

config_relaxed = PortfolioConfig(
    min_strategies=5, max_strategies=5,
    max_pearson_corr=0.99, max_spearman_corr=0.99,
    max_co_loss_freq=0.99, same_asset_same_day=False,
    n_portfolios=200, random_seed=42,
)
combinations = sample_combinations(strategies, config_relaxed)
static_passed, *_ = filter_static_only(combinations, strategies, config_relaxed, verbose=False)
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


from matplotlib.widgets import Button as MplButton

METHOD_COLORS = {
    "equal":        "#00d4ff",   # cyan
    "min_variance": "#f0a500",   # amber
    "risk_parity":  "#2ecc71",   # green
    "hrp":          "#b267e6",   # purple
}

# Pre-build equity curves for both states
eq_before = {m: _equity_curve(weighted_portfolios[m]) for m in methods}
eq_after  = {m: _equity_curve(scaled_results[m].portfolio_df) for m in methods}

fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG)
fig.subplots_adjust(bottom=0.13, top=0.93, left=0.07, right=0.97)

ax.set_facecolor(AX_BG)
ax.tick_params(colors=TEXT, labelsize=9)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID)
ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
ax.axhline(0, color=GRID, linewidth=0.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.tick_params(axis="x", rotation=20)
ax.set_ylabel("Cumulative P&L ($)", color=TEXT, fontsize=9)

title_obj = ax.set_title(
    "Equity Curves — All Weighting Methods  |  After Rescaling",
    color=TEXT, fontsize=11, pad=8,
)

# Draw initial lines (rescaled) + critical day markers
lines         = {}
crit_scatters = {}
for method in methods:
    sr    = scaled_results[method]
    color = METHOD_COLORS[method]
    label = method.replace("_", " ").title()
    eq    = eq_after[method]

    line, = ax.plot(
        eq.index, eq.values,
        color=color, linewidth=2,
        label=f"{label}  (x{sr.scale_factor:.3f})",
    )
    lines[method] = line

    crit_ts  = pd.Timestamp(sr.critical_day)
    crit_val = eq.get(crit_ts, None)
    sc = ax.scatter(
        [crit_ts]  if crit_val is not None else [],
        [crit_val] if crit_val is not None else [],
        color=CRIT, zorder=5, s=60, marker="v",
    )
    crit_scatters[method] = sc

# Fix axis limits: x starts at first data date, y starts at 0
min_date = min(eq.index.min() for eq in eq_after.values())
max_date = max(eq.index.max() for eq in eq_after.values())
ax.set_xlim(left=min_date, right=max_date)
ax.set_ylim(bottom=0)

ax.legend(fontsize=9, facecolor=AX_BG, labelcolor=TEXT,
          loc="upper left", framealpha=0.8)

# ── Toggle button ─────────────────────────────────────────────────────────────
state = {"rescaled": True}

ax_btn = fig.add_axes([0.43, 0.02, 0.14, 0.05])
ax_btn.set_facecolor(AX_BG)
for sp in ax_btn.spines.values():
    sp.set_edgecolor(GRID)
btn_toggle = MplButton(ax_btn, "Showing: Rescaled", color=AX_BG, hovercolor="#21262d")
btn_toggle.label.set_color("#00d4ff")
btn_toggle.label.set_fontsize(9)


def _toggle(_event=None):
    state["rescaled"] = not state["rescaled"]
    rescaled = state["rescaled"]

    for method in methods:
        sr    = scaled_results[method]
        eq    = eq_after[method] if rescaled else eq_before[method]
        label = method.replace("_", " ").title()

        lines[method].set_xdata(eq.index)
        lines[method].set_ydata(eq.values)
        lines[method].set_label(
            f"{label}  (x{sr.scale_factor:.3f})" if rescaled else f"{label}  (pre-scale)"
        )

        crit_ts  = pd.Timestamp(sr.critical_day)
        crit_val = eq.get(crit_ts, None)
        if crit_val is not None:
            crit_scatters[method].set_offsets(
                [[mdates.date2num(crit_ts.date()), crit_val]]
            )
        else:
            crit_scatters[method].set_offsets([])

    btn_toggle.label.set_text("Showing: Rescaled" if rescaled else "Showing: Pre-scale")
    title_obj.set_text(
        "Equity Curves — All Weighting Methods  |  After Rescaling"
        if rescaled else
        "Equity Curves — All Weighting Methods  |  Before Rescaling"
    )
    ax.relim()
    ax.autoscale_view()
    ax.set_xlim(left=min_date, right=max_date)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, facecolor=AX_BG, labelcolor=TEXT,
              loc="upper left", framealpha=0.8)
    fig.canvas.draw_idle()


btn_toggle.on_clicked(_toggle)

# ── Metrics window ────────────────────────────────────────────────────────────

CYAN    = "#00d4ff"
SECTION = "#2a2a4a"

computed_after = {
    m: compute_metrics(scaled_results[m].portfolio_df, initial_capital=config.account_balance)
    for m in methods
}

method_labels_col = [m.replace("_", " ").title() for m in methods]

# Build rows: (row_label, [col0, col1, col2, col3], is_section)
table_rows = []

def _sec(label):
    table_rows.append((label, ["", "", "", ""], True))

def _row(label, values):
    table_rows.append((label, values, False))

_sec("PORTFOLIO COMPOSITION")
for name in combo:
    _row(short[name], [f"{all_weights[m][name]:.4f}" for m in methods])

_sec("SIZING")
_row("Scale Factor",  [f"x{scaled_results[m].scale_factor:.4f}"      for m in methods])
_row("Critical Day",  [str(scaled_results[m].critical_day)             for m in methods])
_row("Worst Day ($)", [f"${scaled_results[m].worst_day_loss:,.0f}"    for m in methods])

_sec("RISK / TRADE ($)")
for name in combo:
    _row(short[name], [f"${scaled_results[m].risk_per_trade[name]:,.0f}" for m in methods])

_sec("PERFORMANCE (after rescaling)")
for key, (label, fmt) in METRICS.items():
    vals = []
    for m in methods:
        try:
            vals.append(fmt.format(computed_after[m][key]))
        except Exception:
            vals.append("-")
    _row(label, vals)

# Build matplotlib table data
cell_text   = [r[1] for r in table_rows]
row_labels  = [r[0] for r in table_rows]
is_section  = [r[2] for r in table_rows]

fig2 = plt.figure(figsize=(14, 0.38 * len(table_rows) + 1.2), facecolor=BG)
fig2.suptitle("Portfolio Metrics — All Weighting Methods",
              color=TEXT, fontsize=12, y=0.99)

ax2 = fig2.add_subplot(111)
ax2.set_facecolor(BG)
ax2.axis("off")

tbl = ax2.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=method_labels_col,
    cellLoc="right",
    rowLoc="right",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.55)

# Style all cells
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_linewidth(0.5)

    if ri == 0:
        # Column header row
        cell.set_facecolor("#0f1117")
        if ci >= 0:
            color = list(METHOD_COLORS.values())[ci] if ci < len(methods) else TEXT
            cell.get_text().set_color(color)
            cell.get_text().set_fontweight("bold")
        else:
            cell.get_text().set_color(DIM)
    elif ci == -1:
        # Row label column
        sec = is_section[ri - 1]
        cell.set_facecolor(SECTION if sec else AX_BG)
        cell.get_text().set_color(CYAN if sec else DIM)
        if sec:
            cell.get_text().set_fontweight("bold")
    else:
        # Data cells
        sec = is_section[ri - 1]
        cell.set_facecolor(SECTION if sec else AX_BG)
        cell.get_text().set_color(CYAN if sec else TEXT)
        if sec:
            cell.get_text().set_fontweight("bold")

fig2.tight_layout(rect=[0, 0, 1, 0.97])

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

from alphaforge.portfolio.generator.combo_result import CombinationResult, METHODS
from alphaforge.metrics.metrics import compute_metrics as _cm

# Build a CombinationResult from the 4 method results for this single combo
_portfolios_dict = {method: None for method in METHODS}
for vp in valid_portfolios:
    _portfolios_dict[vp.method] = vp

_eq_vp = _portfolios_dict.get("equal") or next((v for v in _portfolios_dict.values() if v), None)
_raw_rdd = _eq_vp.metrics.get("return_dd_ratio", 0.0) if _eq_vp else 0.0
_raw_metrics = _eq_vp.metrics if _eq_vp else {}

cr = CombinationResult(
    combination   = combo,
    rank          = 1,
    raw_return_dd = _raw_rdd,
    raw_metrics   = _raw_metrics,
    portfolios    = _portfolios_dict,
)

if valid_portfolios:
    print(f"\n  Launching Portfolio Explorer...")
    run_portfolio_dashboard([cr], config, strategies=strategies)
else:
    print("\n  No valid portfolios — skipping dashboard.")
