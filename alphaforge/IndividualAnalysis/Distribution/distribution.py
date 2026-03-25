"""
distribution.py — Trade PnL distribution visualisation helpers.

Available plots:
  - plot_distribution()          Histogram + KDE + normal fit + CI + VaR
  - plot_win_loss_distribution() Overlaid KDEs for winning vs losing trades
  - plot_cdf()                   Empirical CDF vs normal CDF
  - plot_rolling_distribution()  KDE per chronological period (drift detection)
  - plot_trades_distribution()   Wrapper: distribution from a trades DataFrame
  - plot_trades_win_loss()       Wrapper: win/loss from a trades DataFrame
  - plot_trades_cdf()            Wrapper: CDF from a trades DataFrame
  - plot_all()                   All four in a single 2x2 figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── Shared style ───────────────────────────────────────────────────────────────

_BG_OUTER  = "#1a1a2e"
_BG_INNER  = "#16213e"
_BG_BOX    = "#0f3460"
_ACCENT    = "#4a9eda"
_KDE_COLOR = "#00d4ff"
_NORMAL    = "#f0a500"
_CI_COLOR  = "#ff4c6a"
_MEAN_CLR  = "#aaffaa"
_GRID      = "#2a2a4a"
_SPINE     = "#3a3a5c"

PERIOD_COLORS = ["#00d4ff", "#f0a500", "#2ecc71", "#e056fd", "#ff6b6b", "#ffd32a"]


def _style_ax(ax):
    ax.set_facecolor(_BG_INNER)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.6, zorder=0)


def _legend(ax):
    ax.legend(facecolor=_BG_BOX, edgecolor=_ACCENT, labelcolor="white", fontsize=9)


def _dollar_fmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))


def _stats_box(ax, text):
    ax.text(
        0.98, 0.97, text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=_BG_BOX, alpha=0.85, edgecolor=_ACCENT),
    )


def _make_fig_ax(figsize):
    """Create a standalone figure and styled axis."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_BG_OUTER)
    _style_ax(ax)
    return fig, ax


def _finalize(fig, ax, title, xlabel, ylabel, show, save_path, standalone):
    ax.set_title(title, color="white", fontsize=13 if standalone else 11, pad=10)
    ax.set_xlabel(xlabel, color="#cccccc", fontsize=10)
    ax.set_ylabel(ylabel, color="#cccccc", fontsize=10)
    _dollar_fmt(ax)
    _legend(ax)
    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        elif show:
            plt.show()


# ── Distribution ───────────────────────────────────────────────────────────────

def plot_distribution(
    data: pd.Series,
    *,
    title: str = "Trade PnL Distribution",
    xlabel: str = "Trade PnL ($)",
    bins: int = 60,
    figsize: tuple = (12, 6),
    ax: plt.Axes = None,
    show: bool = True,
    save_path: str = None,
) -> None:
    """
    Histogram (green/red bars) + KDE + semi-transparent normal fit +
    95% CI lines + VaR 99% + skewness/kurtosis flags.
    """
    values = data.dropna().values
    mu, sigma       = values.mean(), values.std()
    ci_low, ci_high = np.percentile(values, [2.5, 97.5])
    var_99          = np.percentile(values, 1)
    skewness        = stats.skew(values)
    kurt            = stats.kurtosis(values)

    standalone = ax is None
    if standalone:
        fig, ax = _make_fig_ax(figsize)
    else:
        _style_ax(ax)

    x_min   = min(values.min(), mu - 4 * sigma)
    x_max   = max(values.max(), mu + 4 * sigma)
    x_range = np.linspace(x_min, x_max, 500)

    counts, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_colors  = ["#2ecc71" if c >= 0 else "#e74c3c" for c in bin_centers]
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color=bar_colors,
           alpha=0.6, edgecolor=_BG_OUTER, linewidth=0.3, align="edge",
           label="Histogram", zorder=2)

    ax.axvline(0, color="black", linewidth=1.8, zorder=5)

    kde = stats.gaussian_kde(values)
    ax.plot(x_range, kde(x_range), color=_KDE_COLOR, linewidth=2.0, label="KDE", zorder=4)

    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    ax.fill_between(x_range, normal_pdf, alpha=0.18, color=_NORMAL, zorder=3)
    ax.plot(x_range, normal_pdf, color=_NORMAL, linewidth=1.5,
            linestyle="--", label="Normal fit", zorder=4)

    for val, label, side in [
        (ci_low,  f"2.5% \${ci_low:,.0f}",  "left"),
        (ci_high, f"97.5% \${ci_high:,.0f}", "right"),
    ]:
        ax.axvline(val, color=_CI_COLOR, linewidth=1.4, linestyle="--", zorder=5)
        ha, offset = ("right", -4) if side == "left" else ("left", 4)
        ax.annotate(label, xy=(val, 0), xytext=(offset, 120),
                    textcoords="offset points", color=_CI_COLOR, fontsize=8, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=_CI_COLOR, lw=0.8), zorder=6)

    ax.axvline(var_99, color="#ff9f43", linewidth=1.4, linestyle=":", zorder=5)
    ax.annotate(f"VaR 99% \${var_99:,.0f}", xy=(var_99, 0), xytext=(-6, 90),
                textcoords="offset points", color="#ff9f43", fontsize=8, ha="right",
                arrowprops=dict(arrowstyle="-", color="#ff9f43", lw=0.8), zorder=6)

    ax.axvline(mu, color=_MEAN_CLR, linewidth=1.3, linestyle=":", zorder=5)
    ax.annotate(f"Mean \${mu:,.2f}", xy=(mu, 0), xytext=(6, 150),
                textcoords="offset points", color=_MEAN_CLR, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=_MEAN_CLR, lw=0.8), zorder=6)

    skew_flag = "  ⚠" if abs(skewness) > 0.5 else ""
    kurt_flag = "  ⚠" if abs(kurt) > 1.0    else ""
    _stats_box(ax, (
        f"μ = \${mu:,.2f}\n"
        f"σ = \${sigma:,.2f}\n"
        f"VaR 99% = \${var_99:,.2f}\n"
        f"Skew = {skewness:.3f}{skew_flag}\n"
        f"Kurt = {kurt:.3f}{kurt_flag}"
    ))

    _finalize(fig if standalone else None, ax, title,
              xlabel, "Density", show, save_path, standalone)


# ── Win / Loss ─────────────────────────────────────────────────────────────────

def plot_win_loss_distribution(
    data: pd.Series,
    *,
    title: str = "Win vs Loss Distribution",
    figsize: tuple = (12, 6),
    ax: plt.Axes = None,
    show: bool = True,
    save_path: str = None,
) -> None:
    """Overlaid KDEs for winning and losing trades."""
    values = data.dropna().values
    wins   = values[values > 0]
    losses = values[values < 0]

    standalone = ax is None
    if standalone:
        fig, ax = _make_fig_ax(figsize)
    else:
        _style_ax(ax)

    x_range = np.linspace(values.min() * 1.1, values.max() * 1.1, 500)

    for subset, label, color in [
        (wins,   "Wins",   "#2ecc71"),
        (losses, "Losses", "#e74c3c"),
    ]:
        if len(subset) < 2:
            continue
        kde = stats.gaussian_kde(subset)
        y   = kde(x_range)
        ax.fill_between(x_range, y, alpha=0.25, color=color, zorder=2)
        ax.plot(x_range, y, color=color, linewidth=2.0, label=label, zorder=3)
        ax.axvline(subset.mean(), color=color, linewidth=1.2, linestyle="--", alpha=0.8, zorder=4)

    ax.axvline(0, color="black", linewidth=1.8, zorder=5)

    _stats_box(ax, (
        f"Wins   n={len(wins)}   μ=\${wins.mean():,.2f}   σ=\${wins.std():,.2f}\n"
        f"Losses n={len(losses)}   μ=\${losses.mean():,.2f}   σ=\${losses.std():,.2f}\n"
        f"Ratio  {len(wins)/len(losses):.2f}   Payoff  {wins.mean()/abs(losses.mean()):.2f}"
    ))

    _finalize(fig if standalone else None, ax, title,
              "Trade PnL ($)", "Density", show, save_path, standalone)


# ── CDF ───────────────────────────────────────────────────────────────────────

def plot_cdf(
    data: pd.Series,
    *,
    title: str = "Cumulative Distribution Function",
    figsize: tuple = (12, 6),
    ax: plt.Axes = None,
    show: bool = True,
    save_path: str = None,
) -> None:
    """Empirical CDF vs normal CDF with CI and VaR 99% markers."""
    values   = np.sort(data.dropna().values)
    mu, sigma = values.mean(), values.std()
    ecdf     = np.arange(1, len(values) + 1) / len(values)
    x_range  = np.linspace(values.min(), values.max(), 500)

    standalone = ax is None
    if standalone:
        fig, ax = _make_fig_ax(figsize)
    else:
        _style_ax(ax)

    ax.step(values, ecdf, color=_KDE_COLOR, linewidth=2.0, label="Empirical CDF", zorder=4)
    ax.plot(x_range, stats.norm.cdf(x_range, mu, sigma), color=_NORMAL,
            linewidth=1.5, linestyle="--", alpha=0.8, label="Normal CDF", zorder=3)

    for pct, color, lbl in [
        (0.025, _CI_COLOR,  "2.5%"),
        (0.975, _CI_COLOR,  "97.5%"),
        (0.01,  "#ff9f43",  "VaR 99%"),
    ]:
        xval = np.percentile(values, pct * 100)
        ax.axvline(xval, color=color, linewidth=1.2, linestyle=":", zorder=5)
        ax.axhline(pct,  color=color, linewidth=0.7, linestyle=":", alpha=0.5, zorder=5)
        ax.annotate(f"{lbl}\n\${xval:,.0f}", xy=(xval, pct),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=8, va="center")

    ax.axvline(0, color="black", linewidth=1.8, zorder=5)
    ax.set_ylim(0, 1)

    _finalize(fig if standalone else None, ax, title,
              "Trade PnL ($)", "Cumulative Probability", show, save_path, standalone)


# ── Rolling distribution ───────────────────────────────────────────────────────

def plot_rolling_distribution(
    df: pd.DataFrame,
    *,
    n_periods: int = 4,
    title: str = None,
    figsize: tuple = (12, 6),
    ax: plt.Axes = None,
    show: bool = True,
    save_path: str = None,
) -> None:
    """KDE per chronological period — reveals drift or strategy decay."""
    sorted_df = df.sort_values("Close time").reset_index(drop=True)
    chunks    = np.array_split(sorted_df, n_periods)

    if title is None:
        name  = df["strategy"].iloc[0] if "strategy" in df.columns else "Strategy"
        title = f"Rolling Distribution ({n_periods} periods) — {name}"

    standalone = ax is None
    if standalone:
        fig, ax = _make_fig_ax(figsize)
    else:
        _style_ax(ax)

    all_pnl = sorted_df["Profit/Loss"].dropna().values
    x_range = np.linspace(all_pnl.min() * 1.1, all_pnl.max() * 1.1, 500)

    for i, chunk in enumerate(chunks):
        pnl   = chunk["Profit/Loss"].dropna().values
        if len(pnl) < 2:
            continue
        color = PERIOD_COLORS[i % len(PERIOD_COLORS)]
        start = chunk["Close time"].min().strftime("%Y-%m")
        end   = chunk["Close time"].max().strftime("%Y-%m")
        label = (
            f"P{i+1}  {start} \u2192 {end}"
            f"   {len(pnl)} trades"
            f"   Net: \${pnl.sum():,.2f}"
            f"   Avg: \${pnl.mean():,.2f}"
        )

        kde = stats.gaussian_kde(pnl)
        y   = kde(x_range)
        ax.fill_between(x_range, y, alpha=0.15, color=color, zorder=2)
        ax.plot(x_range, y, color=color, linewidth=2.0, label=label, zorder=3)
        ax.axvline(pnl.mean(), color=color, linewidth=1.1, linestyle="--", alpha=0.7, zorder=4)

    ax.axvline(0, color="black", linewidth=1.8, zorder=5)

    _finalize(fig if standalone else None, ax, title,
              "Trade PnL ($)", "Density", show, save_path, standalone)


# ── Convenience wrappers ───────────────────────────────────────────────────────

def _strategy_name(df):
    return df["strategy"].iloc[0] if "strategy" in df.columns else "Strategy"


def _symbol(df):
    """Extract clean symbol (e.g. 'USDJPY') from the Symbol column."""
    if "Symbol" not in df.columns:
        return None
    raw = str(df["Symbol"].iloc[0])
    return raw.split("_")[0] if "_" in raw else raw


def plot_trades_distribution(df, *, title=None, bins=60, figsize=(12, 6),
                              ax=None, show=True, save_path=None):
    """Main distribution plot from a trades DataFrame."""
    pnl = df["Profit/Loss"].dropna()
    plot_distribution(pnl, title=title or f"Trade PnL Distribution — {_strategy_name(df)}",
                      bins=bins, figsize=figsize, ax=ax, show=show, save_path=save_path)


def plot_trades_win_loss(df, *, title=None, figsize=(12, 6),
                         ax=None, show=True, save_path=None):
    """Win vs Loss KDE overlay from a trades DataFrame."""
    pnl = df["Profit/Loss"].dropna()
    plot_win_loss_distribution(pnl, title=title or f"Win vs Loss — {_strategy_name(df)}",
                               figsize=figsize, ax=ax, show=show, save_path=save_path)


def plot_trades_cdf(df, *, title=None, figsize=(12, 6),
                    ax=None, show=True, save_path=None):
    """CDF plot from a trades DataFrame (standalone)."""
    pnl = df["Profit/Loss"].dropna()
    plot_cdf(pnl, title=title or f"CDF — {_strategy_name(df)}",
             figsize=figsize, ax=ax, show=show, save_path=save_path)


def plot_trades_mae_mfe(df, *, title=None, figsize=(12, 6),
                        ax=None, show=True, save_path=None):
    """MAE / MFE distribution from a trades DataFrame."""
    plot_mae_mfe(df, title=title, figsize=figsize, ax=ax, show=show, save_path=save_path)


# ── MAE / MFE ─────────────────────────────────────────────────────────────────

def plot_mae_mfe(
    df: pd.DataFrame,
    *,
    title: str = None,
    figsize: tuple = (12, 6),
    ax: plt.Axes = None,
    show: bool = True,
    save_path: str = None,
) -> None:
    """
    Overlaid KDEs for MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable
    Excursion). MAE shows how much heat trades took; MFE shows how much profit
    was available. The efficiency ratio (avg PnL / avg MFE) measures how much
    of the available profit the strategy actually captured.
    """
    mae_col = next((c for c in df.columns if "MAE" in c), None)
    mfe_col = next((c for c in df.columns if "MFE" in c), None)
    if mae_col is None or mfe_col is None:
        raise ValueError("DataFrame must contain MAE and MFE columns.")

    mae = df[mae_col].dropna().values          # negative values
    mfe = df[mfe_col].dropna().values          # positive values
    pnl = df["Profit/Loss"].dropna().values

    avg_mae      = mae.mean()
    avg_mfe      = mfe.mean()
    avg_pnl      = pnl.mean()
    efficiency   = (avg_pnl / avg_mfe * 100) if avg_mfe != 0 else float("nan")
    mae_captured = (avg_pnl / abs(avg_mae) * 100) if avg_mae != 0 else float("nan")

    standalone = ax is None
    if standalone:
        fig, ax = _make_fig_ax(figsize)
    else:
        _style_ax(ax)

    all_vals = np.concatenate([mae, mfe])
    x_range  = np.linspace(all_vals.min() * 1.1, all_vals.max() * 1.1, 500)

    for values, label, color in [
        (mae, "MAE (adverse)",   "#e74c3c"),
        (mfe, "MFE (favorable)", "#2ecc71"),
    ]:
        if len(values) < 2:
            continue
        kde = stats.gaussian_kde(values)
        y   = kde(x_range)
        ax.fill_between(x_range, y, alpha=0.2, color=color, zorder=2)
        ax.plot(x_range, y, color=color, linewidth=2.0, label=label, zorder=3)
        ax.axvline(values.mean(), color=color, linewidth=1.2,
                   linestyle="--", alpha=0.8, zorder=4)

    # Avg PnL marker
    ax.axvline(avg_pnl, color=_MEAN_CLR, linewidth=1.3, linestyle=":", zorder=5)
    ax.annotate(f"Avg PnL \${avg_pnl:,.2f}", xy=(avg_pnl, 0), xytext=(6, 100),
                textcoords="offset points", color=_MEAN_CLR, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=_MEAN_CLR, lw=0.8), zorder=6)

    ax.axvline(0, color="black", linewidth=1.8, zorder=5)

    _stats_box(ax, (
        f"Avg MAE  = \${avg_mae:,.2f}\n"
        f"Avg MFE  = \${avg_mfe:,.2f}\n"
        f"Avg PnL  = \${avg_pnl:,.2f}\n"
        f"Efficiency (PnL/MFE) = {efficiency:.1f}%\n"
        f"PnL/MAE ratio        = {mae_captured:.1f}%"
    ))

    if title is None:
        name  = df["strategy"].iloc[0] if "strategy" in df.columns else "Strategy"
        title = f"MAE / MFE Distribution — {name}"

    _finalize(fig if standalone else None, ax, title,
              "Excursion ($)", "Density", show, save_path, standalone)


# ── Rolling breakdown ─────────────────────────────────────────────────────────

def plot_rolling_breakdown(
    df: pd.DataFrame,
    *,
    n_periods: int = 4,
    figsize: tuple = None,
    show: bool = True,
    save_path: str = None,
) -> None:
    """
    One full trade distribution subplot per chronological period.
    Reveals how the shape of the return distribution evolves over time.

    Grid is sized automatically: 2 cols for <= 4 periods, 3 cols for more.
    """
    import math

    sorted_df  = df.sort_values("Close time").reset_index(drop=True)
    chunks     = np.array_split(sorted_df, n_periods)
    name       = _strategy_name(df)
    symbol     = _symbol(df)
    title_str  = (
        f"Rolling Distribution Breakdown ({n_periods} periods) — {name}  |  {symbol}"
        if symbol else
        f"Rolling Distribution Breakdown ({n_periods} periods) — {name}"
    )

    ncols = 2 if n_periods <= 4 else 3
    nrows = math.ceil(n_periods / ncols)

    if figsize is None:
        figsize = (ncols * 10, nrows * 5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(_BG_OUTER)
    fig.suptitle(title_str, color="white", fontsize=14, y=0.98)

    # Flatten axes for easy iteration, handle single-row case
    axes_flat = axes.flatten() if n_periods > 1 else [axes]

    for i, (chunk, ax) in enumerate(zip(chunks, axes_flat)):
        pnl     = chunk["Profit/Loss"].dropna()
        color   = PERIOD_COLORS[i % len(PERIOD_COLORS)]
        start   = chunk["Close time"].min().strftime("%Y-%m")
        end     = chunk["Close time"].max().strftime("%Y-%m")
        net_pnl  = pnl.sum()
        win_pct  = (pnl > 0).mean() * 100
        equity   = pnl.cumsum()
        max_dd   = (equity.cummax() - equity).max()
        period_title = (
            f"Period {i+1}   |   {start} \u2192 {end}   |   {len(pnl)} trades\n"
            f"Net: \${net_pnl:,.2f}     Win Rate: {win_pct:.1f}%     Max DD: \${max_dd:,.2f}"
        )

        if len(pnl) < 2:
            ax.set_visible(False)
            continue

        plot_distribution(
            pnl,
            title=period_title,
            ax=ax,
            show=False,
        )
        # Tint the title with the period colour for quick visual identification
        ax.set_title(period_title, color=color, fontsize=11, pad=8)

    # Hide any unused subplots
    for ax in axes_flat[n_periods:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    elif show:
        plt.show()


# ── plot_all ──────────────────────────────────────────────────────────────────

def plot_all(
    df: pd.DataFrame,
    *,
    figsize: tuple = (20, 12),
    show: bool = True,
    save_path: str = None,
) -> None:
    """
    All four distribution plots in a single 2x2 figure:
    Trade PnL Distribution | Win vs Loss
    Rolling Distribution   | MAE / MFE
    """
    name   = _strategy_name(df)
    symbol = _symbol(df)
    title_str = f"Distribution Analysis — {name}  |  {symbol}" if symbol else f"Distribution Analysis — {name}"
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor(_BG_OUTER)
    fig.suptitle(title_str, color="white", fontsize=15, y=0.98)

    plot_trades_distribution(df,  ax=axes[0, 0])
    plot_trades_win_loss(df,      ax=axes[0, 1])
    plot_rolling_distribution(df, ax=axes[1, 0])
    plot_mae_mfe(df,              ax=axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    elif show:
        plt.show()
