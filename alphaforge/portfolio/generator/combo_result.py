"""
combo_result.py — CombinationResult dataclass.

Groups all 4 weighting-method results for a single strategy combination
produced by the portfolio generation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


METHODS = ("equal", "min_variance", "risk_parity", "hrp")

METHOD_COLORS = {
    "equal":        "#00d4ff",
    "min_variance": "#f0a500",
    "risk_parity":  "#2ecc71",
    "hrp":          "#b267e6",
}

METHOD_LABELS = {
    "equal":        "Equal",
    "min_variance": "Min Var",
    "risk_parity":  "Risk Par",
    "hrp":          "HRP",
}


@dataclass
class CombinationResult:
    """
    All 4 weighting-method portfolios for one strategy combination.

    `portfolios` maps method name → ValidPortfolio (or None if the
    portfolio failed total-DD validation after scaling).

    `raw_metrics` and `raw_return_dd` come from the equal-weight scaled
    portfolio used to rank combinations before applying the other methods.
    `fitness_score` is the normalised composite score used for ranking
    (return/DD × w1 + annual return × w2 + winning months × w3).
    """
    combination   : tuple[str, ...]
    rank          : int                  # 1-based rank by fitness score
    raw_return_dd : float                # equal-weight scaled return/DD (kept for display)
    raw_metrics   : dict                 # full metrics of equal-weight scaled portfolio
    portfolios    : dict                 # method -> ValidPortfolio | None
    fitness_score : float = 0.0         # composite normalised fitness score

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def short_names(self) -> list[str]:
        return [n.split("/")[-1] for n in self.combination]

    @property
    def combo_label(self) -> str:
        return " x ".join(self.short_names)

    @property
    def n_valid_methods(self) -> int:
        """Number of methods that passed the total-DD validation."""
        return sum(1 for vp in self.portfolios.values()
                   if vp is not None and not vp.dd_failed)

    def best_valid(self):
        """Return (method, ValidPortfolio) with highest Sharpe among DD-passed methods."""
        valid = [(m, vp) for m, vp in self.portfolios.items()
                 if vp is not None and not vp.dd_failed]
        if not valid:
            # Fall back to all methods if none passed DD
            valid = [(m, vp) for m, vp in self.portfolios.items() if vp is not None]
        if not valid:
            return None
        return max(valid, key=lambda x: x[1].sharpe)
