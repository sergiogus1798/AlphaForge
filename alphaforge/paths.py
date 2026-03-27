"""
paths.py — Central path constants for AlphaForge.

Folder layout:
    strategies/
        raw/        SQX exports that haven't been individually analysed yet.
        approved/   Strategies that passed individual analysis.
                    This is the folder the portfolio constructor reads from.
    portfolios/     Generated portfolios (wiped before every run).
                    One subfolder per combination, one sub-subfolder per method.
"""

import os
import shutil

# ── Strategy folders ────────────────────────────────────────────────────────────

STRATEGIES_RAW      = "strategies/raw"
STRATEGIES_APPROVED = "strategies/approved"

# ── Portfolio output ────────────────────────────────────────────────────────────

PORTFOLIOS_OUTPUT   = "portfolios"


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_portfolios_output(verbose: bool = True) -> None:
    """
    Wipe and recreate the portfolios/ output directory before a new run.
    Prints what is being deleted so nothing is lost silently.
    """
    if not os.path.isdir(PORTFOLIOS_OUTPUT):
        os.makedirs(PORTFOLIOS_OUTPUT, exist_ok=True)
        if verbose:
            print(f"  Created output folder: {PORTFOLIOS_OUTPUT}/")
        return

    entries = os.listdir(PORTFOLIOS_OUTPUT)
    if entries and verbose:
        print(f"  Cleaning {PORTFOLIOS_OUTPUT}/  ({len(entries)} existing item(s) removed)")
        for e in entries:
            print(f"    ✗  {e}")

    shutil.rmtree(PORTFOLIOS_OUTPUT)
    os.makedirs(PORTFOLIOS_OUTPUT)


def portfolio_combination_dir(rank: int) -> str:
    """Return the path for a combination folder, e.g. portfolios/Combination001/"""
    return os.path.join(PORTFOLIOS_OUTPUT, f"Combination{rank:03d}")


def portfolio_method_dir(rank: int, method: str) -> str:
    """
    Return the path for a (combination, method) folder.
    e.g. portfolios/Combination001/HRP/
    """
    labels = {
        "equal":        "Equal",
        "min_variance": "MinVariance",
        "risk_parity":  "RiskParity",
        "hrp":          "HRP",
    }
    return os.path.join(portfolio_combination_dir(rank), labels.get(method, method))


def ensure_method_dir(rank: int, method: str) -> str:
    """Create and return the method output directory."""
    path = portfolio_method_dir(rank, method)
    os.makedirs(path, exist_ok=True)
    return path
