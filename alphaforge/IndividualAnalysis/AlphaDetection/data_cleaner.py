"""
data_cleaner.py — Load and cache cleaned D1 price data for AlphaDetection.

Logic:
  1. Check if a clean file ({pair}_D1.csv) exists in the data folder with
     the required columns (Open, High, Low, Close, DayOfWeek, WeekendGap...).
  2. If yes  → load and return immediately (no reprocessing).
  3. If no   → find the raw OHLCV file, add derived columns, save the clean
               file, and return the result.

Usage:
    from alphaforge.IndividualAnalysis.AlphaDetection.data_cleaner import load_price_data

    price_df = load_price_data("AUDJPY")
    price_df = load_price_data("USDJPY", data_folder="AssetsData")
"""

from pathlib import Path

import pandas as pd

# Columns that must be present for the file to be considered "clean"
_REQUIRED_COLS = {"Open", "High", "Low", "Close"}
_DERIVED_COLS  = {"DayOfWeek", "DayName", "WeekendGap", "WeekendGapPct"}


# ── Public entry point ─────────────────────────────────────────────────────────

def load_price_data(pair: str, data_folder: str = "AssetsData") -> pd.DataFrame:
    """
    Load cleaned D1 price data for a currency pair.

    Checks for a cached clean file first. If the file is missing or lacks
    the derived columns, regenerates and saves it before returning.

    Args:
        pair:        Instrument identifier, e.g. "AUDJPY" or "USDJPY".
        data_folder: Folder containing the price CSV files (default "AssetsData").

    Returns:
        pd.DataFrame indexed by date with columns:
            Open, High, Low, Close, Volume (if available),
            DayOfWeek, DayName, WeekendGap, WeekendGapPct

    Raises:
        FileNotFoundError: if no price file can be found for the pair.
    """
    folder     = Path(data_folder)
    clean_path = folder / f"{pair}_D1.csv"

    # ── Step 1: try loading the existing clean file ────────────────────────────
    if clean_path.exists():
        try:
            df = pd.read_csv(clean_path, index_col="Date", parse_dates=True)
            if _is_valid(df):
                # Already has all derived columns — return immediately
                if _DERIVED_COLS.issubset(df.columns):
                    return df
                # Has OHLCV but missing derived columns — add them and resave
                df = _add_derived_columns(df)
                df.to_csv(clean_path)
                print(f"[data_cleaner] Added derived columns to {clean_path.name}")
                return df
        except Exception as e:
            print(f"[data_cleaner] Warning: could not read {clean_path.name} ({e}). "
                  f"Searching for raw file...")

    # ── Step 2: no clean file — look for any raw file containing the pair name ─
    candidates = list(folder.glob(f"*{pair}*.csv")) + list(folder.glob(f"*{pair}*.xlsx"))
    candidates = [p for p in candidates if "D1" in p.name.upper()]

    if not candidates:
        raise FileNotFoundError(
            f"No price file found for '{pair}' in '{folder}'. "
            f"Expected '{pair}_D1.csv' or a raw file containing '{pair}' and 'D1' "
            f"in its name."
        )

    raw_path = candidates[0]
    print(f"[data_cleaner] Raw file found: {raw_path.name} — cleaning...")

    # ── Step 3: load raw file ──────────────────────────────────────────────────
    if raw_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(raw_path, index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    df.index.name = "Date"
    df.index      = pd.to_datetime(df.index).normalize()

    # Standardise column names (capitalise first letter)
    df.columns = [c.strip().capitalize() for c in df.columns]

    if not _is_valid(df):
        raise ValueError(
            f"Raw file '{raw_path.name}' is missing required columns "
            f"{_REQUIRED_COLS}. Found: {set(df.columns)}"
        )

    # ── Step 4: add derived columns and save clean file ───────────────────────
    df = _add_derived_columns(df)
    df.to_csv(clean_path)
    print(f"[data_cleaner] Clean file saved: {clean_path.name} ({len(df):,} rows)")

    return df


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_valid(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame has at minimum the OHLC columns."""
    return _REQUIRED_COLS.issubset(df.columns)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add DayOfWeek, DayName, WeekendGap, WeekendGapPct to an OHLCV DataFrame."""
    df = df.copy()
    df["DayOfWeek"] = df.index.dayofweek          # 0=Mon, 4=Fri
    df["DayName"]   = df.index.day_name()

    # Weekend gap: Monday Open minus previous Friday Close
    prev_close = df["Close"].shift(1)
    is_monday  = df["DayOfWeek"] == 0

    df["WeekendGap"]    = None
    df["WeekendGapPct"] = None

    df.loc[is_monday, "WeekendGap"] = (
        df.loc[is_monday, "Open"] - prev_close[is_monday]
    ).round(5)

    df.loc[is_monday, "WeekendGapPct"] = (
        (df.loc[is_monday, "Open"] - prev_close[is_monday])
        / prev_close[is_monday] * 100
    ).round(5)

    return df
