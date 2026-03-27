"""
loader.py — Reads, loads, and filters trading strategy CSV files.

Each CSV is semicolon-separated and may contain the following columns:
    Ticket, Symbol, Type, Open time, Open price, Size,
    Close time, Close price, Profit/Loss, Close type,
    MAE (), MFE(), MFE (), Time in trade,
    Balance, Sample type, Comment  ← dropped

Returns one clean DataFrame per strategy, and a combined dict for the full run.
"""

import os
import pandas as pd

# Columns to remove — not useful for analysis
COLUMNS_TO_DROP = ["Balance", "Sample type", "Comment", "Time in trade", "Ticket"]

# Expected columns after dropping — used for validation
EXPECTED_COLUMNS = {
    "Symbol",
    "Type",
    "Open time",
    "Open price",
    "Size",
    "Close time",
    "Close price",
    "Profit/Loss",
    "Close type",
}

# Columns that should be numeric
NUMERIC_COLUMNS = ["Open price", "Close price", "Size", "Profit/Loss"]

# Columns that contain date/time strings
DATETIME_COLUMNS = ["Open time", "Close time"]


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names to handle minor formatting variations."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _drop_ignored_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are not needed for analysis."""
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_present)


def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are entirely empty or contain only whitespace."""
    return df.dropna(how="all").reset_index(drop=True)


def _drop_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    MT4/MT5 exports often include summary/footer rows at the bottom.
    Keep only rows where Profit/Loss is a valid number.
    """
    if "Profit/Loss" not in df.columns:
        return df
    numeric = pd.to_numeric(
        df["Profit/Loss"].astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    mask = numeric.notna()
    removed = (~mask).sum()
    if removed > 0:
        print(f"    Dropped {removed} non-trade row(s) (summaries/headers).")
    return df[mask].reset_index(drop=True)


def _parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Open time and Close time to proper datetime objects."""
    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].astype(str).str.strip(),
                format="%Y.%m.%d %H:%M:%S",
                errors="coerce",
            )
    return df


def _parse_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse numeric columns to float.
    Handles both dot and comma as decimal separators.
    """
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)  # EU decimal comma → dot
                .str.replace(" ", "", regex=False)   # remove thousands spaces
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_mae_mfe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse any MAE/MFE columns (they may have varying names due to export quirks).
    Converts them to float if present.
    """
    mae_mfe_cols = [c for c in df.columns if "MAE" in c or "MFE" in c]
    for col in mae_mfe_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _validate_columns(df: pd.DataFrame, name: str) -> None:
    """Warn if any expected columns are missing after loading."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        print(f"    Warning [{name}]: missing expected columns: {missing}")


def _compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add a duration_minutes column computed from Open/Close time."""
    if "Open time" in df.columns and "Close time" in df.columns:
        df["duration_minutes"] = (
            df["Close time"] - df["Open time"]
        ).dt.total_seconds() / 60
    return df


def load_strategy(filepath: str) -> pd.DataFrame:
    """
    Load and clean a single strategy CSV file.

    Steps:
      1. Read raw CSV (semicolon-separated, all as strings initially)
      2. Normalize column names
      3. Drop ignored columns
      4. Drop fully empty rows
      5. Drop non-trade summary rows
      6. Parse datetime columns
      7. Parse numeric columns (PnL, prices, size)
      8. Parse MAE/MFE columns
      9. Compute trade duration
      10. Tag each row with the strategy name
      11. Validate expected columns are present

    Returns a clean DataFrame ready for analysis.
    """
    name = os.path.splitext(os.path.basename(filepath))[0]

    # Read everything as strings first to control parsing ourselves
    df = pd.read_csv(filepath, sep=";", dtype=str, encoding="utf-8-sig")

    df = _normalize_column_names(df)
    df = _drop_ignored_columns(df)
    df = _drop_empty_rows(df)
    df = _drop_summary_rows(df)
    df = _parse_datetime_columns(df)
    df = _parse_numeric_columns(df)
    df = _parse_mae_mfe_columns(df)
    df = _compute_duration(df)

    # Tag rows with their source strategy
    df["strategy"] = name

    _validate_columns(df, name)

    return df


def _collect_csv_files(folder: str) -> list[str]:
    """
    Recursively collect all CSV file paths under a folder,
    regardless of subfolder structure or naming.
    """
    found = []
    for root, _dirs, files in os.walk(folder):
        for filename in sorted(files):
            if filename.lower().endswith(".csv"):
                found.append(os.path.join(root, filename))
    return found


def load_folder(folder: str) -> dict[str, pd.DataFrame]:
    """
    Recursively load all CSV files from a folder and its subfolders.

    The strategy name is built from the relative path so that two files
    with the same name in different subfolders don't collide:
        strategies/approved/EURUSD/strategy_a.csv  →  key: "EURUSD/strategy_a"

    Args:
        folder: Root directory to scan (e.g. "strategies/approved/").

    Returns:
        A dict mapping relative_path_name → clean DataFrame.
        Files that fail to load are skipped with a warning.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    csv_files = _collect_csv_files(folder)

    if not csv_files:
        print(f"No CSV files found in: {folder}")
        return {}

    strategies: dict[str, pd.DataFrame] = {}

    print(f"\nScanning '{folder}' — found {len(csv_files)} CSV file(s).\n")

    for filepath in csv_files:
        # Build a readable key: subfolder/filename (no extension)
        rel = os.path.relpath(filepath, folder)
        name = os.path.splitext(rel)[0].replace("\\", "/")
        print(f"  [{name}]")
        try:
            df = load_strategy(filepath)
            # Override the strategy tag with the full relative name
            df["strategy"] = name
            strategies[name] = df
            print(f"    OK — {len(df)} trades loaded.\n")
        except Exception as e:
            print(f"    ERROR: {e}\n")

    print(f"Done. {len(strategies)}/{len(csv_files)} files loaded successfully.")
    return strategies
