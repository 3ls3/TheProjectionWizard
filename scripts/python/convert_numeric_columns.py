#!/usr/bin/env python3
"""Convert numeric columns in `berlin_flats_mock.csv` to integers.

The script reads the CSV located in the same directory, converts every
numeric column to an integer type **except** for `size_sqm` (the only
continuous/true float feature), and writes the cleaned data back to the
same folder as `berlin_flats_mock_ints.csv`.

Running the script is idempotent – executing it multiple times will not
alter already-converted columns beyond the first run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def cast_numeric_columns_to_int(df: pd.DataFrame, skip: set[str] | None = None) -> pd.DataFrame:
    """Return *df* where all numeric columns (excluding *skip*) are cast to Int64.

    We use pandas' nullable integer (Int64) so missing values (NaN) stay
    intact while dropping redundant decimal zeros for actual numbers.
    """
    if skip is None:
        skip = set()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols_to_int = [c for c in numeric_cols if c not in skip]

    for col in cols_to_int:
        # Convert to numeric to coerce non-numeric leftovers to NaN, then round
        # (defensive) and cast to pandas' nullable Int64.
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    return df


def main(path: Path | None = None) -> None:
    """Entry-point for CLI execution."""
    if path is None:
        # Default: CSV sits next to this script
        path = Path(__file__).with_name("berlin_flats_mock.csv")

    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    df = pd.read_csv(path)
    df = cast_numeric_columns_to_int(df, skip={"size_sqm"})

    output_path = path.with_stem(path.stem + "_ints")  # berlin_flats_mock_ints.csv
    df.to_csv(output_path, index=False)
    print(f"Wrote cleaned data to {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    # Optional CLI: allow passing a custom path e.g. python convert_numeric_columns.py custom.csv
    custom_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(custom_path) 