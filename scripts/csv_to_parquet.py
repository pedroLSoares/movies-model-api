#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def convert_dir(
    input_dir: Path,
    output_dir: Path | None,
    *,
    overwrite: bool,
    low_memory: bool,
) -> list[Path]:
    """
    Convert every *.csv in input_dir to Parquet.

    If output_dir is None, writes alongside each CSV with the same stem + .parquet.
    Otherwise writes all Parquet files under output_dir (flat; basenames must be unique).
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}", file=sys.stderr)
        return []

    written: list[Path] = []
    out_base = output_dir.resolve() if output_dir else None
    if out_base:
        out_base.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        parquet_name = csv_path.with_suffix(".parquet").name
        dest = (out_base / parquet_name) if out_base else csv_path.with_suffix(".parquet")
        if dest.exists() and not overwrite:
            print(f"Skip (exists): {dest}", file=sys.stderr)
            continue

        df = pd.read_csv(
            csv_path,
            low_memory=low_memory,
            on_bad_lines="skip",
        )
        df.to_parquet(dest, index=False, engine="pyarrow")
        written.append(dest)
        print(f"{csv_path.name} -> {dest}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all CSV files in a folder to Parquet."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .csv files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as each CSV)",
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help="Overwrite existing .parquet files",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Pass low_memory=True to pandas.read_csv (slower, less memory)",
    )
    args = parser.parse_args()

    convert_dir(
        args.input_dir,
        args.output_dir,
        overwrite=args.overwrite,
        low_memory=args.low_memory,
    )


if __name__ == "__main__":
    main()
