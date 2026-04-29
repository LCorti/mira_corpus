#!/usr/bin/env python3
"""
Convert selected papers from JSONL format to Parquet dataset.

This script reads a JSONL file containing scientific papers and converts it to a
Parquet dataset with the following columns:
- doc_idx: Sequential integer ID starting from 0
- paper_id: Original paper identifier
- doc_title: Paper title
- doc_abs: Paper abstract
- doc_text: Full paper text
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse a JSONL file containing paper data.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of dictionaries containing parsed data
    """
    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse the JSON line
                entry = json.loads(line)

                # Extract required fields
                records.append(
                    {
                        "doc_idx": line_num - 1,  # 0-indexed
                        "paper_id": entry.get("paper_id", ""),
                        "doc_title": entry.get("title", ""),
                        "doc_abs": entry.get("abstract", ""),
                        "doc_text": entry.get("full_text", ""),
                    }
                )

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Warning: Missing key in line {line_num}: {e}")
                continue

    return records


def convert_to_parquet(input_file: str, output_file: str | None = None):
    """
    Convert JSONL file to Parquet format.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output Parquet file (optional)
    """
    # Set default output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix(".parquet"))

    print(f"Reading from: {input_file}")

    # Parse the JSONL file
    records = parse_jsonl_file(input_file)

    if not records:
        print("Error: No records found in the input file")
        return

    print(f"Parsed {len(records)} paper records")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Display some statistics
    print("\nDataset statistics:")
    print(f"  Total papers: {len(df)}")
    print(f"  Doc idx range: {df['doc_idx'].min()} to {df['doc_idx'].max()}")
    print(f"  Unique paper IDs: {df['paper_id'].nunique()}")

    # Check for any missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print("\nMissing values:")
        print(missing_counts[missing_counts > 0])

    # Display text length statistics
    print("\nText length statistics:")
    print(f"  Title - avg: {df['doc_title'].str.len().mean():.0f} chars")
    print(f"  Abstract - avg: {df['doc_abs'].str.len().mean():.0f} chars")
    print(f"  Full text - avg: {df['doc_text'].str.len().mean():.0f} chars")

    # Save to Parquet
    print(f"\nSaving to: {output_file}")
    df.to_parquet(output_file, index=False)

    print("\nFirst few rows (showing only metadata columns):")
    print(df[["doc_idx", "paper_id", "doc_title"]].head(10))

    print("\nDataset schema:")
    print(df.dtypes)

    print(f"\nSuccessfully created Parquet file: {output_file}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert papers JSONL file to Parquet dataset"
    )
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Path to output Parquet file (default: input_file.parquet)",
        default="out_file.parquet",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Convert to Parquet
    Path("parquet").mkdir(exist_ok=True)
    out_path = Path("parquet", args.output_file)
    convert_to_parquet(args.input_file, out_path)


if __name__ == "__main__":
    main()
