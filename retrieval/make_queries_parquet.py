#!/usr/bin/env python3
"""
Convert GPT-generated queries from JSONL format to Parquet dataset.

This script reads a JSONL file containing GPT-generated queries for scientific papers
and converts it to a Parquet dataset with the following columns:
- query_id: Unique integer identifier for each query (starting from 1)
- query: The actual query text
"""

import argparse
import json
import pandas as pd

from pathlib import Path
from typing import List, Dict, Any


def parse_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse a JSONL file containing GPT queries.

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
                # Parse the outer JSON
                entry = json.loads(line)

                # Parse the inner JSON string (gpt_queries)
                gpt_queries = json.loads(entry["gpt_queries"])

                # Extract each query
                for query in gpt_queries["queries"]:
                    records.append({"query": query["text"]})

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

    print(f"Parsed {len(records)} query records")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Add query_id column (starting from 1)
    df.insert(0, "query_id", range(1, len(df) + 1))

    # Display some statistics
    print("\nDataset statistics:")
    print(f"  Total queries: {len(df)}")
    print(f"  Unique queries: {df['query'].nunique()}")

    # Save to Parquet
    print(f"\nSaving to: {output_file}")
    df.to_parquet(output_file, index=False)

    print("\nFirst few rows:")
    print(df.head(10))

    print("\nDataset schema:")
    print(df.dtypes)

    print(f"\nSuccessfully created Parquet file: {output_file}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert GPT queries JSONL file to Parquet dataset"
    )
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Path to output Parquet file (default: input_file.parquet)",
        default="output.parquet",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    # relevance_indices = None  # Set to None to use empty lists for all queries

    # Convert to Parquet
    Path("parquet").mkdir(exist_ok=True)
    out_path = Path("parquet", args.output_file)
    convert_to_parquet(args.input_file, out_path)


if __name__ == "__main__":
    main()
