#!/usr/bin/env python3
"""
Create qrels (query relevance) datasets for train and test splits.

This script generates qrels parquet files with the following columns:
- query-id: Integer identifier for the query (1-30)
- corpus-id: Integer identifier for the corpus document (0-9)
- score: Binary relevance score (1 for relevant)
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def create_qrels_records(relevance_indices: List[List[int]]) -> List[Dict[str, Any]]:
    """
    Create qrels records from relevance indices.

    Args:
        relevance_indices: List of lists where each sublist contains corpus-ids
                          relevant to the corresponding query (query-id = index + 1)

    Returns:
        List of dictionaries with query-id, corpus-id, and score
    """
    records = []

    for query_idx, corpus_ids in enumerate(relevance_indices):
        query_id = query_idx + 1  # Query IDs start from 1

        for corpus_id in corpus_ids:
            records.append({"query-id": query_id, "corpus-id": corpus_id, "score": 1})

    return records


def save_qrels_parquet(relevance_indices: List[List[int]], output_file: str):
    """
    Save qrels to a parquet file.

    Args:
        relevance_indices: List of lists containing corpus-ids for each query
        output_file: Path to output parquet file
    """
    records = create_qrels_records(relevance_indices)

    if not records:
        print(f"Warning: No qrels records created for {output_file}")
        return

    df = pd.DataFrame(records)

    print(f"\nQrels statistics for {Path(output_file).name}:")
    print(f"  Total relevance judgments: {len(df)}")
    print(f"  Unique queries: {df['query-id'].nunique()}")
    print(f"  Unique corpus docs: {df['corpus-id'].nunique()}")
    print(f"  Avg relevant docs per query: {len(df) / df['query-id'].nunique():.2f}")

    # Save to parquet
    df.to_parquet(output_file, index=False)
    print(f"  Saved to: {output_file}")

    # Show sample
    print(f"\nFirst few rows:")
    print(df.head(10))


def main():
    """Main entry point for the script."""

    # Train split relevance indices (only paper)
    train_relevance_indices = [
        [0, 1, 2, 3, 4, 6, 8, 9],  # 1
        [0, 4, 6],  # 2
        [0],  # 3
        [9],  # 4
        [1],  # 5
        [0, 1, 6],  # 6
        [0, 1, 2, 4, 6],  # 7
        [2],  # 8
        [0, 2, 4],  # 9
        [3],  # 10
        [3],  # 11
        [3],  # 12
        [4, 6],  # 13
        [4],  # 14
        [4],  # 15
        [5],  # 16
        [5],  # 17
        [5],  # 18
        [0, 1, 2, 4, 6],  # 19
        [2, 6],  # 20
        [6],  # 21
        [0, 4, 5, 6, 7, 8],  # 22
        [7],  # 23
        [7],  # 24
        [8],  # 25
        [8],  # 26
        [8],  # 27
        [9],  # 28
        [9],  # 29
        [9],  # 30
    ]

    # Test split relevance indices (similarity aware)
    test_relevance_indices = [
        [0, 4, 6, 7, 8, 9],  # 1
        [1, 2, 4, 6, 7],  # 2
        [0],  # 3
        [0, 1, 2, 4, 6, 7, 9],  # 4
        [1, 6, 7],  # 5
        [1, 6],  # 6
        [1, 2, 3, 4, 6, 7, 9],  # 7
        [2, 8],  # 8
        [2, 5],  # 9
        [3],  # 10
        [3, 4, 9],  # 11
        [3],  # 12
        [4],  # 13 (corrected from [])
        [4, 9],  # 14
        [4, 9],  # 15
        [5, 6],  # 16
        [5],  # 17
        [5],  # 18
        [0, 6],  # 19
        [3, 4, 5, 6, 7, 8, 9],  # 20
        [6],  # 21
        [4, 6, 7, 8],  # 22
        [7],  # 23
        [2, 7, 8],  # 24
        [7, 8],  # 25
        [4, 5, 8],  # 26
        [8],  # 27
        [9],  # 28
        [9],  # 29
        [9],  # 30
    ]

    # Create output directory if it doesn't exist
    output_dir = Path(Path(__file__).parent, "parquet")
    output_dir.mkdir(exist_ok=True)

    # Generate and save train qrels
    print("=" * 60)
    print("Creating train qrels...")
    print("=" * 60)
    train_output = Path(output_dir, "train_qrels.parquet")
    save_qrels_parquet(train_relevance_indices, str(train_output))

    # Generate and save test qrels
    print("\n" + "=" * 60)
    print("Creating test qrels...")
    print("=" * 60)
    test_output = Path(output_dir, "test_qrels.parquet")
    save_qrels_parquet(test_relevance_indices, str(test_output))

    print("\n" + "=" * 60)
    print("Done! Created both qrels files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
