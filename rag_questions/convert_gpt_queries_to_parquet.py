#!/usr/bin/env python3
"""
Convert GPT-generated queries from JSONL format to Parquet dataset.

This script reads a JSONL file containing GPT-generated queries for scientific papers
and converts it to a Parquet dataset with the following columns:
- paper_id: Identifier for the paper
- query_type: Type of query (breadth: high_recall, balanced, or high_precision_distinctive)
- query_text: The actual query text
- similarity_scores: Empty list for now (placeholder for future use)
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any


def parse_jsonl_file(filepath: str, relevance_indices: List[List[int]] | None = None) -> List[Dict[str, Any]]:
    """
    Parse a JSONL file containing GPT queries.
    
    Args:
        filepath: Path to the JSONL file
        relevance_indices: Optional list of lists containing relevance indices for each query. Must have the same length as the total number of queries.
        
    Returns:
        List of dictionaries containing parsed data
    """
    records = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the outer JSON
                entry = json.loads(line)
                paper_id = entry['paper_id']
                
                # Parse the inner JSON string (gpt_queries)
                gpt_queries = json.loads(entry['gpt_queries'])
                
                # Extract each query
                for query in gpt_queries['queries']:
                    records.append({
                        'paper_id': paper_id,
                        'query_type': query['breadth'],
                        'query': query['text'],
                        'relevance_idx': []  # Will be filled later if relevance_indices provided
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Warning: Missing key in line {line_num}: {e}")
                continue
    
    # Fill in relevance indices if provided
    if relevance_indices is not None:
        if len(relevance_indices) != len(records):
            raise ValueError(
                f"Length mismatch: relevance_indices has {len(relevance_indices)} elements "
                f"but there are {len(records)} query records"
            )
        for i, idx_list in enumerate(relevance_indices):
            records[i]['relevance_idx'] = idx_list
                
    return records


def convert_to_parquet(input_file: str, output_file: str | None = None, relevance_indices: List[List[int]] | None = None):
    """
    Convert JSONL file to Parquet format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output Parquet file (optional)
        relevance_indices: Optional list of lists containing relevance indices for each query.
                          Must have the same length as the total number of queries.
    """
    # Set default output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.parquet'))
    
    print(f"Reading from: {input_file}")
    
    # Parse the JSONL file
    records = parse_jsonl_file(input_file, relevance_indices)
    
    if not records:
        print("Error: No records found in the input file")
        return
    
    print(f"Parsed {len(records)} query records")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Display some statistics
    print("\nDataset statistics:")
    print(f"  Total queries: {len(df)}")
    print(f"  Unique papers: {df['paper_id'].nunique()}")
    print("\nQuery type distribution:")
    print(df['query_type'].value_counts())
    
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
        description='Convert GPT queries JSONL file to Parquet dataset'
    )
    parser.add_argument(
        'input_file',
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path to output Parquet file (default: input_file.parquet)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # relevance_indices = None  # Set to None to use empty lists for all queries

    relevance_indices = [
        [0, 1, 4, 6],   # 1
        [0, 6],         # 2
        [0, 2, 4, 6],   # 3
        [0, 1, 9],      # 4
        [1],            # 5
        [1],            # 6
        [1, 2, 4],      # 7
        [2],            # 8
        [0, 2, 3, 4],   # 9
        [3],            # 10
        [0, 3],         # 11
        [3],            # 12
        [0, 4],         # 13
        [0, 3, 4],      # 14
        [0, 4],         # 15
        [5],            # 16
        [5],            # 17
        [5],            # 18
        [0, 4, 6],      # 19
        [0, 1, 6],      # 20
        [6],            # 21
        [7],            # 22
        [7],            # 23
        [7],            # 24
        [4, 5, 8, 9],   # 25
        [5, 8],         # 26
        [8],            # 27
        [9],            # 28
        [9],            # 29
        [9]             # 30
    ]
    
    # Convert to Parquet
    convert_to_parquet(args.input_file, args.output, relevance_indices)


if __name__ == '__main__':
    main()
