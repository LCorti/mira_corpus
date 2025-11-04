#!/usr/bin/env python3
"""
Convert scientific papers JSONL files to parquet format for Hugging Face datasets.
Supports two formats:
1. gpt_simplified.jsonl - contains simplified summaries
2. gpt_summaries.jsonl - contains full summaries with F1 scores

Each sample represents one paper with its metadata and different types of summaries.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
import random
from typing import Optional


def convert_simplified_jsonl_to_parquet(input_path: str, output_path: str, num_samples: Optional[int] = None):
    """
    Convert gpt_simplified.jsonl file to parquet format.
    
    Args:
        input_path: Path to the input JSONL file
        output_path: Path to the output parquet file
        num_samples: Number of random samples to select (None for all)
    """
    papers = []
    
    print(f"Reading from {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                paper = json.loads(line.strip())
                
                # Extract the fields we need
                paper_data = {
                    'paper_id': paper.get('paper_id', ''),
                    'title': extract_title_from_text(paper.get('text', '')),
                    'abstract': paper.get('abstract', ''),
                    'full_text': paper.get('text', ''),
                    'human_summary': paper.get('human', ''),
                    'gpt_simplified_summary': paper.get('gpt_simple', '')
                }
                
                papers.append(paper_data)
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} papers...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Successfully processed {len(papers)} papers")
    
    # Sample if requested
    if num_samples is not None and len(papers) > num_samples:
        print(f"Randomly sampling {num_samples} papers from {len(papers)} total papers...")
        papers = random.sample(papers, num_samples)
        print(f"Selected {len(papers)} papers")
    
    # Create DataFrame
    df = pd.DataFrame(papers)
    
    # Print some statistics
    print(f"\nDataset statistics:")
    print(f"Total papers: {len(df)}")
    print(f"Papers with abstracts: {df['abstract'].notna().sum()}")
    print(f"Papers with human summaries: {df['human_summary'].notna().sum()}")
    print(f"Papers with GPT simplified summaries: {df['gpt_simplified_summary'].notna().sum()}")
    
    # Show sample of data
    print(f"\nSample paper IDs: {df['paper_id'].head().tolist()}")
    
    # Save to parquet
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    print("Conversion completed successfully!")
    return df


def convert_summaries_jsonl_to_parquet(input_path: str, output_path: str, num_samples: Optional[int] = None):
    """
    Convert gpt_summaries.jsonl file to parquet format.
    This version includes F1 scores and different field names.
    
    Args:
        input_path: Path to the input JSONL file
        output_path: Path to the output parquet file
        num_samples: Number of random samples to select (None for all)
    """
    papers = []
    
    print(f"Reading from {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                paper = json.loads(line.strip())
                
                # Extract the fields we need for summaries format
                paper_data = {
                    'paper_id': paper.get('paper_id', ''),
                    'title': extract_title_from_text(paper.get('text', '')),
                    'abstract': paper.get('abstract', ''),
                    'full_text': paper.get('text', ''),
                    'human_summary': paper.get('human', ''),
                    'gpt_summary': paper.get('gpt_summary', ''),
                    'f1_score': paper.get('f1', None)
                }
                
                papers.append(paper_data)
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} papers...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Successfully processed {len(papers)} papers")
    
    # Sample if requested
    if num_samples is not None and len(papers) > num_samples:
        print(f"Randomly sampling {num_samples} papers from {len(papers)} total papers...")
        papers = random.sample(papers, num_samples)
        print(f"Selected {len(papers)} papers")
    
    # Create DataFrame
    df = pd.DataFrame(papers)
    
    # Print some statistics
    print(f"\nDataset statistics:")
    print(f"Total papers: {len(df)}")
    print(f"Papers with abstracts: {df['abstract'].notna().sum()}")
    print(f"Papers with human summaries: {df['human_summary'].notna().sum()}")
    print(f"Papers with GPT summaries: {df['gpt_summary'].notna().sum()}")
    if 'f1_score' in df.columns:
        print(f"Papers with F1 scores: {df['f1_score'].notna().sum()}")
        print(f"Average F1 score: {df['f1_score'].mean():.4f}")
    
    # Show sample of data
    print(f"\nSample paper IDs: {df['paper_id'].head().tolist()}")
    
    # Save to parquet
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    print("Conversion completed successfully!")
    return df


def extract_title_from_text(text: str) -> str:
    """
    Extract title from the full text.
    Assumes the title is on the first line after "Title:" or just the first non-empty line.
    """
    if not text:
        return ""
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Title:'):
            # Remove "Title:" prefix and return the rest
            return line[6:].strip()
        elif line and not line.startswith(('ABSTRACT', 'Abstract')):
            # If no "Title:" found, return the first non-empty line that's not "ABSTRACT"
            return line
    
    return ""


def main():
    parser = argparse.ArgumentParser(description='Convert scientific papers JSONL to parquet format')
    parser.add_argument('--input', '-i', default='data_scisumm/gpt_simplified.jsonl', help='Path to input JSONL file (default: data_scisumm/gpt_simplified.jsonl)')
    parser.add_argument('--output', '-o', default='parquet/scientific_papers_dataset.parquet', help='Path to output parquet file (default: parquet/scientific_papers_dataset.parquet)')
    parser.add_argument('--format', '-f', choices=['simplified', 'summaries'], default='simplified', 
                        help='Format of the input file: "simplified" for gpt_simplified.jsonl or "summaries" for gpt_summaries.jsonl (default: simplified)')
    parser.add_argument('--num-samples', '-n', type=int, default=None,
                        help='Number of random samples to select from the dataset (default: all papers)')
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist!")
        return 1
    
    try:
        # Choose the appropriate conversion function based on format
        if args.format == 'simplified':
            df = convert_simplified_jsonl_to_parquet(str(input_path), str(output_path), args.num_samples)
        else:  # summaries
            df = convert_summaries_jsonl_to_parquet(str(input_path), str(output_path), args.num_samples)
        
        # Display some sample data
        print(f"\nFirst few rows of the dataset:")
        print(df[['paper_id', 'title']].head())
        
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())