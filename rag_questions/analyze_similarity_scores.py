#!/usr/bin/env python3
"""
Script to analyze similarity scores from similarity_papers.jsonl
Counts papers with similarity >= 0.8, grouped by 0.01 bins
Excludes self-similarity (score = 1.0)
"""

import json
from collections import defaultdict


def load_similarity_data(filepath):
    """Load similarity data from JSONL file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def analyze_similarities(filepath, min_threshold=0.8):
    """
    Analyze similarity scores and count papers per 0.01 bin
    
    Args:
        filepath: Path to the similarity_papers.jsonl file
        min_threshold: Minimum similarity threshold (default: 0.8)
    
    Returns:
        Dictionary with bins as keys and counts as values
    """
    # Dictionary to store counts per bin
    bin_counts = defaultdict(int)
    
    # Process each paper's similarity scores
    for paper_data in load_similarity_data(filepath):
        paper_id = paper_data.get('paper_id')
        similarity_scores = paper_data.get('similarity_scores', {})
        
        for other_paper_id, score in similarity_scores.items():
            # Exclude self-similarity
            if other_paper_id == paper_id:
                continue
            
            # Only consider scores >= threshold
            if score >= min_threshold:
                # Calculate bin (e.g., 0.80, 0.81, 0.82, etc.)
                bin_value = round(score, 2)
                bin_counts[bin_value] += 1
    
    return bin_counts


def print_results(bin_counts):
    """Print results in a formatted table"""
    print("=" * 50)
    print("Similarity Score Distribution (≥ 0.80)")
    print("Excludes self-similarity")
    print("=" * 50)
    print(f"{'Bin':<10} {'Count':<10}")
    print("-" * 50)
    
    # Sort bins from 0.80 to 1.00
    total_count = 0
    for bin_value in sorted(bin_counts.keys()):
        if 0.8 <= bin_value < 1.0:  # Exclude 1.0 (self-similarity)
            count = bin_counts[bin_value]
            print(f"{bin_value:<10.2f} {count:<10}")
            total_count += count
    
    print("-" * 50)
    print(f"{'Total':<10} {total_count:<10}")
    print("=" * 50)


def main():
    # Path to the similarity file
    filepath = r"D:\Repositories\cl_scientific_papers\sem_similarity\Qwen3-Embedding-0.6B\all\similarity_papers.jsonl"
    
    print("Analyzing similarity scores...")
    print(f"File: {filepath}\n")
    
    # Analyze similarities
    bin_counts = analyze_similarities(filepath, min_threshold=0.8)
    
    # Print results
    print_results(bin_counts)
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"Bins with data: {len([b for b in bin_counts.keys() if 0.8 <= b < 1.0])}")
    
    # Find bin with most papers
    if bin_counts:
        max_bin = max((b for b in bin_counts.keys() if 0.8 <= b < 1.0), key=lambda x: bin_counts[x], default=None)
        if max_bin:
            print(f"Most common bin: {max_bin:.2f} with {bin_counts[max_bin]} papers")


if __name__ == "__main__":
    main()
