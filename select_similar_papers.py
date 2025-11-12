#!/usr/bin/env python3
"""
Script to select a group of papers that are highly similar to each other.
The goal is to find ~10 papers where each paper has high similarity (>= 0.8) 
to several other papers in the group.
"""

import json
import sys
from typing import Dict, List, Tuple
from collections import defaultdict

def load_similarity_data(filepath: str) -> Dict[str, Dict[str, float]]:
    """Load the similarity data from JSONL file."""
    similarity_matrix = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            paper_id = data['paper_id']
            similarity_scores = data.get('similarity_scores', {})
            similarity_matrix[paper_id] = similarity_scores
    
    return similarity_matrix

def find_highly_connected_papers(
    similarity_matrix: Dict[str, Dict[str, float]], 
    threshold: float = 0.8,
    target_size: int = 10
) -> List[str]:
    """
    Find a group of papers that are highly connected.
    
    Strategy:
    1. For each paper, count how many other papers it has similarity >= threshold with
    2. Start with the paper that has the most high-similarity connections
    3. Build a cluster by iteratively adding papers that have high similarity to existing cluster members
    """
    
    # Count high-similarity connections for each paper
    connection_counts = {}
    high_similarity_papers = defaultdict(set)
    
    for paper_id, scores in similarity_matrix.items():
        high_sim_papers = {other_id for other_id, score in scores.items() 
                          if score >= threshold and other_id != paper_id}
        connection_counts[paper_id] = len(high_sim_papers)
        high_similarity_papers[paper_id] = high_sim_papers
    
    # Sort papers by number of high-similarity connections
    sorted_papers = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 papers by number of connections with similarity >= {threshold}:")
    for i, (paper_id, count) in enumerate(sorted_papers[:20]):
        print(f"  {i+1}. {paper_id}: {count} connections")
    
    # Strategy: Find a clique or densely connected subgraph
    # Start with papers that have many mutual high-similarity connections
    
    # Find the paper with most connections
    start_paper = sorted_papers[0][0]
    
    # Get all papers similar to the start paper
    candidates = high_similarity_papers[start_paper].copy()
    
    # Build cluster by finding papers that are mutually similar
    cluster = {start_paper}
    
    # Sort candidates by how many connections they have to papers already in cluster
    while len(cluster) < target_size and candidates:
        best_paper = None
        best_score = -1
        
        for candidate in candidates:
            # Count connections to current cluster
            # Need to check BOTH directions since similarity is stored only one way
            connections_to_cluster = 0
            for cluster_paper in cluster:
                # Check if candidate is similar to cluster_paper
                if cluster_paper in high_similarity_papers[candidate]:
                    connections_to_cluster += 1
                # Also check if cluster_paper is similar to candidate
                elif candidate in high_similarity_papers[cluster_paper]:
                    connections_to_cluster += 1
            
            if connections_to_cluster > best_score:
                best_score = connections_to_cluster
                best_paper = candidate
        
        if best_paper is None or best_score == 0:
            print(f"\nCouldn't find more papers with connections to cluster. Stopping at {len(cluster)} papers.")
            break
        
        cluster.add(best_paper)
        candidates.remove(best_paper)
        # Add new candidates that are similar to the newly added paper
        new_candidates = high_similarity_papers[best_paper] - cluster
        candidates.update(new_candidates)
        
        print(f"Added {best_paper} (similar to {best_score}/{len(cluster)-1} papers in cluster)")
    
    return list(cluster)

def analyze_cluster(
    cluster: List[str], 
    similarity_matrix: Dict[str, Dict[str, float]],
    threshold: float = 0.8
) -> None:
    """Analyze the connectivity within the selected cluster."""
    
    print(f"\n{'='*80}")
    print(f"CLUSTER ANALYSIS")
    print(f"{'='*80}")
    print(f"\nCluster size: {len(cluster)} papers")
    print(f"Similarity threshold: {threshold}")
    
    # Build connectivity matrix for cluster
    print(f"\nConnectivity matrix (showing similarities >= {threshold}):")
    print(f"\n{' '*12}", end='')
    for paper_id in cluster:
        print(f"{paper_id[:8]:>10}", end='')
    print()
    
    all_connections = []
    for paper1 in cluster:
        print(f"{paper1[:10]:10} ", end='')
        connections_for_paper = 0
        for paper2 in cluster:
            if paper1 == paper2:
                print(f"{'---':>10}", end='')
            else:
                # Check both directions since similarity is not symmetric in the data
                score = similarity_matrix.get(paper1, {}).get(paper2, 0.0)
                if score < threshold:
                    score = similarity_matrix.get(paper2, {}).get(paper1, 0.0)
                
                if score >= threshold:
                    print(f"{score:.3f}:>10", end='')
                    connections_for_paper += 1
                    # Only add connection once (avoid duplicates)
                    if paper1 < paper2:  # lexicographic comparison to avoid duplicates
                        all_connections.append((paper1, paper2, score))
                else:
                    print(f"{'':>10}", end='')
        print(f"  ({connections_for_paper} connections)")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    total_possible = len(cluster) * (len(cluster) - 1) // 2
    total_connections = len(all_connections)  # Already deduplicated
    
    print(f"\nTotal possible connections: {total_possible}")
    print(f"Actual connections (>= {threshold}): {total_connections}")
    if total_possible > 0:
        print(f"Connectivity rate: {100*total_connections/total_possible:.1f}%")
    
    # Per-paper statistics
    connection_counts = defaultdict(int)
    for paper1, paper2, _ in all_connections:
        connection_counts[paper1] += 1
        connection_counts[paper2] += 1  # Count for both papers since we deduplicated
    
    print(f"\nConnections per paper:")
    for paper_id in cluster:
        count = connection_counts[paper_id]
        print(f"  {paper_id}: {count}/{len(cluster)-1} ({100*count/(len(cluster)-1):.1f}%)")
    
    # Show strongest connections
    all_connections_sorted = sorted(all_connections, key=lambda x: x[2], reverse=True)
    print(f"\nTop 20 strongest connections:")
    shown = set()
    count = 0
    for paper1, paper2, score in all_connections_sorted:
        pair = tuple(sorted([paper1, paper2]))
        if pair not in shown:
            shown.add(pair)
            print(f"  {count+1}. {paper1} <-> {paper2}: {score:.4f}")
            count += 1
            if count >= 20:
                break

def load_paper_data(filepath: str) -> Dict[str, Dict]:
    """Load full paper data from JSONL file."""
    papers = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            papers[data['paper_id']] = data
    return papers

def save_selected_papers_with_relevance(
    cluster: List[str],
    similarity_matrix: Dict[str, Dict[str, float]],
    paper_data: Dict[str, Dict],
    output_file: str
) -> None:
    """
    Save selected papers to JSONL file with relevance scores to other selected papers only.
    
    Args:
        cluster: List of selected paper IDs
        similarity_matrix: Dictionary mapping paper_id to similarity scores
        paper_data: Dictionary mapping paper_id to full paper data
        output_file: Path to output JSONL file
    """
    print(f"\nSaving {len(cluster)} papers with relevance scores to {output_file}...")
    
    # Convert cluster to set for faster lookup
    cluster_set = set(cluster)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper_id in cluster:
            # Get the paper data
            paper = paper_data.get(paper_id, {})
            
            # Create a copy to avoid modifying original
            output_paper = dict(paper)
            
            # Add relevance scores only for other papers in the cluster
            relevance = {}
            
            for other_paper_id in cluster:
                if other_paper_id == paper_id:
                    # Self-similarity is 1.0
                    relevance[other_paper_id] = 1.0
                else:
                    # Check if we have similarity score in either direction
                    score = 0.0
                    
                    # Check paper_id -> other_paper_id
                    if paper_id in similarity_matrix:
                        score = similarity_matrix[paper_id].get(other_paper_id, 0.0)
                    
                    # If not found, check other_paper_id -> paper_id
                    if score == 0.0 and other_paper_id in similarity_matrix:
                        score = similarity_matrix[other_paper_id].get(paper_id, 0.0)
                    
                    relevance[other_paper_id] = score
            
            output_paper['relevance'] = relevance
            
            # Write to file
            f.write(json.dumps(output_paper) + '\n')
    
    print(f"Successfully saved {len(cluster)} papers to {output_file}")

def main():
    similarity_filepath = "sem_similarity/gte-Qwen2-1.5B-instruct/all_new/similarity_papers.jsonl"
    papers_filepath = "data_scisumm/cl_scisumm.jsonl"
    output_filepath = "data_scisumm/selected_papers.jsonl"
    
    print("Loading similarity data...")
    similarity_matrix = load_similarity_data(similarity_filepath)
    print(f"Loaded similarity data for {len(similarity_matrix)} papers")
    
    print("\nLoading paper data...")
    paper_data = load_paper_data(papers_filepath)
    print(f"Loaded full data for {len(paper_data)} papers")
    
    # Try different thresholds
    cluster = None
    for threshold in [0.85, 0.82, 0.80]:
        print(f"\n\n{'#'*80}")
        print(f"# TRYING THRESHOLD = {threshold}")
        print(f"{'#'*80}")
        
        cluster = find_highly_connected_papers(
            similarity_matrix, 
            threshold=threshold,
            target_size=10
        )
        
        print(f"\nSelected papers:")
        for i, paper_id in enumerate(cluster, 1):
            print(f"  {i}. {paper_id}")
        
        analyze_cluster(cluster, similarity_matrix, threshold)
        
        if len(cluster) >= 10:
            break
    
    # Save the selected papers with relevance scores
    if cluster and len(cluster) >= 10:
        save_selected_papers_with_relevance(
            cluster[:10],  # Take first 10 papers
            similarity_matrix,
            paper_data,
            output_filepath
        )
    else:
        print("\nWarning: Could not find 10 highly connected papers!")

if __name__ == "__main__":
    main()
