#!/usr/bin/env python3
"""
Analyze the differences between paper_only and similarity_aware generated queries.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(filepath: str) -> Dict[str, dict]:
    """Load JSONL file and return dictionary keyed by paper_id"""
    data = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data[entry["paper_id"]] = entry
    return data


def extract_queries(gpt_queries_str: str) -> List[Dict[str, str]]:
    """Extract query objects (text and breadth) from the gpt_queries JSON string"""
    try:
        queries_obj = json.loads(gpt_queries_str)
        return [
            {"text": q["text"], "breadth": q.get("breadth", "unknown")}
            for q in queries_obj["queries"]
        ]
    except Exception as _:
        return []


def compare_queries(paper_only: Dict[str, dict], similarity_aware: Dict[str, dict]):
    """Compare queries between the two datasets"""

    common_papers = set(paper_only.keys()) & set(similarity_aware.keys())

    print("=" * 80)
    print("Query Comparison Analysis")
    print("=" * 80)
    print(f"\nTotal papers in paper_only: {len(paper_only)}")
    print(f"Total papers in similarity_aware: {len(similarity_aware)}")
    print(f"Common papers: {len(common_papers)}")
    print()

    if not common_papers:
        print("No common papers found!")
        return

    # Analyze differences for each paper
    identical_count = 0
    different_count = 0
    differences = []

    for paper_id in sorted(common_papers):
        po_queries = extract_queries(paper_only[paper_id]["gpt_queries"])
        sa_queries = extract_queries(similarity_aware[paper_id]["gpt_queries"])

        if not po_queries or not sa_queries:
            continue

        # Check if queries are identical (comparing just text)
        po_texts = [q["text"] for q in po_queries]
        sa_texts = [q["text"] for q in sa_queries]

        if po_texts == sa_texts:
            identical_count += 1
        else:
            different_count += 1
            differences.append(
                {
                    "paper_id": paper_id,
                    "po_queries": po_queries,
                    "sa_queries": sa_queries,
                }
            )

    print(f"Papers with IDENTICAL queries: {identical_count}")
    print(f"Papers with DIFFERENT queries: {different_count}")
    print()

    # Show detailed differences
    if differences:
        print("=" * 80)
        print("DETAILED DIFFERENCES")
        print("=" * 80)
        print()

        for i, diff in enumerate(differences, 1):
            print(f"{'-' * 80}")
            print(f"Paper {i}: {diff['paper_id']}")
            print(f"{'-' * 80}")
            print()

            for j, (po_q, sa_q) in enumerate(
                zip(diff["po_queries"], diff["sa_queries"]), 1
            ):
                print(f"Query {j} ({po_q['breadth']}):")
                print(f"  Paper Only:        {po_q['text']}")
                print(f"  Similarity Aware:  {sa_q['text']}")

                if po_q["text"] == sa_q["text"]:
                    print("  Status: IDENTICAL")
                else:
                    print("  Status: DIFFERENT")
                print()

            if i < len(differences):
                print()

    # Analysis of similarity information
    print(f"\n{'=' * 80}")
    print("SIMILARITY INFORMATION ANALYSIS")
    print(f"{'=' * 80}\n")

    for paper_id in sorted(list(common_papers)[:3]):  # Show first 3
        sa_entry = similarity_aware[paper_id]
        if "similarity" in sa_entry:
            print(f"Paper {paper_id}:")
            print(f"  Has similarity score: {sa_entry['similarity']:.4f}")

        if "similar_papers" in sa_entry:
            print(f"  Similar papers: {len(sa_entry.get('similar_papers', []))}")
            for sp in sa_entry.get("similar_papers", [])[:3]:
                print(
                    f"    - {sp.get('paper_id', 'unknown')}: {sp.get('similarity', 0):.4f}"
                )
        print()


if __name__ == "__main__":
    paper_only_path = Path(
        Path(__file__).parent, "data", "gpt_questions_paper_only.jsonl"
    )
    similarity_aware_path = Path(
        Path(__file__).parent, "data", "gpt_questions_similarity_aware.jsonl"
    )

    print("Loading data...")
    paper_only = load_jsonl(paper_only_path)
    similarity_aware = load_jsonl(similarity_aware_path)
    compare_queries(paper_only, similarity_aware)
