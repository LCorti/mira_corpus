"""
Generate relevance indices for document retrieval evaluation.

Based on LLM answerability judgments, create a list mapping each question
to the paper indices (positions in selected_papers.jsonl) that can answer it.

Output format:
relevance_indices = [
    [0, 1, 4],  # Question 1: papers at indices 0, 1, 4 are relevant
    [2, 5],     # Question 2: papers at indices 2, 5 are relevant
    ...
]
"""

import json
import sys

from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
import utils


def load_answerability_data(file_path: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load answerability evaluations from JSONL.

    Returns a dict: {(paper_id, question_id): evaluation_data}
    """
    answerability = {}

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            paper_id = data["paper_id"]
            question_id = data["question_id"]
            answerability[(paper_id, question_id)] = data

    return answerability


def extract_answerability_bool(evaluation: Dict) -> str:
    """
    Extract the binary answerability decision from an evaluation.

    Returns: "yes", "no", or "error"
    """
    llm_response = evaluation.get("llm_response", {})

    if "error" in llm_response:
        return "error"

    answerable = llm_response.get("answerable", "unknown")
    return answerable if answerable in ["yes", "no"] else "unknown"


def create_paper_id_to_index_map(papers: List[Dict]) -> Dict[str, int]:
    """
    Create a mapping from paper_id to its index in the papers list.

    Args:
        papers: List of paper dictionaries from selected_papers.jsonl

    Returns:
        Dict mapping paper_id to index (0-based)
    """
    return {paper["paper_id"]: idx for idx, paper in enumerate(papers)}


def extract_question_order(papers: List[Dict]) -> List[str]:
    """
    Extract the canonical question order based on paper order.

    For each paper (in order), we have 3 questions: Q1, Q2, Q3

    Returns:
        List of question_ids in canonical order
    """
    question_ids = []
    for paper in papers:
        paper_id = paper["paper_id"]
        for q_num in [1, 2, 3]:
            question_ids.append(f"{paper_id}_Q{q_num}")

    return question_ids


def generate_relevance_indices(
    papers: List[Dict], answerability: Dict[Tuple[str, str], Dict]
) -> List[List[int]]:
    """
    Generate relevance indices for each question.

    Args:
        papers: List of papers from selected_papers.jsonl
        answerability: Dict of answerability evaluations

    Returns:
        List of lists, where each inner list contains paper indices that
        can answer the question at that position
    """
    paper_id_to_idx = create_paper_id_to_index_map(papers)
    question_ids = extract_question_order(papers)

    relevance_indices = []

    for question_id in question_ids:
        relevant_paper_indices = []

        # Check all papers for this question
        for paper_id, paper_idx in paper_id_to_idx.items():
            pair_key = (paper_id, question_id)

            if pair_key in answerability:
                eval_data = answerability[pair_key]
                answerable = extract_answerability_bool(eval_data)

                if answerable == "yes":
                    relevant_paper_indices.append(paper_idx)

        # Sort indices for consistent output
        relevant_paper_indices.sort()
        relevance_indices.append(relevant_paper_indices)

    return relevance_indices


def format_relevance_indices(relevance_indices: List[List[int]]) -> str:
    """
    Format relevance indices as a Python list literal.

    Returns:
        String representation suitable for copy-paste into code
    """
    lines = ["relevance_indices = ["]

    for i, indices in enumerate(relevance_indices, 1):
        indices_str = str(indices).replace(" ", "")
        comment = f"# {i}"

        # Pad the line for alignment
        line = f"        {indices_str},"

        # Add comment aligned at a fixed column
        if len(line) < 40:
            line = line.ljust(40) + comment
        else:
            line = line + "  " + comment

        lines.append(line)

    lines.append("    ]")

    return "\n".join(lines)


def write_statistics(relevance_indices: List[List[int]], output_path: Path):
    """
    Write statistics about the relevance indices.
    """
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("RELEVANCE INDICES STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        total_questions = len(relevance_indices)
        total_relevances = sum(len(indices) for indices in relevance_indices)

        f.write(f"Total questions: {total_questions}\n")
        f.write(f"Total relevance judgments: {total_relevances}\n")
        f.write(
            f"Average relevant papers per question: {total_relevances / total_questions:.2f}\n\n"
        )

        # Distribution of relevant papers per question
        f.write("Distribution of relevant papers per question:\n")
        f.write("-" * 80 + "\n")

        from collections import Counter

        distribution = Counter(len(indices) for indices in relevance_indices)

        for num_papers in sorted(distribution.keys()):
            count = distribution[num_papers]
            f.write(f"  {num_papers} relevant paper(s): {count} questions\n")

        f.write("\n")

        # Questions with no relevant papers (should be 0 based on analysis)
        questions_without_relevant = sum(
            1 for indices in relevance_indices if len(indices) == 0
        )
        f.write(
            f"Questions without any relevant papers: {questions_without_relevant}\n"
        )

        if questions_without_relevant > 0:
            f.write("\nWARNING: Some questions have no relevant papers!\n")
            for i, indices in enumerate(relevance_indices, 1):
                if len(indices) == 0:
                    f.write(f"  Question {i} has no relevant papers\n")

        f.write("\n")

        # Detailed breakdown
        f.write("=" * 80 + "\n")
        f.write("DETAILED QUESTION-BY-QUESTION BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")

        for i, indices in enumerate(relevance_indices, 1):
            f.write(f"Question {i}: {len(indices)} relevant paper(s) -> {indices}\n")


def main():
    # Paths
    papers_path = Path("data", "selected_papers.jsonl")
    answerability_path = Path(
        "data",
        "question_answerability_evaluations_similarity_aware.jsonl",
    )
    output_code_path = Path("rag_questions", "data", "relevance_indices.py")
    output_stats_path = Path(
        "output",
        "similarity_aware",
        "relevance_indices_statistics.txt",
    )

    print("Loading data...")
    papers = utils.load_jsonl(papers_path)
    answerability = load_answerability_data(answerability_path)

    print(f"Loaded {len(papers)} papers")
    print(f"Loaded {len(answerability)} answerability evaluations")

    print("\nGenerating relevance indices...")
    relevance_indices = generate_relevance_indices(papers, answerability)

    print(f"Generated relevance indices for {len(relevance_indices)} questions")

    # Format as Python code
    print("\nFormatting output...")
    code_output = format_relevance_indices(relevance_indices)

    # Write to file
    print(f"Writing to {output_code_path}...")
    with open(output_code_path, "w") as f:
        f.write('"""\n')
        f.write("Relevance indices for document retrieval evaluation.\n\n")
        f.write(
            "Each list contains the paper indices (0-based positions in selected_papers.jsonl)\n"
        )
        f.write("that can answer the question at that position.\n\n")
        f.write("Generated automatically from LLM answerability judgments.\n")
        f.write('"""\n\n')
        f.write(code_output)
        f.write("\n")

    print(f"Relevance indices saved to {output_code_path}")

    # Write statistics
    print(f"\nWriting statistics to {output_stats_path}...")
    write_statistics(relevance_indices, output_stats_path)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total questions: {len(relevance_indices)}")
    total_relevances = sum(len(indices) for indices in relevance_indices)
    print(f"Total relevance judgments: {total_relevances}")
    print(
        f"Average relevant papers per question: {total_relevances / len(relevance_indices):.2f}"
    )

    questions_without_relevant = sum(
        1 for indices in relevance_indices if len(indices) == 0
    )
    print(f"Questions without relevant papers: {questions_without_relevant}")

    if questions_without_relevant == 0:
        print("\n[OK] All questions have at least one relevant paper!")
        print("  Dataset is ready for document retrieval evaluation.")
    else:
        print("\n!!! WARNING: Some questions have no relevant papers!")

    print("\nFiles generated:")
    print(f"  - {output_code_path}")
    print(f"  - {output_stats_path}")


if __name__ == "__main__":
    main()
