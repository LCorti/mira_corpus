"""
Analyze the overlap between semantic similarity scores and LLM answerability judgments.

This script compares:
1. Semantic similarity scores (from embedding-based similarity)
2. LLM answerability judgments (from gpt-oss evaluations)

Goal: Determine if high similarity correlates with answerability, and ensure each question has at least one paper that can answer it.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def parse_similarity_file(file_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse the similarity values text file.

    Returns a nested dict: {paper_id: {question_id: similarity_score}}
    """
    similarities = {}
    current_paper = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Check if this is a paper header
            if line.startswith("Paper:"):
                current_paper = line.split("Paper:")[1].strip()
                similarities[current_paper] = {}

            # Check if this is a question-similarity pair
            elif ":" in line and current_paper:
                parts = line.split(":")
                if len(parts) == 2:
                    question_id = parts[0].strip()
                    try:
                        similarity = float(parts[1].strip())
                        similarities[current_paper][question_id] = similarity
                    except ValueError:
                        # Not a similarity line, skip
                        pass

    return similarities


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


def analyze_overlap(
    similarities: Dict[str, Dict[str, float]],
    answerability: Dict[Tuple[str, str], Dict],
) -> Dict:
    """
    Analyze the overlap between similarity scores and answerability.

    Returns statistics and data for visualization.
    """
    results = {
        "all_pairs": [],
        "by_question": defaultdict(list),
        "by_paper": defaultdict(list),
        "answerable_pairs": [],
        "not_answerable_pairs": [],
        "error_pairs": [],
    }

    # Process all paper-question pairs
    for paper_id, questions in similarities.items():
        for question_id, similarity in questions.items():
            pair_key = (paper_id, question_id)

            if pair_key in answerability:
                eval_data = answerability[pair_key]
                answerable = extract_answerability_bool(eval_data)

                pair_info = {
                    "paper_id": paper_id,
                    "question_id": question_id,
                    "similarity": similarity,
                    "answerable": answerable,
                }

                results["all_pairs"].append(pair_info)
                results["by_question"][question_id].append(pair_info)
                results["by_paper"][paper_id].append(pair_info)

                if answerable == "yes":
                    results["answerable_pairs"].append(pair_info)
                elif answerable == "no":
                    results["not_answerable_pairs"].append(pair_info)
                else:
                    results["error_pairs"].append(pair_info)

    return results


def compute_statistics(results: Dict) -> Dict:
    """
    Compute summary statistics about the overlap.
    """
    stats = {}

    # Overall statistics
    stats["total_pairs"] = len(results["all_pairs"])
    stats["answerable_count"] = len(results["answerable_pairs"])
    stats["not_answerable_count"] = len(results["not_answerable_pairs"])
    stats["error_count"] = len(results["error_pairs"])

    # Similarity statistics by answerability
    if results["answerable_pairs"]:
        answerable_sims = [p["similarity"] for p in results["answerable_pairs"]]
        stats["answerable_sim_mean"] = np.mean(answerable_sims)
        stats["answerable_sim_std"] = np.std(answerable_sims)
        stats["answerable_sim_min"] = np.min(answerable_sims)
        stats["answerable_sim_max"] = np.max(answerable_sims)

    if results["not_answerable_pairs"]:
        not_answerable_sims = [p["similarity"] for p in results["not_answerable_pairs"]]
        stats["not_answerable_sim_mean"] = np.mean(not_answerable_sims)
        stats["not_answerable_sim_std"] = np.std(not_answerable_sims)
        stats["not_answerable_sim_min"] = np.min(not_answerable_sims)
        stats["not_answerable_sim_max"] = np.max(not_answerable_sims)

    # Per-question analysis: does each question have at least one answerable paper?
    stats["questions_with_answerable_paper"] = 0
    stats["questions_without_answerable_paper"] = 0
    stats["question_details"] = {}

    for question_id, pairs in results["by_question"].items():
        answerable_count = sum(1 for p in pairs if p["answerable"] == "yes")
        has_answerable = answerable_count > 0

        if has_answerable:
            stats["questions_with_answerable_paper"] += 1
        else:
            stats["questions_without_answerable_paper"] += 1

        # Find best similarity among answerable papers
        answerable_pairs = [p for p in pairs if p["answerable"] == "yes"]
        best_answerable_sim = (
            max([p["similarity"] for p in answerable_pairs])
            if answerable_pairs
            else None
        )

        # Find best similarity overall
        best_overall_sim = max([p["similarity"] for p in pairs])
        best_overall_paper = max(pairs, key=lambda p: p["similarity"])["paper_id"]

        stats["question_details"][question_id] = {
            "answerable_count": answerable_count,
            "total_papers": len(pairs),
            "best_answerable_similarity": best_answerable_sim,
            "best_overall_similarity": best_overall_sim,
            "best_overall_paper": best_overall_paper,
            "has_answerable_paper": has_answerable,
        }

    return stats


def create_visualizations(results: Dict, stats: Dict, output_dir: Path):
    """
    Create visualizations of the overlap analysis.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Histogram: Similarity distribution by answerability
    fig, ax = plt.subplots(figsize=(12, 6))

    if results["answerable_pairs"]:
        answerable_sims = [p["similarity"] for p in results["answerable_pairs"]]
        ax.hist(
            answerable_sims, bins=30, alpha=0.6, label="Answerable (yes)", color="green"
        )

    if results["not_answerable_pairs"]:
        not_answerable_sims = [p["similarity"] for p in results["not_answerable_pairs"]]
        ax.hist(
            not_answerable_sims,
            bins=30,
            alpha=0.6,
            label="Not answerable (no)",
            color="red",
        )

    ax.set_xlabel("Semantic Similarity Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Similarity Scores by Answerability", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        Path(output_dir, "similarity_distribution_by_answerability_paper_only.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Box plot: Similarity by answerability
    fig, ax = plt.subplots(figsize=(8, 6))

    data_to_plot = []
    labels = []

    if results["answerable_pairs"]:
        data_to_plot.append([p["similarity"] for p in results["answerable_pairs"]])
        labels.append("Answerable")

    if results["not_answerable_pairs"]:
        data_to_plot.append([p["similarity"] for p in results["not_answerable_pairs"]])
        labels.append("Not Answerable")

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, patch_artist=True)
        ax.set_xticklabels(labels)
        bp["boxes"][0].set_facecolor("lightgreen")
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor("lightcoral")

    ax.set_ylabel("Semantic Similarity Score", fontsize=12)
    ax.set_title("Similarity Score Distribution by Answerability", fontsize=14)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        Path(output_dir, "similarity_boxplot_by_answerability_paper_only.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Per-question analysis: Does the highest similarity paper answer the question?
    question_ids = sorted(stats["question_details"].keys())
    highest_sim_is_answerable = []

    for qid in question_ids:
        details = stats["question_details"][qid]
        # Check if the paper with highest similarity is answerable
        question_pairs = results["by_question"][qid]
        best_pair = max(question_pairs, key=lambda p: p["similarity"])
        highest_sim_is_answerable.append(1 if best_pair["answerable"] == "yes" else 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["green" if x == 1 else "red" for x in highest_sim_is_answerable]
    ax.bar(range(len(question_ids)), highest_sim_is_answerable, color=colors, alpha=0.7)
    ax.set_xticks(range(len(question_ids)))
    ax.set_xticklabels(question_ids, rotation=90, fontsize=8)
    ax.set_ylabel("Highest Similarity Paper Answerable? (1=yes, 0=no)", fontsize=10)
    ax.set_title("Does the Most Similar Paper Answer Each Question?", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        Path(output_dir, "highest_similarity_answerability_paper_only.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Visualizations saved to {output_dir}/")


def write_report(stats: Dict, results: Dict, output_path: Path):
    """
    Write a detailed text report of the analysis.
    """
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SIMILARITY-ANSWERABILITY OVERLAP ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total paper-question pairs: {stats['total_pairs']}\n")
        f.write(
            f"Answerable pairs: {stats['answerable_count']} ({stats['answerable_count'] / stats['total_pairs'] * 100:.1f}%)\n"
        )
        f.write(
            f"Not answerable pairs: {stats['not_answerable_count']} ({stats['not_answerable_count'] / stats['total_pairs'] * 100:.1f}%)\n"
        )
        f.write(
            f"Error/unknown pairs: {stats['error_count']} ({stats['error_count'] / stats['total_pairs'] * 100:.1f}%)\n\n"
        )

        # Similarity statistics
        f.write("SIMILARITY SCORE STATISTICS\n")
        f.write("-" * 80 + "\n")

        if "answerable_sim_mean" in stats:
            f.write("\nAnswerable pairs:\n")
            f.write(f"  Mean similarity: {stats['answerable_sim_mean']:.4f}\n")
            f.write(f"  Std deviation: {stats['answerable_sim_std']:.4f}\n")
            f.write(
                f"  Range: {stats['answerable_sim_min']:.4f} - {stats['answerable_sim_max']:.4f}\n"
            )

        if "not_answerable_sim_mean" in stats:
            f.write("\nNot answerable pairs:\n")
            f.write(f"  Mean similarity: {stats['not_answerable_sim_mean']:.4f}\n")
            f.write(f"  Std deviation: {stats['not_answerable_sim_std']:.4f}\n")
            f.write(
                f"  Range: {stats['not_answerable_sim_min']:.4f} - {stats['not_answerable_sim_max']:.4f}\n\n"
            )

        # Difference in means
        if "answerable_sim_mean" in stats and "not_answerable_sim_mean" in stats:
            diff = stats["answerable_sim_mean"] - stats["not_answerable_sim_mean"]
            f.write(f"Difference in means: {diff:.4f}\n")
            f.write("  (positive = answerable pairs have higher similarity)\n\n")

        # Per-question coverage
        f.write("PER-QUESTION COVERAGE\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Questions with at least one answerable paper: {stats['questions_with_answerable_paper']}\n"
        )
        f.write(
            f"Questions without any answerable paper: {stats['questions_without_answerable_paper']}\n\n"
        )

        # Document retrieval task readiness
        total_questions = (
            stats["questions_with_answerable_paper"]
            + stats["questions_without_answerable_paper"]
        )
        coverage_pct = (
            stats["questions_with_answerable_paper"] / total_questions * 100
            if total_questions > 0
            else 0
        )
        f.write(f"Document retrieval task coverage: {coverage_pct:.1f}%\n")
        f.write(
            f"  (i.e., {coverage_pct:.1f}% of questions have at least one paper that can answer them)\n\n"
        )

        # Detailed per-question breakdown
        f.write("=" * 80 + "\n")
        f.write("DETAILED PER-QUESTION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        for question_id in sorted(stats["question_details"].keys()):
            details = stats["question_details"][question_id]
            f.write(f"Question: {question_id}\n")
            f.write(f"  Total papers evaluated: {details['total_papers']}\n")
            f.write(f"  Answerable papers: {details['answerable_count']}\n")
            f.write(
                f"  Has answerable paper: {'YES' if details['has_answerable_paper'] else 'NO'}\n"
            )
            f.write(
                f"  Best overall similarity: {details['best_overall_similarity']:.4f} (paper: {details['best_overall_paper']})\n"
            )

            if details["best_answerable_similarity"] is not None:
                f.write(
                    f"  Best answerable similarity: {details['best_answerable_similarity']:.4f}\n"
                )

                # Check if highest similarity paper is answerable
                if (
                    details["best_answerable_similarity"]
                    == details["best_overall_similarity"]
                ):
                    f.write("  >>> Highest similarity paper IS answerable <<<\n")
                else:
                    f.write("  >>> Highest similarity paper is NOT answerable <<<\n")
            else:
                f.write("  Best answerable similarity: None (no answerable papers)\n")

            f.write("\n")

        # Questions without answerable papers
        if stats["questions_without_answerable_paper"] > 0:
            f.write("=" * 80 + "\n")
            f.write("QUESTIONS WITHOUT ANSWERABLE PAPERS\n")
            f.write("=" * 80 + "\n\n")

            for question_id, details in stats["question_details"].items():
                if not details["has_answerable_paper"]:
                    f.write(f"Question: {question_id}\n")
                    f.write(
                        f"  Best similarity: {details['best_overall_similarity']:.4f} (paper: {details['best_overall_paper']})\n"
                    )
                    f.write(
                        "  Action needed: This question needs at least one answerable paper for document retrieval\n\n"
                    )

    print(f"Report saved to {output_path}")


def main():
    # Paths
    similarity_file = Path(
        "output",
        "paper_question_similarity_values_similarity_aware.txt",
    )
    answerability_file = Path(
        "data",
        "question_answerability_evals_similarity_aware.jsonl",
    )
    output_dir = Path("rag_questions", "output", "similarity_aware")
    report_path = Path(output_dir, "similarity_answerability_overlap_report.txt")

    print("Loading data...")
    similarities = parse_similarity_file(similarity_file)
    answerability = load_answerability_data(answerability_file)

    print(f"Loaded {len(similarities)} papers with similarity scores")
    print(f"Loaded {len(answerability)} answerability evaluations")

    print("\nAnalyzing overlap...")
    results = analyze_overlap(similarities, answerability)

    print("Computing statistics...")
    stats = compute_statistics(results)

    print("\nCreating visualizations...")
    create_visualizations(results, stats, output_dir)

    print("Writing report...")
    write_report(stats, results, report_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total pairs analyzed: {stats['total_pairs']}")
    print(
        f"Answerable: {stats['answerable_count']} ({stats['answerable_count'] / stats['total_pairs'] * 100:.1f}%)"
    )
    print(
        f"Not answerable: {stats['not_answerable_count']} ({stats['not_answerable_count'] / stats['total_pairs'] * 100:.1f}%)"
    )

    if "answerable_sim_mean" in stats and "not_answerable_sim_mean" in stats:
        print(f"\nMean similarity (answerable): {stats['answerable_sim_mean']:.4f}")
        print(
            f"Mean similarity (not answerable): {stats['not_answerable_sim_mean']:.4f}"
        )
        print(
            f"Difference: {stats['answerable_sim_mean'] - stats['not_answerable_sim_mean']:.4f}"
        )

    total_questions = (
        stats["questions_with_answerable_paper"]
        + stats["questions_without_answerable_paper"]
    )
    print(
        f"\nQuestions with answerable paper: {stats['questions_with_answerable_paper']}/{total_questions}"
    )
    print(
        f"Document retrieval coverage: {stats['questions_with_answerable_paper'] / total_questions * 100:.1f}%"
    )

    if stats["questions_without_answerable_paper"] > 0:
        print(
            f"\nWARNING: {stats['questions_without_answerable_paper']} question(s) have NO answerable papers!"
        )
        print("See report for details.")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
