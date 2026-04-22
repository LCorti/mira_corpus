import argparse
import json
import ollama
import os
import sys
import time

from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
import utils


def check_ollama_auth():
    """Check if Ollama API key is set."""
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print("\n" + "=" * 80)
        print("ERROR: OLLAMA_API_KEY environment variable is not set!")
        print("=" * 80)
        print("\nTo set it in PowerShell:")
        print("  For current session:")
        print('    $env:OLLAMA_API_KEY = "your_api_key"')
        print("\n  For permanent (requires terminal restart):")
        print(
            "    [System.Environment]::SetEnvironmentVariable('OLLAMA_API_KEY', 'your_api_key', 'User')"
        )
        print("\nThen run the script again.")
        print("=" * 80 + "\n")
        sys.exit(1)


def build_answerability_prompt(
    paper_text: str, question: str, paper_id: str, question_id: str
) -> str:
    """
    Build the prompt for evaluating question answerability.

    Separates the prompt into three parts:
    1. Goal/task description
    2. Input/query (the question)
    3. Paper/context (the paper text)
    """
    prompt = f"""You are an expert research assistant evaluating whether a research question can be answered using a given scientific paper.

GOAL:
Determine if the question below can be answered using the provided paper. Provide a structured analysis.

INPUT/QUERY:
Question ID: {question_id}
Question: {question}

PAPER/CONTEXT:
Paper ID: {paper_id}
{paper_text}

TASK:
Analyze whether the paper can answer the question and provide your response in the following JSON format:

{{
    "answerable": "yes" or "no",
    "analysis": {{
        "if_yes": {{
            "sections": ["list of section/subsection names where answer is found"],
            "reasoning": "Step-by-step explanation of how the paper answers the question, referencing specific sections"
        }},
        "if_no": {{
            "missing_information": "What specific information is needed but not present in the paper",
            "reasoning": "Step-by-step explanation of why the paper cannot answer the question"
        }}
    }}
}}

IMPORTANT:
- Set "answerable" to "yes" ONLY if the paper directly addresses the question with sufficient detail
- If answerable="yes", fill ONLY the "if_yes" fields; leave "if_no" empty
- If answerable="no", fill ONLY the "if_no" fields; leave "if_yes" empty
- For sections, use the actual section/subsection names from the paper (e.g., "SECTION 2: Methods", "3.1 Experimental Setup")
- If section names are not clearly marked, describe the location (e.g., "Introduction paragraph 2", "Discussion of evaluation results")
- Provide detailed chain-of-thought reasoning in both cases

Your response (JSON only, no additional text):"""

    return prompt


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the LLM response, extracting JSON.

    Returns the parsed JSON or an error dictionary if parsing fails.
    """
    try:
        # Try to extract JSON if it's embedded in other text
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return {"error": "No JSON found in response", "raw_response": response_text}
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parsing failed: {str(e)}",
            "raw_response": response_text,
        }


def evaluate_pair(
    paper_id: str,
    paper_text: str,
    question_id: str,
    question: str,
    gen_options: Dict[str, Any],
    client: ollama.Client,
) -> Dict[str, Any]:
    """
    Evaluate a single paper-question pair.

    Returns a dictionary with the evaluation results.
    """
    prompt = build_answerability_prompt(paper_text, question, paper_id, question_id)

    # Try to get response from Ollama, with retry logic for rate limits
    try:
        response = client.generate(
            model="gpt-oss:120b", prompt=prompt, options=gen_options
        )
    except Exception as e:
        sleep_mins = 30
        tqdm.write(
            f"    Hit Ollama API limits. Sleeping for {sleep_mins} mins. Error: {str(e)}"
        )
        time.sleep(60 * sleep_mins)
        tqdm.write("    Ending timeout. Trying again.")
        try:
            response = client.generate(
                model="gpt-oss:120b", prompt=prompt, options=gen_options
            )
        except Exception as e2:
            tqdm.write(f"    Second attempt failed: {str(e2)}")
            raise e2

    # Parse the response
    parsed_response = parse_llm_response(response["response"])

    return {
        "paper_id": paper_id,
        "question_id": question_id,
        "question": question,
        "llm_response": parsed_response,
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate question answerability across papers"
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process papers in reverse order (for parallel processing on multiple machines)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start processing from this paper index (0-based)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Stop processing at this paper index (exclusive)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Hard stop: only process this many papers from the list (useful for splitting work across machines)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help='Suffix to add to output filename (e.g., "_machine2")',
    )
    args = parser.parse_args()

    # Check API key is set
    check_ollama_auth()

    # Initialize Ollama client with API key
    api_key = os.environ.get("OLLAMA_API_KEY")
    client = ollama.Client(
        host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"}
    )

    # Load papers and questions
    script_dir = Path(__file__).parent
    # selected_papers_path = script_dir / "data" / "selected_papers.jsonl"
    selected_papers_path = Path(script_dir.parent, "raw_data", "cl_scisumm.jsonl")
    gpt_questions_path = Path(
        script_dir, "data", "gpt_questions_similarity_aware.jsonl"
    )

    # Add suffix to output filename if specified
    output_filename = (
        f"question_answerability_evals_similarity_aware_full{args.output_suffix}.jsonl"
    )
    output_path = Path(script_dir, "data", output_filename)

    print("Loading data...")
    selected_papers = utils.load_jsonl(selected_papers_path)
    gpt_questions = utils.load_jsonl(gpt_questions_path)

    print(f"Loaded {len(selected_papers)} papers")
    print(f"Loaded {len(gpt_questions)} question sets")

    # Create mappings
    paper_texts = {p["paper_id"]: p["text"] for p in selected_papers}
    paper_questions = {}

    for q in gpt_questions:
        paper_id = q["paper_id"]
        try:
            queries = json.loads(q["gpt_queries"])["queries"]
            paper_questions[paper_id] = [
                (f"{paper_id}_{query['id']}", query["text"]) for query in queries
            ]
        except Exception as e:
            print(f"Warning: Could not parse queries for {paper_id}: {e}")
            continue
    print(f"Mapped questions to {len(paper_questions)} papers")

    # Load already processed pairs if they exist
    processed_pairs = set()
    if output_path.exists():
        print(f"\nFound existing output file: {output_path}")
        existing_results = utils.load_jsonl(output_path)
        processed_pairs = {(r["paper_id"], r["question_id"]) for r in existing_results}
        print(f"Already processed {len(processed_pairs)} paper-question pairs")

    # Generation options for Ollama
    gen_options = {"num_ctx": 130000}

    # Get all paper IDs (for evaluation) and all papers with questions
    all_paper_ids = sorted(paper_texts.keys())
    papers_with_questions = sorted(paper_questions.keys())

    # Apply reverse order if requested
    if args.reverse:
        all_paper_ids = list(reversed(all_paper_ids))
        print("Processing papers in REVERSE order")

    # Apply index range if specified
    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else len(all_paper_ids)
        all_paper_ids = all_paper_ids[start:end]
        print(
            f"Processing papers from index {start} to {end} (total: {len(all_paper_ids)} papers)"
        )

    # Apply limit if specified (hard stop after K papers)
    if args.limit is not None:
        all_paper_ids = all_paper_ids[: args.limit]
        print(f"Hard limit applied: will process only {len(all_paper_ids)} papers")

    print(f"\nFound {len(all_paper_ids)} papers with text to process")
    print(f"Found {len(papers_with_questions)} papers with questions")

    # Build full matrix of evaluations to do
    total_pairs = 0
    for paper_id in all_paper_ids:
        for question_paper_id in papers_with_questions:
            total_pairs += len(paper_questions[question_paper_id])

    print(f"Total paper-question pairs to evaluate: {total_pairs}")
    print(f"Remaining pairs to process: {total_pairs - len(processed_pairs)}")

    if len(processed_pairs) == total_pairs:
        print("\nAll pairs already processed!")
        return

    # Process all paper-question pairs (full matrix)
    remaining_pairs = total_pairs - len(processed_pairs)

    with tqdm(total=remaining_pairs, desc="Evaluating pairs", unit="pair") as pbar:
        for paper_id in all_paper_ids:
            paper_text = paper_texts[paper_id]

            # Truncate very long papers to avoid memory issues
            if len(paper_text) > 32000:
                tqdm.write(
                    f"Truncating paper {paper_id} from {len(paper_text)} to 32000 characters"
                )
                paper_text = paper_text[:32000]

            # For each paper, evaluate against ALL questions from ALL papers
            for question_paper_id in papers_with_questions:
                questions = paper_questions[question_paper_id]

                for question_id, question_text in questions:
                    # Skip if already processed
                    if (paper_id, question_id) in processed_pairs:
                        continue

                    pbar.set_postfix_str(f"Paper: {paper_id}, Q: {question_id}")

                    try:
                        # Evaluate this pair
                        result = evaluate_pair(
                            paper_id=paper_id,
                            paper_text=paper_text,
                            question_id=question_id,
                            question=question_text,
                            gen_options=gen_options,
                            client=client,
                        )
                    except Exception as e:
                        tqdm.write(f"  ERROR during evaluation: {str(e)}")
                        raise e

                    # Save result immediately
                    utils.append_to_jsonl(result, output_path)
                    processed_pairs.add((paper_id, question_id))

                    # Print summary
                    if "error" not in result["llm_response"]:
                        answerable = result["llm_response"].get("answerable", "unknown")
                        tqdm.write(f"  {paper_id} vs {question_id}: {answerable}")
                    else:
                        tqdm.write(
                            f"  {paper_id} vs {question_id}: ERROR - {result['llm_response']['error']}"
                        )

                    pbar.update(1)

    print(f"\n{'=' * 80}")
    print("Evaluation complete!")
    print(f"Total pairs evaluated: {total_pairs}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
