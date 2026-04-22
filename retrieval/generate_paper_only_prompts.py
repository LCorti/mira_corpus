#!/usr/bin/env python3
"""
Generate prompts for query generation based on selected papers only (no neighbors).
Creates a new JSONL file with paper_id and prompt attributes.
"""

import json
from pathlib import Path


def create_prompt(paper_data: dict) -> str:
    """
    Create a prompt for generating retrieval queries for a target paper.

    Args:
        paper_data: Dict containing paper metadata

    Returns:
        The formatted prompt string
    """

    # Format the input JSON with paper metadata
    input_data = {
        "title": paper_data.get("title", ""),
        "authors": paper_data.get("authors", []),
        "venue": paper_data.get("venue", ""),
        "year": paper_data.get("year", ""),
        "abstract": paper_data.get("abstract", ""),
        "keywords": paper_data.get("keywords", []),
        "sections": paper_data.get("sections", []),
    }

    # Remove empty optional fields
    if not input_data["keywords"]:
        del input_data["keywords"]
    if not input_data["sections"]:
        del input_data["sections"]

    input_json = json.dumps(input_data, indent=2)

    # Create the full prompt
    prompt = f"""You are a research assistant generating search queries for ACL papers.

Goal:
Write EXACTLY THREE natural-language questions that could be used to retrieve this paper in a semantic-search system.
Use only the information in the TARGET paper (title and abstract).
Each question should be ≤ 20 words and end with a question mark.

Input:
{input_json}

Output format:
{{
    "queries": [
        {{"id":"Q1","breadth":"high_recall","text":"..."}},
        {{"id":"Q2","breadth":"balanced","text":"..."}},
        {{"id":"Q3","breadth":"high_precision_distinctive","text":"..."}}
    ]
}}

Guidelines:
- **Q1 (high recall)**: Broad, field-defining. Use 1-2 general domain terms from the paper's title or abstract.
- **Q2 (balanced)**: Moderate specificity. Combine the main task or method with one contextual detail (dataset, language, or evaluation).
- **Q3 (high precision / distinctive)**: Include distinctive technical or methodological phrases that uniquely identify this paper.

Rules:
- No invented results, datasets, or model names.
- Avoid filler verbs like *propose*, *present*, *investigate*, *paper*.
- Use fluent, retrieval-style questions suitable for semantic search.
- Do not start questions with “Does the paper”, “Can the system”, or “What evaluation”.
- Prefer formulations beginning with “How”, “Which”, or “What … in”.
"""

    return prompt


if __name__ == "__main__":
    # Read the selected_papers.jsonl file
    input_file = Path(".", "data", "selected_papers.jsonl")
    output_file = Path(".", "data", "prompts_paper_only.jsonl")

    print(f"Reading papers from: {input_file}")

    # Process each paper
    paper_count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        with open(input_file, "r", encoding="utf-8") as in_f:
            for line_num, line in enumerate(in_f, 1):
                try:
                    paper_data = json.loads(line)
                    paper_id = paper_data.get("paper_id")

                    if not paper_id:
                        print(f"Warning: Line {line_num} missing paper_id, skipping")
                        continue

                    paper_count += 1
                    print(f"Processing paper {paper_count}: {paper_id}...", end="\r")

                    # Create prompt based on paper data only
                    prompt = create_prompt(paper_data)

                    # Write to output
                    output_entry = {"paper_id": paper_id, "prompt": prompt}

                    out_f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue

    print(f"\nDone! Processed {paper_count} papers. Output written to: {output_file}")
