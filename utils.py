import json
import pandas as pd
import random

from pathlib import Path
from typing import Optional

# Handle JSONs


def load_jsonl(file_path: str | Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as fp:
        data = [json.loads(line) for line in fp]
    return data


def append_to_jsonl(data: dict, file_path: str | Path) -> None:
    with open(file_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(data))
        fp.write("\n")


def save_json(data: dict, file_path: str | Path, indent: int = 2) -> None:
    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=indent)


# Prompt builder


def build_prompt(payload: str, task="summarisation") -> str:
    base_prompt = ""
    if task == "summarisation":
        base_prompt = (
            "Summarise the following scientific article from the Association for "
            "Computational Linguistics.\n"
            "== Begin article == \n{}\n== End article =="
        )
    elif task == "simplification":
        base_prompt = (
            "Simplify the following scientific article from the Association"
            " for Computational Linguistics. The target audience is students "
            "knowledgeable in Natural Language Processing.\n"
            "== Begin article == \n{}\n== End article =="
        )
    return base_prompt.format(payload)


# Parquet conversion (for Hugging Face)


def convert_jsonl_to_parquet(
    input_path: str | Path,
    output_path: str | Path,
    task: str,
    num_samples: Optional[int] = None,
) -> pd.DataFrame:
    papers = []

    print(f"Reading from {str(input_path)}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                papers.append(json.loads(line.strip()))

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
        print(
            f"Randomly sampling {num_samples} papers from {len(papers)} total papers..."
        )
        papers = random.sample(papers, num_samples)
        print(f"Selected {len(papers)} papers")
    # Create DataFrame
    df = pd.DataFrame(papers)

    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total papers: {len(df)}")
    print(f"Papers with abstracts: {df['abstract'].notna().sum()}")
    print(f"Papers with human summaries: {df['human_summary'].notna().sum()}")

    if task == "summarisation":
        print(f"Papers with GPT summary: {df['gpt_summary'].notna().sum()}")
    elif task == "simplification":
        print(
            f"Papers with GPT simplification: {df['gpt_simplification'].notna().sum()}"
        )

    # Save to parquet
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)

    print("Conversion completed successfully!")
    return df
