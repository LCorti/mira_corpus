import json
from pathlib import Path

from datasets import load_dataset

"""
Script to load from HF and check that the summarisation data is present correctly.
"""

if __name__ == "__main__":
    ds_name = "dtk-llms/mira_summarization"
    ds = load_dataset(ds_name)
    paper_ids = {}

    for k in ds.keys():
        print(f"{k}: {len(ds[k])} entries.")
        print(f" - Columns: {ds[k].column_names}")
        paper_ids[k] = ds[k]["paper_id"]

    # There should only be two keys: "train" and "test".
    # 1. Check whether "train" is contained in "test".

    intersect = list(set(paper_ids["train"]) & set(paper_ids["test"]))
    print(len(intersect))

    # Create jsonl structure
    data = []
    out_file = Path("summarisation", "summ_corpus_test.jsonl")

    for paper in ds["test"]:
        data.append(
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "full_text": paper["full_text"],
                "human_summary": paper["human_summary"],
                "gpt_summary": paper["gpt_summary"],
            }
        )

    with open(out_file, "a", encoding="utf-8") as fp:
        for paper in data:
            fp.write(json.dumps(paper))
            fp.write("\n")
