from pathlib import Path

from utils import convert_jsonl_to_parquet

if __name__ == "__main__":
    task = "simplification"
    base_dir = Path(task)
    input_path = Path(base_dir, f"{task}_corpus.jsonl").resolve()
    output_path = Path(base_dir, "parquet", f"{task}_corpus.parquet").resolve()
    df = convert_jsonl_to_parquet(input_path, output_path, task=task)
