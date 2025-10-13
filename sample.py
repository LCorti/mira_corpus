import numpy as np
import utils

from pathlib import Path

if __name__ == "__main__":
    # Load data to sample
    summaries_path = Path(".", "data_scisumm", "gpt_summaries.jsonl")
    summaries = utils.load_jsonl(summaries_path)
    # Sample
    sample = np.random.choice(summaries, size=50, replace=False).tolist()
    # Remove useless keys and save
    sample_path = Path("summ_sample_2.jsonl")
    for s in sample:
        s.pop("text")
        s.pop("abstract")
        s.pop("f1")
    utils.save_json(sample, sample_path, indent=4)
