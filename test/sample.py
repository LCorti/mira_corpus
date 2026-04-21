import numpy as np
import utils

from pathlib import Path

if __name__ == "__main__":
    # Load summaries to sample
    summaries_path = Path("..", "summarisation", "corpus_summarisation.jsonl")
    summaries = utils.load_jsonl(summaries_path)
    # Sample
    sample = np.random.choice(summaries, size=50, replace=False).tolist()
    # Remove useless keys and save
    sample_path = Path("summ_sample.jsonl")
    for s in sample:
        s.pop("text")
        s.pop("abstract")
        s.pop("f1")
    utils.save_json(sample, sample_path, indent=4)

    # Load simplifications to sample
    simpl_path = Path("..", "simplification", "corpus_summarisation.jsonl")
    simplifications = utils.load_jsonl(simpl_path)
    sample = np.random.choice(summaries, size=50, replace=False).tolist()
    sample_path = Path("simple_sample.jsonl")
    for s in sample:
        s.pop("text")
        s.pop("abstract")
        s.pop("f1")
    utils.save_json(sample, sample_path, indent=4)
