import ollama
import time
import utils

from evaluate import load
from pathlib import Path

if __name__ == "__main__":
    # Load papers
    papers_path = Path(".", "data_scisumm", "cl_scisumm.jsonl")
    papers = utils.load_jsonl(papers_path)
    """
    [{
        "paper_id: "<ACL_paper_id>",
        "text": "...",
        "abstract": "...",
        "human": "..."
    }]
    """
    # Load summaries
    summaries_path = Path(".", "data_scisumm", "gpt_summaries.jsonl")
    try:
        summaries = utils.load_jsonl(summaries_path)
        processed_ids = [s["paper_id"] for s in summaries]
    except Exception as _:
        print(f"File {str(summaries_path)} not found. Starting from scratch.")
        processed_ids = []

    # Generation options
    gen_options = {"num_ctx": 130000}
    bertscore = load("bertscore")

    for paper in papers:
        if paper["paper_id"] in processed_ids:
            print(f"Already processed paper {paper['paper_id']}.")
            continue

        print(f"Summarising paper {paper['paper_id']}...")
        to_save = paper.copy()

        # Get summary from gpt-oss
        prompt = utils.build_prompt(paper["text"], task="summarisation")
        # This is a very hacky solution but for now it should do the job.
        try:
            response = ollama.generate(
                model="gpt-oss:120b-cloud", prompt=prompt, options=gen_options
            )
        except Exception as _:
            sleep_mins = 30
            print(f"Hit Ollama API (hourly) limits. Sleeping for {sleep_mins} mins.")
            time.sleep(60 * sleep_mins)
            print("Ending timeout. Trying again.")
            response = ollama.generate(
                model="gpt-oss:120b-cloud", prompt=prompt, options=gen_options
            )
        to_save["gpt_summary"] = response["response"]

        # Compute generation performance with human summary
        scores = bertscore.compute(
            predictions=[to_save["gpt_summary"]],
            references=[paper["human"]],
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
        )
        to_save["f1"] = scores["f1"][0]
        print(f"BERTScore (F1) between gpt-oss and human: {to_save['f1']:.2f}.")

        # Save gpt summary + metric
        utils.append_to_jsonl(to_save, summaries_path)
