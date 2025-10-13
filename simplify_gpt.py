import ollama
import time
import utils

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
    # Load simplified papers
    simple_papers_path = Path(".", "data_scisumm", "gpt_simplified.jsonl")
    try:
        simple_papers = utils.load_jsonl(simple_papers_path)
        processed_ids = [s["paper_id"] for s in simple_papers]
    except Exception as _:
        print(f"File {str(simple_papers_path)} not found. Starting from scratch.")
        processed_ids = []

    # Generation options
    gen_options = {"num_ctx": 130000}

    for paper in papers:
        if paper["paper_id"] in processed_ids:
            print(f"Already processed paper {paper['paper_id']}.")
            continue

        print(f"Simplifying paper {paper['paper_id']}...")
        to_save = paper.copy()

        # Get simplification from gpt-oss
        prompt = utils.build_prompt(paper["text"], task="simplification")
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
        to_save["gpt_simple"] = response["response"]

        # Save gpt simplification + metric
        utils.append_to_jsonl(to_save, simple_papers_path)
