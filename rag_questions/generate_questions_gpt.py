import ollama
import time
import utils

from pathlib import Path

if __name__ == "__main__":
    # Load papers
    # prompt_path = Path(".", "data", "prompts_similarity_aware.jsonl")
    prompt_path = Path(".", "data", "prompts_paper_only.jsonl")
    prompt_objects = utils.load_jsonl(prompt_path)
    """
    [{
        "paper_id": "<paper id>",
        "role": "user",
        "prompt": "<full paper text>"
    }]
    """
    # Load generated questions
    # questions_path = Path(".", "data", "gpt_questions_similarity_aware.jsonl")
    questions_path = Path(".", "data", "gpt_questions_paper_only.jsonl")
    try:
        questions = utils.load_jsonl(questions_path)
        processed_ids = [q["paper_id"] for q in questions]
    except Exception as _:
        print(f"File {str(questions_path)} not found. Starting from scratch.")
        processed_ids = []

    # Generation options
    gen_options = {"num_ctx": 130000}

    for prompt_object in prompt_objects:
        if prompt_object["paper_id"] in processed_ids:
            print(f"Already processed paper {prompt_object['paper_id']}.")
            continue

        print(f"Generating questions for paper {prompt_object['paper_id']}...")
        to_save = prompt_object.copy()

        # Get simplification from gpt-oss
        prompt = prompt_object["prompt"]

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
        to_save["gpt_queries"] = response["response"]

        # Save gpt simplification + metric
        utils.append_to_jsonl(to_save, questions_path)
