import json

from pathlib import Path


def load_jsonl(file_path: str | Path) -> list[dict]:
    with open(file_path, "r") as fp:
        data = [json.loads(line) for line in fp]
    return data


def append_to_jsonl(data: dict, file_path: str | Path) -> None:
    with open(file_path, "a") as fp:
        fp.write(json.dumps(data))
        fp.write("\n")


def save_json(data: dict, file_path: str | Path, indent: int = 2) -> None:
    with open(file_path, "w") as fp:
        json.dump(data, fp, indent=indent)


def build_prompt(payload: str, task="summarisation") -> str:
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
