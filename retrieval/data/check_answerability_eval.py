import json

file_names = [
    "question_answerability_evals_similarity_aware.jsonl",
]


def load_jsonl(fn):
    data = []
    with open(fn, "r") as fp:
        for line in fp.readlines():
            data.append(json.loads(line))
    return data


all_ids = set()
for fn in file_names:
    curr = load_jsonl(fn)
    fn_short = fn.replace("question_answerability_evaluations_", "")
    # Create map with paper IDs
    curr_ids = set()
    for paper in curr:
        curr_ids.add(paper["paper_id"])
    all_ids.update(curr_ids)

    print(f"Checking {fn_short}: {len(curr)} entries.")
    print(f" - Unique paper ids: {len(curr_ids)}.")
    print("=" * 40)
    print(f"All unique ids: {len(all_ids)}")
    print("=" * 40)
