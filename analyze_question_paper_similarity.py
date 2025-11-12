"""
Calculate semantic similarity between papers and their generated questions.
Now with:
A) query-side prompt + normalized embeddings (dot product)
B) passage-level scoring (mean of top-3 passage sims per paper)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -------------------------------
# IO helpers
# -------------------------------
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# -------------------------------
# Text utilities
# -------------------------------
def build_doc_text(paper):
    title = paper.get('title', '')
    abstract = paper.get('abstract', paper.get('text', ''))
    # Title + abstract generally works better than abstract alone
    combined = (title + " " + abstract).strip()
    return combined if combined else abstract

def chunk_text_words(text, max_tokens=180, stride=60):
    """
    Simple word-based chunking (token-agnostic but effective for abstracts).
    Produces overlapping chunks so queries can match local passages.
    """
    words = text.split()
    if not words:
        return [""]

    chunks = []
    step = max_tokens - stride
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
        if i + max_tokens >= len(words):
            break
        i += step
    return chunks

def score_query_to_paper(q_emb, passage_embs, topk=3):
    """
    passage_embs: tensor [num_passages, dim] (already normalized)
    q_emb: tensor [dim] (already normalized)
    Returns mean of top-k dot products (cosine when normalized).
    """
    sims = (passage_embs @ q_emb).cpu().numpy()  # [num_passages]
    if sims.size == 0:
        return 0.0
    sims.sort()
    if topk <= 1 or sims.size == 1:
        return float(sims[-1])
    k = min(topk, sims.size)
    return float(np.mean(sims[-k:]))

# -------------------------------
# Paths
# -------------------------------
selected_papers_path = Path("data_scisumm", "selected_papers.jsonl")
gpt_questions_path   = Path("data_scisumm", "gpt_questions_paper_only.jsonl")
heatmap_out          = Path("data_scisumm", "paper_question_similarity_heatmap.png")
txt_out              = Path("data_scisumm", "paper_question_similarity_values.txt")

print("Loading data...")
selected_papers = load_jsonl(selected_papers_path)
gpt_questions   = load_jsonl(gpt_questions_path)

print(f"Loaded {len(selected_papers)} papers")
print(f"Loaded {len(gpt_questions)} question sets")

# Map paper_id -> doc text and questions
paper_texts = {p['paper_id']: build_doc_text(p) for p in selected_papers}
paper_questions = {}
for q in gpt_questions:
    pid = q['paper_id']
    try:
        queries = json.loads(q['gpt_queries'])['queries']
        paper_questions[pid] = [item['text'] for item in queries]
    except Exception as e:
        print(f"Warning: Could not parse queries for {pid}: {e}")

# Keep only IDs that have both
common_paper_ids = sorted(set(paper_texts) & set(paper_questions))
print(f"\nFound {len(common_paper_ids)} papers with both text and questions")
if not common_paper_ids:
    raise SystemExit("Error: No papers found with both text and questions!")

# -------------------------------
# Model (A: query prompt + normalization)
# -------------------------------
model_name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
print(f"\nLoading sentence transformer model ({model_name})")
# trust_remote_code=True so we can use prompt_name="query"
model = SentenceTransformer(model_name, trust_remote_code=True)

# -------------------------------
# Encode documents as passages (B)
# -------------------------------
print("Chunking and encoding paper passages (normalized)...")
paper_passages = {}        # pid -> [passage_texts]
paper_passage_embs = {}    # pid -> tensor [num_passages, dim]

for pid in common_paper_ids:
    doc_text = paper_texts[pid]
    passages = chunk_text_words(doc_text, max_tokens=180, stride=60)
    # If doc is short, ensure at least one chunk
    if not passages:
        passages = [doc_text]
    paper_passages[pid] = passages

    embs = model.encode(
        passages,
        convert_to_tensor=True,
        normalize_embeddings=True  # IMPORTANT: normalize docs
    )
    paper_passage_embs[pid] = embs

# -------------------------------
# Encode queries with query prompt (A)
# -------------------------------
print("Encoding queries with query prompt (normalized)...")
question_embs = {}  # pid -> [q_emb, q_emb, q_emb]
for pid in common_paper_ids:
    qs = paper_questions[pid]
    q_embs = [
        model.encode(
            q_text,
            prompt_name="query",          # IMPORTANT: query-side prompt
            convert_to_tensor=True,
            normalize_embeddings=True     # normalize queries
        )
        for q_text in qs
    ]
    question_embs[pid] = q_embs

# -------------------------------
# Build similarity matrix (mean top-3 over passages)
# -------------------------------
print("\nCalculating similarity scores (mean top-3 over passages)...")
n = len(common_paper_ids)
qpp = 3  # questions per paper
similarity_matrix = np.zeros((n, n * qpp), dtype=float)
row_labels, col_labels = [], []

for i, pid in enumerate(common_paper_ids):
    row_labels.append(pid)
    for j, other_pid in enumerate(common_paper_ids):
        for k, q_emb in enumerate(question_embs[other_pid]):
            col_idx = j * qpp + k
            sim = score_query_to_paper(q_emb, paper_passage_embs[pid], topk=3)
            similarity_matrix[i, col_idx] = sim
            if i == 0:
                col_labels.append(f"{other_pid}_Q{k+1}")

print(f"Similarity matrix shape: {similarity_matrix.shape}")

# -------------------------------
# Heatmap
# -------------------------------
print("\nCreating heatmap...")
plt.figure(figsize=(20, 12))
sns.heatmap(
    similarity_matrix,
    xticklabels=col_labels,
    yticklabels=row_labels,
    cmap='YlOrRd',
    annot=True,
    fmt='.2f',
    cbar_kws={'label': 'Cosine Similarity (mean top-3 passages)'},
    linewidths=0.5,
    linecolor='gray',
    vmin=0,
    vmax=1,
    annot_kws={'fontsize': 6}
)
plt.title('Semantic Similarity (passage-level): Papers vs. Generated Questions', fontsize=16, pad=20)
plt.xlabel('Questions (Paper_ID + Question Number)', fontsize=12)
plt.ylabel('Papers (Paper_ID)', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
heatmap_out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(heatmap_out, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {heatmap_out}")

# -------------------------------
# Save values + summary
# -------------------------------
print("\nSaving similarity values to text file...")
with open(txt_out, 'w', encoding='utf-8') as f:
    f.write("Semantic Similarity (passage-level, mean top-3): Papers vs. Generated Questions\n")
    f.write("=" * 80 + "\n\n")
    f.write("Format: Paper_ID | Question_ID | Similarity_Score\n")
    f.write("-" * 80 + "\n\n")

    for i, pid in enumerate(row_labels):
        f.write(f"\nPaper: {pid}\n")
        f.write("-" * 40 + "\n")
        for j, label in enumerate(col_labels):
            f.write(f"  {label:30s} : {similarity_matrix[i, j]:.4f}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Overall similarity range: {similarity_matrix.min():.4f} - {similarity_matrix.max():.4f}\n")
    f.write(f"Mean similarity: {similarity_matrix.mean():.4f}\n")
    f.write(f"Std deviation: {similarity_matrix.std():.4f}\n")

    # self-similarity = each paper vs its own three questions
    self_sims = []
    for i in range(n):
        for k in range(qpp):
            col_idx = i * qpp + k
            self_sims.append(similarity_matrix[i, col_idx])
    f.write(f"\nSelf-similarity (papers vs their own questions):\n")
    f.write(f"  Mean: {np.mean(self_sims):.4f}\n")
    f.write(f"  Range: {np.min(self_sims):.4f} - {np.max(self_sims):.4f}\n")

    # Top-10 pairs
    f.write("\n" + "=" * 80 + "\n")
    f.write("TOP 10 MOST SIMILAR PAPER-QUESTION PAIRS\n")
    f.write("=" * 80 + "\n\n")
    flat_idx = np.argsort(similarity_matrix.flatten())[::-1][:10]
    for rank, idx in enumerate(flat_idx, 1):
        r = idx // similarity_matrix.shape[1]
        c = idx % similarity_matrix.shape[1]
        f.write(f"{rank:2d}. Paper: {row_labels[r]:15s} <-> Question: {col_labels[c]:30s} | {similarity_matrix[r, c]:.4f}\n")

print(f"Similarity values saved to: {txt_out}")

# Console stats
print("\n=== Statistics ===")
print(f"Overall similarity range: {similarity_matrix.min():.3f} - {similarity_matrix.max():.3f}")
print(f"Mean similarity: {similarity_matrix.mean():.3f}")
print(f"Std deviation: {similarity_matrix.std():.3f}")

self_sims = [similarity_matrix[i, i*qpp + k] for i in range(n) for k in range(qpp)]
print(f"\nSelf-similarity (papers vs their own questions):")
print(f"  Mean: {np.mean(self_sims):.3f}")
print(f"  Range: {np.min(self_sims):.3f} - {np.max(self_sims):.3f}")

print("\nAnalysis complete!")