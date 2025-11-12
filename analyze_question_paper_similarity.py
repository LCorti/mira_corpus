"""
Calculate semantic similarity between papers and their generated questions.
Creates a heatmap showing how related each question is to each paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load the data
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Load papers and questions
selected_papers_path = Path("data_scisumm", "selected_papers.jsonl")
gpt_questions_path = Path("data_scisumm", "gpt_questions_paper_only.jsonl")

print("Loading data...")
selected_papers = load_jsonl(selected_papers_path)
gpt_questions = load_jsonl(gpt_questions_path)

print(f"Loaded {len(selected_papers)} papers")
print(f"Loaded {len(gpt_questions)} question sets")

# Create a mapping of paper_id to paper text and questions
paper_texts = {p['paper_id']: p['text'] for p in selected_papers}
paper_questions = {}

for q in gpt_questions:
    paper_id = q['paper_id']
    try:
        queries = json.loads(q['gpt_queries'])['queries']
        paper_questions[paper_id] = [query['text'] for query in queries]
    except:
        print(f"Warning: Could not parse queries for {paper_id}")
        continue

# Filter to only papers that have both text and questions
common_paper_ids = sorted(set(paper_texts.keys()) & set(paper_questions.keys()))
print(f"\nFound {len(common_paper_ids)} papers with both text and questions")

if len(common_paper_ids) == 0:
    print("Error: No papers found with both text and questions!")
    exit(1)

# Load sentence transformer model
model_name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
print(f"\nLoading sentence transformer model ({model_name})")
model = SentenceTransformer(model_name, trust_remote_code=True)

# Prepare all texts for encoding
print("Preparing texts for encoding...")
all_paper_texts = []
all_questions = []

for paper_id in common_paper_ids:
    text = paper_texts[paper_id]
    # Truncate very long texts to avoid memory issues
    if len(text) > 32000:
        print(f"Truncating text for paper {paper_id} from {len(text)} to 32000 characters")
        text = text[:32000]
    all_paper_texts.append(text)
    all_questions.extend(paper_questions[paper_id])

# Encode all papers and questions in batch
print(f"Encoding {len(all_paper_texts)} papers...")
paper_embeddings = model.encode(all_paper_texts, show_progress_bar=True)

print(f"Encoding {len(all_questions)} questions...")
question_embeddings = model.encode(all_questions, show_progress_bar=True)

# Calculate similarity matrix using model.similarity()
print("\nCalculating similarity matrix...")
similarity_matrix = model.similarity(paper_embeddings, question_embeddings).cpu().numpy()

print(f"Similarity matrix shape: {similarity_matrix.shape}")

# Create labels
row_labels = common_paper_ids
col_labels = []
for paper_id in common_paper_ids:
    for k in range(3):  # 3 questions per paper
        col_labels.append(f"{paper_id}_Q{k+1}")

# Create heatmap
print("\nCreating heatmap...")
plt.figure(figsize=(20, 12))

# Create heatmap with better visibility and annotations
sns.heatmap(
    similarity_matrix,
    xticklabels=col_labels,
    yticklabels=row_labels,
    cmap='YlOrRd',
    annot=True,
    fmt='.2f',
    cbar_kws={'label': 'Cosine Similarity'},
    linewidths=0.5,
    linecolor='gray',
    vmin=0,
    vmax=1,
    annot_kws={'fontsize': 6}
)

plt.title('Semantic Similarity: Papers vs. Generated Questions', fontsize=16, pad=20)
plt.xlabel('Questions (Paper_ID + Question Number)', fontsize=12)
plt.ylabel('Papers (Paper_ID)', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

# Save the figure
output_path = Path("data_scisumm", "paper_question_similarity_heatmap.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {output_path}")

# Save similarity values to text file
print("\nSaving similarity values to text file...")
txt_output_path = Path("data_scisumm", "paper_question_similarity_values.txt")
with open(txt_output_path, 'w') as f:
    f.write("Semantic Similarity: Papers vs. Generated Questions\n")
    f.write("=" * 80 + "\n\n")
    
    # Write header
    f.write("Format: Paper_ID | Question_ID | Similarity_Score\n")
    f.write("-" * 80 + "\n\n")
    
    # Write all similarity values
    for i, paper_id in enumerate(row_labels):
        f.write(f"\nPaper: {paper_id}\n")
        f.write("-" * 40 + "\n")
        for j, question_label in enumerate(col_labels):
            similarity = similarity_matrix[i, j]
            f.write(f"  {question_label:30s} : {similarity:.4f}\n")
    
    # Write summary statistics
    f.write("\n" + "=" * 80 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Overall similarity range: {similarity_matrix.min():.4f} - {similarity_matrix.max():.4f}\n")
    f.write(f"Mean similarity: {similarity_matrix.mean():.4f}\n")
    f.write(f"Std deviation: {similarity_matrix.std():.4f}\n")
    
    # Self-similarity stats
    self_similarities_temp = []
    for i, paper_id in enumerate(row_labels):
        for k in range(3):
            col_idx = i * 3 + k
            self_similarities_temp.append(similarity_matrix[i, col_idx])
    
    f.write(f"\nSelf-similarity (papers vs their own questions):\n")
    f.write(f"  Mean: {np.mean(self_similarities_temp):.4f}\n")
    f.write(f"  Range: {np.min(self_similarities_temp):.4f} - {np.max(self_similarities_temp):.4f}\n")
    
    # Top 10 pairs
    f.write("\n" + "=" * 80 + "\n")
    f.write("TOP 10 MOST SIMILAR PAPER-QUESTION PAIRS\n")
    f.write("=" * 80 + "\n\n")
    flat_indices_temp = np.argsort(similarity_matrix.flatten())[::-1][:10]
    for rank, idx in enumerate(flat_indices_temp, 1):
        row = idx // similarity_matrix.shape[1]
        col = idx % similarity_matrix.shape[1]
        paper_id_temp = row_labels[row]
        question_label_temp = col_labels[col]
        similarity_temp = similarity_matrix[row, col]
        f.write(f"{rank:2d}. Paper: {paper_id_temp:15s} <-> Question: {question_label_temp:30s} | {similarity_temp:.4f}\n")

print(f"Similarity values saved to: {txt_output_path}")

# Print statistics
print("\n=== Statistics ===")
print(f"Overall similarity range: {similarity_matrix.min():.3f} - {similarity_matrix.max():.3f}")
print(f"Mean similarity: {similarity_matrix.mean():.3f}")
print(f"Std deviation: {similarity_matrix.std():.3f}")

# Calculate self-similarity (diagonal elements for each paper's own questions)
self_similarities = []
for i, paper_id in enumerate(common_paper_ids):
    for k in range(3):  # 3 questions per paper
        col_idx = i * 3 + k
        self_similarities.append(similarity_matrix[i, col_idx])

print(f"\nSelf-similarity (papers vs their own questions):")
print(f"  Mean: {np.mean(self_similarities):.3f}")
print(f"  Range: {np.min(self_similarities):.3f} - {np.max(self_similarities):.3f}")

# Calculate and display top similar paper-question pairs
print("\n=== Top 10 Most Similar Paper-Question Pairs ===")
flat_indices = np.argsort(similarity_matrix.flatten())[::-1][:10]
for rank, idx in enumerate(flat_indices, 1):
    row = idx // similarity_matrix.shape[1]
    col = idx % similarity_matrix.shape[1]
    paper_id = row_labels[row]
    question_label = col_labels[col]
    similarity = similarity_matrix[row, col]
    print(f"{rank}. Paper: {paper_id} <-> Question: {question_label} | Similarity: {similarity:.3f}")

print("\nAnalysis complete!")
