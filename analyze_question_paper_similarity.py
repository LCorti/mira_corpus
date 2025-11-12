"""
Calculate semantic similarity between papers and their generated questions.
Creates a heatmap showing how related each question is to each paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

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
paper_texts = {p['paper_id']: p.get('abstract', p.get('text', '')) for p in selected_papers}
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

# Calculate embeddings for papers (using abstracts)
print("Encoding paper texts...")
paper_embeddings = {}
for paper_id in common_paper_ids:
    text = paper_texts[paper_id]
    # Truncate very long texts to avoid memory issues
    if len(text) > 5000:
        text = text[:5000]
    paper_embeddings[paper_id] = model.encode(text, convert_to_tensor=True)

# Calculate embeddings for questions
print("Encoding questions...")
question_embeddings = {}
for paper_id in common_paper_ids:
    questions = paper_questions[paper_id]
    question_embeddings[paper_id] = [
        model.encode(q, convert_to_tensor=True) for q in questions
    ]

# Calculate similarity matrix
print("\nCalculating similarity scores...")
n_papers = len(common_paper_ids)
n_questions_per_paper = 3  # Q1, Q2, Q3

# Create similarity matrix: rows are papers, columns are questions
similarity_matrix = np.zeros((n_papers, n_papers * n_questions_per_paper))

row_labels = []
col_labels = []

for i, paper_id in enumerate(common_paper_ids):
    row_labels.append(paper_id)
    paper_emb = paper_embeddings[paper_id]
    
    for j, other_paper_id in enumerate(common_paper_ids):
        questions = paper_questions[other_paper_id]
        question_embs = question_embeddings[other_paper_id]
        
        for k, (q_text, q_emb) in enumerate(zip(questions, question_embs)):
            col_idx = j * n_questions_per_paper + k
            
            # Calculate cosine similarity
            similarity = util.cos_sim(paper_emb, q_emb).item()
            similarity_matrix[i, col_idx] = similarity
            
            # Add column label only once
            if i == 0:
                col_labels.append(f"{other_paper_id}_Q{k+1}")

print(f"Similarity matrix shape: {similarity_matrix.shape}")

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
