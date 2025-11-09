import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import utils

def load_similarity_data(embedded_papers_path):
    """Load the embedded papers with similarity scores."""
    try:
        return utils.load_jsonl(embedded_papers_path)
    except FileNotFoundError:
        print(f"File {embedded_papers_path} not found. Please run semantic_similarity.py first.")
        return []

def extract_similarity_matrix(embedded_papers):
    """Extract similarity matrix from embedded papers data."""
    if not embedded_papers:
        return None, []
    
    # Get all unique paper IDs
    paper_ids = []
    for paper in embedded_papers:
        paper_ids.append(paper['paper_id'])
        # Also add any paper IDs that appear in similarity_scores but not as main papers
        for other_id in paper['similarity_scores'].keys():
            if other_id not in paper_ids:
                paper_ids.append(other_id)
    
    paper_ids = sorted(list(set(paper_ids)))
    n_papers = len(paper_ids)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n_papers, n_papers))
    
    # Fill the matrix
    for paper in embedded_papers:
        i = paper_ids.index(paper['paper_id'])
        
        for other_id, similarity in paper['similarity_scores'].items():
            j = paper_ids.index(other_id)
            # Convert tensor to float if needed
            if hasattr(similarity, 'item'):
                similarity = similarity.item()
            similarity_matrix[i][j] = similarity
            # Make matrix symmetric (since similarity should be symmetric)
            similarity_matrix[j][i] = similarity
    
    return similarity_matrix, paper_ids

def create_paper_labels(paper_ids, papers_data, use_short_labels=False, max_length=50):
    """Create readable labels for papers using abstracts or titles."""
    if use_short_labels:
        # Just return the paper IDs for cleaner visualization
        return paper_ids
    
    labels = []
    
    # Create a lookup dict for paper data
    papers_lookup = {paper['paper_id']: paper for paper in papers_data}
    
    for paper_id in paper_ids:
        if paper_id in papers_lookup:
            paper = papers_lookup[paper_id]
            
            # Try to extract a meaningful short description
            if 'abstract' in paper and paper['abstract']:
                # Use first sentence of abstract
                abstract = paper['abstract']
                first_sentence = abstract.split('.')[0].strip()
                if len(first_sentence) > max_length:
                    label = first_sentence[:max_length] + "..."
                else:
                    label = first_sentence
            else:
                # Fall back to paper ID
                label = paper_id
                
            # Add paper ID for reference
            label = f"{paper_id}: {label}"
        else:
            label = paper_id
            
        labels.append(label)
    
    return labels

def plot_similarity_heatmap(similarity_matrix, paper_labels, title="Paper Similarity Heatmap", highlight_threshold=0.85):
    """Create a heatmap visualization of paper similarities."""
    # Determine figure size based on number of papers
    n_papers = len(paper_labels)
    if n_papers > 20:
        fig_size = (16, 14)
        font_size = 8
    elif n_papers > 10:
        fig_size = (14, 12)
        font_size = 10
    else:
        fig_size = (12, 10)
        font_size = 12
    
    plt.figure(figsize=fig_size)
    
    # For large matrices, don't show individual values
    show_annotations = n_papers <= 15
    
    # Create heatmap
    zero_mask = similarity_matrix == 0  # Mask zero values for better visualization
    
    # Mask lower triangle and diagonal to avoid duplicates and self-similarity
    lower_triangle_mask = np.tril(np.ones_like(similarity_matrix, dtype=bool))
    mask = zero_mask | lower_triangle_mask
    
    # Create the base heatmap
    ax = sns.heatmap(similarity_matrix, 
                    annot=show_annotations, 
                    fmt='.3f' if show_annotations else '',
                    cmap='viridis', 
                    square=True,
                    xticklabels=paper_labels,
                    yticklabels=paper_labels,
                    mask=mask,
                    cbar_kws={'label': 'Similarity Score'})
    
    # Highlight cells with similarity >= threshold
    if highlight_threshold > 0:
        high_sim_mask = (similarity_matrix >= highlight_threshold) & (~mask)
        
        # Add red borders to high similarity cells
        for i in range(n_papers):
            for j in range(n_papers):
                if high_sim_mask[i, j]:
                    # Add a thick red border around the cell
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
        
        # Count high similarity pairs (excluding diagonal and lower triangle)
        high_sim_count = np.sum(high_sim_mask)
        if high_sim_count > 0:
            title += f"\n(Red borders: {high_sim_count} pairs ≥ {highlight_threshold})"
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Papers', fontsize=12)
    plt.ylabel('Papers', fontsize=12)
    
    # Adjust tick parameters based on number of papers
    if n_papers > 20:
        plt.xticks(rotation=90, ha='center', fontsize=font_size)
        plt.yticks(rotation=0, fontsize=font_size)
    else:
        plt.xticks(rotation=45, ha='right', fontsize=font_size)
        plt.yticks(rotation=0, fontsize=font_size)
    
    plt.tight_layout()
    return plt.gcf()

def plot_similarity_distribution(similarity_matrix):
    """Plot distribution of similarity scores."""
    # Get upper triangle (excluding diagonal) to avoid duplicate pairs
    upper_triangle = np.triu(similarity_matrix, k=1)
    similarities = upper_triangle[upper_triangle > 0]  # Remove zeros
    
    if len(similarities) == 0:
        print("No similarity data to plot.")
        return None
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(similarities)
    plt.ylabel('Similarity Score')
    plt.title('Similarity Score Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()

def create_similarity_summary(similarity_matrix, paper_labels):
    """Create a text summary of similarity results."""
    n = len(paper_labels)
    if n == 0:
        return "No data available."
    
    # Get upper triangle (excluding diagonal)
    upper_triangle = np.triu(similarity_matrix, k=1)
    similarities = upper_triangle[upper_triangle > 0]
    
    if len(similarities) == 0:
        return "No similarity data available."
    
    summary = f"""
Similarity Analysis Summary:
===========================
Number of papers: {n}
Number of paper pairs: {len(similarities)}
Average similarity: {np.mean(similarities):.4f}
Median similarity: {np.median(similarities):.4f}
Min similarity: {np.min(similarities):.4f}
Max similarity: {np.max(similarities):.4f}
Standard deviation: {np.std(similarities):.4f}

Most similar pairs:
"""
    
    # Find most similar pairs
    indices = np.unravel_index(np.argsort(-upper_triangle, axis=None), upper_triangle.shape)
    top_pairs = []
    
    for idx in range(min(5, len(similarities))):
        i, j = indices[0][idx], indices[1][idx]
        if upper_triangle[i, j] > 0:
            score = upper_triangle[i, j]
            paper1 = paper_labels[i].split(':')[0]  # Just the ID
            paper2 = paper_labels[j].split(':')[0]  # Just the ID
            top_pairs.append(f"  {paper1} ↔ {paper2}: {score:.4f}")
    
    summary += "\n".join(top_pairs)
    return summary

def main():
    # Paths
    embedded_papers_path = Path(".", "sem_similarity", "embedded_papers.jsonl")
    papers_path = Path(".", "data_scisumm", "cl_scisumm.jsonl")
    
    # Load data
    print("Loading similarity data...")
    embedded_papers = load_similarity_data(embedded_papers_path)
    
    if not embedded_papers:
        print("No similarity data found. Please run semantic_similarity.py first to generate the data.")
        return
    
    print("Loading original papers data...")
    papers_data = utils.load_jsonl(papers_path)
    
    # Extract similarity matrix
    print("Extracting similarity matrix...")
    similarity_matrix, paper_ids = extract_similarity_matrix(embedded_papers)
    
    if similarity_matrix is None:
        print("Could not extract similarity matrix.")
        return
    
    # Create paper labels
    print("Creating paper labels...")
    # Use short labels (just IDs) for large datasets, detailed for small ones
    use_short_labels = len(paper_ids) > 15
    paper_labels = create_paper_labels(paper_ids, papers_data, use_short_labels=use_short_labels)
    
    # Print summary
    print(create_similarity_summary(similarity_matrix, paper_labels))
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Heatmap
    print("Creating similarity heatmap...")
    fig1 = plot_similarity_heatmap(similarity_matrix, paper_labels)
    plt.savefig('similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Distribution plot
    print("Creating similarity distribution plot...")
    fig2 = plot_similarity_distribution(similarity_matrix)
    if fig2:
        plt.savefig('similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\nVisualization complete! Check the generated PNG files.")

if __name__ == "__main__":
    main()