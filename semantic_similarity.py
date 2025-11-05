from sentence_transformers import SentenceTransformer
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
    # Load embedded papers
    embedded_papers_path = Path(".", "sem_similarity", "embedded_papers.jsonl")
    # model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    
    # Compute Embeddings
    embeddings = []
    for i, paper in enumerate(papers):
        print(f"Embedding paper {paper['paper_id']}...")
        prompt = paper["text"]

        embedded_input = model.encode([prompt])
        paper["embedding"] = embedded_input[0].tolist()
        embeddings.append(paper)

    # Compute pair-wise similarities
    for i, paper_a in enumerate(embeddings):
        
        to_save = paper_a.copy()
        to_save["similarity_scores"] = {}
        
        for j, paper_b in enumerate(embeddings):
            if i > j: continue # Avoid duplicates, but keep self-similarity
            
            similarity = model.similarity(paper_a["embedding"], paper_b["embedding"])
            similarity_float = float(similarity.item())  # Convert tensor to float
            print(f"Similarity between {paper_a['paper_id']} and {paper_b['paper_id']}: {similarity_float}")
            to_save["similarity_scores"][paper_b["paper_id"]] = similarity_float
            
        del to_save["embedding"]  # Remove embedding before saving as tensors are not JSON serializable

        utils.append_to_jsonl(to_save, embedded_papers_path)    