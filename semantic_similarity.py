from sentence_transformers import SentenceTransformer
import utils
import json
import gc

try:
    import torch
except Exception:
    torch = None

from pathlib import Path

if __name__ == "__main__":
    # Load papers
    papers_path = Path(".", "data_scisumm", "cl_scisumm_questions.jsonl")
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
    try:
        embedded_papers = utils.load_jsonl(embedded_papers_path)
        processed_ids = [s["paper_id"] for s in embedded_papers]
    except Exception as _:
        print(f"File {str(embedded_papers_path)} not found. Starting from scratch.")
        processed_ids = []
    
    # model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    
    # Compute Embeddings
    embeddings = []
    for paper in papers:
        if paper["paper_id"] in processed_ids: continue

        print(f"Embedding paper {paper['paper_id']}...")

        embedded_input = model.encode([paper["text"]])
        paper["embedding"] = json.dumps(embedded_input[0].tolist())
        utils.append_to_jsonl(paper, embedded_papers_path)
        
        try:
            del embedded_input
        except Exception:
            pass
        
        if "embedding" in paper:
            try:
                del paper["embedding"]
            except Exception:
                pass
        
        gc.collect()
        if torch is not None:
            try:
                # release any CUDA memory held by PyTorch
                torch.cuda.empty_cache()
            except Exception:
                pass

    embedded_papers = utils.load_jsonl(embedded_papers_path)
    for p in embedded_papers:
        emb = p.get("embedding")
        if isinstance(emb, str):
            try:
                p["embedding"] = json.loads(emb)
            except Exception:
                pass
    similarity_papers_path = Path(".", "sem_similarity", "similarity_papers.jsonl")
    
    # Compute pair-wise similarities
    for i, paper_a in enumerate(embedded_papers):

        to_save = paper_a.copy()
        to_save["similarity_scores"] = {}

        for j, paper_b in enumerate(embedded_papers):
            if i > j: continue # Avoid duplicates, but keep self-similarity
            
            similarity = model.similarity(paper_a["embedding"], paper_b["embedding"])
            similarity_float = float(similarity.item())  # Convert tensor to float
            print(f"Similarity between {paper_a['paper_id']} and {paper_b['paper_id']}: {similarity_float}")
            to_save["similarity_scores"][paper_b["paper_id"]] = similarity_float
            
        del to_save["embedding"]  # Remove embedding before saving as tensors are not JSON serializable

        utils.append_to_jsonl(to_save, similarity_papers_path)    