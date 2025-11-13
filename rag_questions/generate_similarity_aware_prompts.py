#!/usr/bin/env python3
"""
Generate prompts for query generation based on selected papers and their neighbors.
Creates a new JSONL file with paper_id and prompt attributes.
"""

import json
from pathlib import Path


def create_prompt(paper_id: str, target_paper: dict, neighbors: list, facet_data: dict) -> str:
    """
    Create a prompt for generating retrieval queries for a target paper.
    
    Args:
        paper_id: The ID of the target paper
        target_paper: Dict with 'title', 'abstract' and 'text' keys for the target paper
        neighbors: List of dicts with 'title', 'abstract', 'text', and 'similarity' keys
        facet_data: Dict with 'shared_facets', 'unique_facets', 'disallowed_terms', and 'facet_weights'
    
    Returns:
        The formatted prompt string
    """
    
    # Format the input JSON
    input_data = {
        "target": target_paper,
        "neighbors": neighbors,
        "shared_facets": facet_data.get('shared_facets', []),
        "unique_facets": facet_data.get('unique_facets', []),
        "disallowed_terms": facet_data.get('disallowed_terms', []),
        "facet_weights": facet_data.get('facet_weights', {})
    }
    
    input_json = json.dumps(input_data, indent=2)
    
    # Create the full prompt
    prompt = f"""You are a research assistant generating search queries for ACL papers.

Goal:
Write EXACTLY THREE natural-language questions that could be used to retrieve this TARGET paper in a semantic-search system.
Use the facet lists to balance recall and distinctiveness relative to its NEIGHBOR papers.

Input JSON:
{input_json}

Write three questions ≤ 20 words each, ending with "?".

### Composition Rules
- **Q1 (high_recall)**  
  - Must include **exactly 1** term from `shared_facets`.  
  - Must include **0** from `disallowed_terms`.  
  - `unique_facets` optional.

- **Q2 (balanced)**  
  - Must include **≥ 1** term from `unique_facets`.  
  - May include **≤ 1** from `shared_facets`.  
  - Must include **0** from `disallowed_terms`.

- **Q3 (high_precision_distinctive)**  
  - Must include **≥ 2** terms from `unique_facets`, at least **1 multi-word** phrase.  
  - Must include **0** from `shared_facets`.  
  - Must include **0** from `disallowed_terms`.

### Style Constraints
- Prefer **capitalized** or **multi-word** items from `unique_facets` (they signal distinctiveness).
- Avoid boilerplate terms: *approach, method, role, stage, paper, propose, present, previous, great, findings*.
- No invented facts; stay consistent with the TARGET paper's abstract.
- Be fluent, question-like, and retrieval-ready.
- Do not start questions with “Does the paper”, “Can the system”, or “What evaluation”.
- Prefer formulations beginning with “How”, “Which”, or “What … in”.

Output JSON:
{{
    "queries": [
        {{"id":"Q1","breadth":"high_recall","text":"..."}},
        {{"id":"Q2","breadth":"balanced","text":"..."}},
        {{"id":"Q3","breadth":"high_precision_distinctive","text":"..."}}
    ],
    "meta": {{
        "used_shared": {{"Q1":[...],"Q2":[...],"Q3":[...]}},
        "used_unique": {{"Q1":[...],"Q2":[...],"Q3":[...]}},
        "violations":[]
    }}
}}
"""
    
    return prompt


def main():
    # Read the selected_papers.jsonl file
    input_file = Path(".", "data_scisumm", "selected_papers.jsonl")
    facet_file = Path(".", "data_scisumm", "facet_analysis_results.jsonl")
    output_file = Path(".", "data_scisumm", "ollama_prompt_similarity_aware.jsonl")
    
    print(f"Reading papers from: {input_file}")
    print(f"Reading facet data from: {facet_file}")
    
    # First, load all papers into memory
    print("Loading all papers into memory...")
    papers_by_id = {}
    with open(input_file, 'r', encoding='utf-8') as in_f:
        for line_num, line in enumerate(in_f, 1):
            try:
                paper_data = json.loads(line)
                paper_id = paper_data.get('paper_id')
                if paper_id:
                    papers_by_id[paper_id] = paper_data
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(papers_by_id)} papers")
    
    # Load facet data
    print("Loading facet data...")
    facets_by_id = {}
    with open(facet_file, 'r', encoding='utf-8') as facet_f:
        for line_num, line in enumerate(facet_f, 1):
            try:
                facet_data = json.loads(line)
                paper_id = facet_data.get('paper_id')
                if paper_id:
                    facets_by_id[paper_id] = facet_data
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing facet JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(facets_by_id)} facet entries")
    
    # Process each paper
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for paper_num, (paper_id, paper_data) in enumerate(papers_by_id.items(), 1):
            print(f"Processing paper {paper_num}/{len(papers_by_id)}: {paper_id}...", end='\r')
            
            try:
                # Check if we have facet data for this paper
                if paper_id not in facets_by_id:
                    print(f"\nWarning: No facet data found for {paper_id}, skipping...")
                    continue
                
                facet_data = facets_by_id[paper_id]
                
                # Extract target paper info
                target = {
                    "title": paper_data.get('title', ''),
                    "abstract": paper_data.get('abstract', ''),
                    "text": paper_data.get('text', '')
                }
                
                # Extract neighbor information
                neighbors = []
                relevance = paper_data.get('relevance', {})
                
                # Get all neighbors except the target paper itself
                for neighbor_id, similarity in relevance.items():
                    if neighbor_id == paper_id:
                        continue
                    
                    # Look up neighbor in our loaded papers
                    neighbor_data = papers_by_id.get(neighbor_id, {})
                    
                    neighbors.append({
                        "paper_id": neighbor_id,
                        "title": neighbor_data.get('title', ''),
                        "abstract": neighbor_data.get('abstract', ''),
                        "text": neighbor_data.get('text', ''),
                        "similarity": similarity
                    })
                
                # Sort neighbors by similarity (descending)
                neighbors.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Prepare facet data for prompt
                facet_info = {
                    'shared_facets': facet_data.get('shared_facets', []),
                    'unique_facets': facet_data.get('unique_facets', []),
                    'disallowed_terms': facet_data.get('disallowed_terms', []),
                    'facet_weights': facet_data.get('facet_weights', {})
                }
                
                prompt = create_prompt(paper_id, target, neighbors, facet_info)
                
                # Write to output
                output_entry = {
                    "paper_id": paper_id,
                    "prompt": prompt
                }
                
                out_f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error processing paper {paper_id}: {e}")
                continue
    
    print(f"\nDone! Output written to: {output_file}")


if __name__ == "__main__":
    main()
