# RAG Questions Pipeline

This directory contains scripts for generating and evaluating questions for Retrieval-Augmented Generation (RAG) on scientific papers from the ACL anthology. The workflow supports two question generation approaches: **paper-only** (using only target paper information) and **similarity-aware** (using semantic facets derived from similar papers).


## Directory Structure

### Subdirectories

- **`data/`** - Contains all input/output data files:
  - Input: `selected_papers.jsonl`, `facet_analysis_results.jsonl`
  - Generated questions: `gpt_questions_paper_only.jsonl`, `gpt_questions_similarity_aware.jsonl`
  - Prompts: `ollama_prompt_paper_only.jsonl`, `ollama_prompt_similarity_aware.jsonl`
  - Evaluations: `question_answerability_evaluations_*.jsonl`
  - Other: `gpt_simplified.jsonl`, `gpt_summaries.jsonl`, `relevance_indices.py`

- **`output/`** - Contains analysis results and visualizations:
  - Text reports and statistics
  - PNG visualizations (heatmaps, distributions, etc.)
  - Subdirectory `similarity_aware/` for similarity-aware specific outputs

- **`parquet/`** - Contains Parquet datasets for model training/evaluation:
  - `rag_queries/` - Queries in Parquet format
  - `rag_qrels/` - Query relevance judgments (qrels)


## Scripts Overview

Scripts are organized by workflow phase. Each script includes detailed parameter documentation below.


## Phase 1: Data Preparation

### 1.1 `extract_titles_and_abstracts.py`

**Purpose**: Extract title and abstract fields from the `text` field in corpus files.

**Functionality**:

- Parses structured text to extract titles (between `Title:` and `ABSTRACT`)
- Extracts abstracts (between `ABSTRACT` and `SECTION 1:`)
- Creates separate `title` and `abstract` fields while preserving original `text`

**Input**:

- `data_scisumm/corpus.jsonl` (or specified input file)

**Output**:

- `data_scisumm/corpus_temp.jsonl` (processed file with extracted fields)

**Usage**:

```bash
python extract_titles_and_abstracts.py
```

**Parameters**: None (hardcoded paths in script)

**Warnings**:

- Input file must have structured format with `Title:`, `ABSTRACT`, and `SECTION 1:` markers
- Prints warnings for papers where extraction fails


### 1.2 `facets.py`

**Purpose**: Analyze semantic facets for each paper relative to its neighbors using NLP and TF-IDF.

**Functionality**:

- Extracts terms from paper titles/abstracts using SpaCy NLP
- Identifies shared facets (common with neighbors), unique facets (distinctive to target), and disallowed terms (too generic)
- Uses TF-IDF scoring to rank terms by distinctiveness
- Generates facet buckets with configurable thresholds

**Input**:

- `data_scisumm/selected_papers.jsonl` (papers with `relevance` scores for neighbors)

**Output**:

- `data_scisumm/facet_analysis_results.jsonl` (facets for each paper)
- `data_scisumm/facet_analysis_summary.json` (statistics)

**Usage**:

```bash
python facets.py
```

**Parameters**:

- `shared_thresh` (default 0.33): Minimum neighbor frequency for shared facets
- `unique_thresh` (default 0.11): Maximum neighbor frequency for unique facets  
- `disallowed_thresh` (default 0.33): Minimum neighbor frequency for disallowed terms
- `top_unique`, `top_shared`, `top_disallowed`: Max terms per category

**Warnings**:

- Requires SpaCy model: `python -m spacy download en_core_web_sm`
- Papers without `relevance` scores will be skipped
- Optimized for ~10 paper corpus (adjust thresholds for larger datasets)


## Phase 2: Question Generation

### 2.1 `generate_paper_only_prompts.py`

**Purpose**: Generate prompts for query generation using only target paper information.

**Functionality**:

- Creates LLM prompts with paper metadata (title, abstract, authors, venue, etc.)
- Instructs LLM to generate 3 questions of varying breadth (high recall, balanced, high precision)
- Does not use neighbor/similarity information

**Input**:

- `data_scisumm/selected_papers.jsonl`

**Output**:

- `data_scisumm/ollama_prompt_paper_only.jsonl`

**Usage**:

```bash
python generate_paper_only_prompts.py
```

**Parameters**: None (hardcoded paths)

**Example Output**:

```json
{
  "paper_id": "C90-2039",
  "prompt": "You are a research assistant generating search queries... [full prompt text]"
}
```


### 2.2 `generate_similarity_aware_prompts.py`

**Purpose**: Generate prompts that incorporate semantic facets from similar papers.

**Functionality**:

- Loads target papers and their neighbor information
- Incorporates facet analysis (shared/unique/disallowed terms) into prompts
- Instructs LLM to use facets to balance recall and distinctiveness
- Generates prompts with strict composition rules for each question type

**Input**:

- `data_scisumm/selected_papers.jsonl` (with `relevance` field)
- `data_scisumm/facet_analysis_results.jsonl`

**Output**:

- `data_scisumm/ollama_prompt_similarity_aware.jsonl`

**Usage**:

```bash
python generate_similarity_aware_prompts.py
```

**Parameters**: None (hardcoded paths)

**Warnings**:

- Requires facets to be generated first (run `facets.py`)
- Skips papers without facet data


### 2.3 `generate_questions_gpt.py`

**Purpose**: Execute LLM query generation using generated prompts.

**Functionality**:

- Loads prompts from JSONL files
- Sends prompts to Ollama API (gpt-oss:120b-cloud model)
- Handles rate limiting with automatic retry/sleep
- Saves generated questions incrementally

**Input**:

- `data_scisumm/ollama_prompt_paper_only.jsonl` (or similarity_aware variant)

**Output**:

- `data_scisumm/gpt_questions_paper_only.jsonl` (or similarity_aware variant)

**Usage**:

```bash
python generate_questions_gpt.py
```

**Parameters**:

- `num_ctx`: 130000 (context window size for LLM)

**Warnings**:

- Requires Ollama API access
- Automatically sleeps 30 minutes if hitting hourly rate limits
- Resumes from where it left off (checks existing output file)


## Phase 3: Evaluation & Analysis

### 3.1 `evaluate_question_answerability.py`

**Purpose**: Evaluate whether each paper can answer each question using LLM judgments.

**Functionality**:

- Creates full matrix of paper-question pairs
- Uses LLM to judge if a paper can answer a question
- Provides structured analysis with section references
- Supports distributed processing across machines

**Input**:

- `data_scisumm/cl_scisumm.jsonl` (all papers)
- `data/gpt_questions_similarity_aware.jsonl`

**Output**:

- `data/question_answerability_evaluations_similarity_aware_full.jsonl`

**Usage**:

```bash
# Standard usage
python evaluate_question_answerability.py

# Advanced: Process in reverse order
python evaluate_question_answerability.py --reverse

# Advanced: Process specific range
python evaluate_question_answerability.py --start-idx 0 --end-idx 50

# Advanced: Limit to first N papers
python evaluate_question_answerability.py --limit 100

# Advanced: Add suffix to output filename
python evaluate_question_answerability.py --output-suffix "_machine2"
```

**Parameters**:

- `--reverse`: Process papers in reverse order
- `--start-idx N`: Start from paper index N (0-based)
- `--end-idx N`: Stop at paper index N (exclusive)
- `--limit N`: Hard stop after N papers
- `--output-suffix STR`: Add suffix to output filename

**Warnings**:

- Requires `OLLAMA_API_KEY` environment variable
- Very slow (full matrix evaluation)
- Automatically handles rate limits (30 min sleep)
- Papers >32K chars are truncated


### 3.2 `analyze_question_paper_similarity.py`

**Purpose**: Calculate semantic similarity between papers and generated questions using embeddings.

**Functionality**:

- Loads papers and questions
- Encodes all text using sentence transformer model (gte-Qwen2-1.5B-instruct)
- Computes cosine similarity matrix
- Generates heatmap visualization

**Input**:

- `data/selected_papers.jsonl`
- `data/gpt_questions_similarity_aware.jsonl`

**Output**:

- `output/paper_question_similarity_heatmap_similarity_aware.png`
- `output/paper_question_similarity_values_similarity_aware.txt`

**Usage**:

```bash
python analyze_question_paper_similarity.py
```

**Parameters**:

- `model_name`: `'Alibaba-NLP/gte-Qwen2-1.5B-instruct'` (embedding model)

**Warnings**:

- Requires sentence-transformers library
- Memory intensive for large corpora (truncates papers >32K chars)
- Downloads model on first run (~3GB)


### 3.3 `analyze_similarity_answerability_overlap.py`

**Purpose**: Analyze correlation between semantic similarity and LLM answerability judgments.

**Functionality**:

- Compares embedding-based similarity scores with LLM answerability decisions
- Computes statistics (mean similarity for answerable vs. non-answerable pairs)
- Checks if each question has at least one answerable paper
- Generates visualizations (histograms, boxplots, per-question analysis)

**Input**:

- `rag_questions/output/paper_question_similarity_values_similarity_aware.txt`
- `rag_questions/data/question_answerability_evaluations_similarity_aware.jsonl`

**Output**:

- `output/similarity_aware/similarity_answerability_overlap_report.txt`
- `output/similarity_aware/similarity_distribution_by_answerability_paper_only.png`
- `output/similarity_aware/similarity_boxplot_by_answerability_paper_only.png`
- `output/similarity_aware/highest_similarity_answerability_paper_only.png`

**Usage**:

```bash
python analyze_similarity_answerability_overlap.py
```

**Parameters**: None

**Warnings**:

- Requires matplotlib, numpy for visualizations
- Depends on outputs from previous analysis scripts


### 3.4 `analyze_query_differences.py`

**Purpose**: Compare questions generated by paper-only vs. similarity-aware approaches.

**Functionality**:

- Loads both question sets
- Identifies papers with identical vs. different questions
- Shows detailed side-by-side comparison of differing questions

**Input**:

- `data_scisumm/gpt_questions_paper_only.jsonl`
- `data_scisumm/gpt_questions_similarity_aware.jsonl`

**Output**:

- Console output (detailed comparison report)

**Usage**:

```bash
python analyze_query_differences.py
```

**Parameters**: None


### 3.5 `analyze_similarity_scores.py`

**Purpose**: Analyze distribution of paper-to-paper similarity scores.

**Functionality**:

- Loads similarity data from JSONL file
- Counts papers in 0.01 bins from threshold (default 0.8) to 1.0
- Excludes self-similarity (score = 1.0)
- Prints formatted distribution table

**Input**:

- `sem_similarity/Qwen3-Embedding-0.6B/all_new/similarity_papers.jsonl`

**Output**:

- Console output (distribution table and statistics)

**Usage**:

```bash
python analyze_similarity_scores.py
```

**Parameters**:

- `min_threshold`: 0.8 (minimum similarity to include)

**Warnings**:

- Hardcoded path to similarity file (modify in `main()` if needed)


## Phase 4: Dataset Conversion & Export

### 4.1 `generate_relevance_indices.py`

**Purpose**: Generate relevance indices for document retrieval evaluation based on LLM judgments.

**Functionality**:

- Processes answerability evaluations to create ground truth relevance
- Maps each question to paper indices that can answer it
- Outputs Python code for direct import

**Input**:

- `rag_questions/data/selected_papers.jsonl`
- `rag_questions/data/question_answerability_evaluations_similarity_aware.jsonl`

**Output**:

- `rag_questions/data/relevance_indices.py` (Python list variable)
- `rag_questions/output/similarity_aware/relevance_indices_statistics.txt`

**Usage**:

```bash
python generate_relevance_indices.py
```

**Parameters**: None

**Warnings**:

- Reports questions without any relevant papers (should be 0)


### 4.2 `create_qrels.py`

**Purpose**: Create qrels (query relevance) Parquet datasets for train/test splits.

**Functionality**:

- Converts relevance indices to standard qrels format
- Generates train and test qrels with columns: query-id, corpus-id, score
- Binary relevance (score = 1 for all relevant pairs)

**Input**:

- Hardcoded relevance indices in script (train and test splits)

**Output**:

- `parquet/train_qrels.parquet`
- `parquet/test_qrels.parquet`

**Usage**:

```bash
python create_qrels.py
```

**Parameters**: None (relevance indices hardcoded in script)

**Example Output Format**:

```
query-id | corpus-id | score------|-----------|------
1        | 0         | 1
1        | 1         | 1
```


### 4.3 `convert_papers_to_parquet.py`

**Purpose**: Convert selected papers from JSONL to Parquet format.

**Functionality**:

- Extracts paper metadata and text
- Creates Parquet dataset with columns: doc_idx, paper_id, doc_title, doc_abs, doc_text
- Sequential 0-based indexing

**Input**:

- `data/selected_papers.jsonl`

**Output**:

- Parquet file (default: `selected_papers.parquet`)

**Usage**:

```bash
# Basic usage
python convert_papers_to_parquet.py

# Specify custom paths
python convert_papers_to_parquet.py --input data/custom.jsonl --output output.parquet
```

**Parameters**:

- `--input FILE`: Input JSONL file path
- `--output FILE`: Output Parquet file path


### 4.4 `convert_gpt_queries_to_parquet.py`

**Purpose**: Convert GPT-generated queries from JSONL to Parquet format.

**Functionality**:

- Parses nested JSON structure (outer JSONL + inner gpt_queries JSON)
- Extracts query text and assigns sequential query_id (1-based)
- Creates columns: query_id, query

**Input**:

- GPT questions JSONL file (e.g., `data/gpt_questions_similarity_aware.jsonl`)

**Output**:

- Parquet file (default: same name with `.parquet` extension)

**Usage**:

```bash
# Basic usage
python convert_gpt_queries_to_parquet.py input.jsonl

# Specify output path
python convert_gpt_queries_to_parquet.py input.jsonl output.parquet
```

**Parameters**:

- `input_file`: Path to input JSONL file (required)
- `output_file`: Path to output Parquet file (optional)

**Warnings**:

- Prints warnings for lines with parsing errors
- Shows count of unique queries (detects duplicates)


### 4.5 `convert_to_parquet.py`

**Purpose**: Convert scientific paper summaries from JSONL to Parquet for Hugging Face datasets.

**Functionality**:

- Supports two formats: `gpt_simplified.jsonl` and `gpt_summaries.jsonl`
- Extracts title from text field
- Creates columns: paper_id, title, abstract, full_text, human_summary, gpt_simplified_summary/gpt_summary
- Optional random sampling

**Input**:

- `data_scisumm/gpt_simplified.jsonl` or `data_scisumm/gpt_summaries.jsonl`

**Output**:

- Parquet file (default: `parquet/scientific_papers_dataset.parquet`)

**Usage**:

```bash
# Convert simplified format
python convert_to_parquet.py -i data_scisumm/gpt_simplified.jsonl -o output.parquet -f simplified

# Convert summaries format
python convert_to_parquet.py -i data_scisumm/gpt_summaries.jsonl -o output.parquet -f summaries

# Random sample of 100 papers
python convert_to_parquet.py -i input.jsonl -o output.parquet -n 100
```

**Parameters**:

- `--input, -i FILE`: Input JSONL file path (default: `data_scisumm/gpt_simplified.jsonl`)
- `--output, -o FILE`: Output Parquet file path (default: `parquet/scientific_papers_dataset.parquet`)
- `--format, -f {simplified,summaries}`: Input format type (default: `simplified`)
- `--num-samples, -n N`: Random sample size (default: all papers)


## Execution Order

### Standard Workflow (Similarity-Aware Questions)

The following order ensures all dependencies are satisfied:

#### **Step 1: Data Preparation**

```bash
# 1. Extract titles and abstracts (if not already present)
python extract_titles_and_abstracts.py

# 2. Generate semantic facets for each paper
python facets.py
```

#### **Step 2: Generate Questions**

```bash
# 3. Create prompts with facet information
python generate_similarity_aware_prompts.py

# 4. Generate questions using LLM
python generate_questions_gpt.py
```

#### **Step 3: Evaluate & Analyze**

```bash
# 5. Calculate semantic similarity between papers and questions
python analyze_question_paper_similarity.py

# 6. Evaluate answerability (slow - full matrix)
python evaluate_question_answerability.py

# 7. Analyze overlap between similarity and answerability
python analyze_similarity_answerability_overlap.py

# 8. Generate relevance indices from answerability judgments
python generate_relevance_indices.py
```

#### **Step 4: Export Datasets**

```bash
# 9. Create qrels for training/evaluation
python create_qrels.py

# 10. Convert to Parquet format for model training
python convert_papers_to_parquet.py
python convert_gpt_queries_to_parquet.py
```


### Alternative Workflow (Paper-Only Questions)

For baseline comparison without similarity information:

```bash
# 1. Extract titles/abstracts (same as above)
python extract_titles_and_abstracts.py

# 2. Generate paper-only prompts
python generate_paper_only_prompts.py

# 3. Generate questions (modify script to use paper-only prompts)
python generate_questions_gpt.py

# 4-10. Follow same evaluation/analysis/export steps
```


### Comparison Workflow

To compare both approaches:

```bash
# After running both workflows above:
python analyze_query_differences.py
```


## Dependency Graph

```
extract_titles_and_abstracts.py
    ↓
facets.py
    ↓
generate_similarity_aware_prompts.py  (OR: generate_paper_only_prompts.py)
    ↓
generate_questions_gpt.py
    ↓
    ├─→ analyze_question_paper_similarity.py
    │       ↓
    └─→ evaluate_question_answerability.py
            ↓
            ├─→ analyze_similarity_answerability_overlap.py
            └─→ generate_relevance_indices.py
                    ↓
                    └─→ create_qrels.py
                            ↓
                            ├─→ convert_papers_to_parquet.py
                            └─→ convert_gpt_queries_to_parquet.py
```

**Independent Scripts** (can run anytime after questions exist):

- `analyze_query_differences.py` (compares two question sets)
- `analyze_similarity_scores.py` (analyzes paper-paper similarity)
- `convert_to_parquet.py` (converts summary files)


## Requirements

### Python Packages

```bash
pip install ollama pandas numpy matplotlib seaborn sentence-transformers spacy scikit-learn
python -m spacy download en_core_web_sm
```

### Environment Variables

- `OLLAMA_API_KEY`: Required for `evaluate_question_answerability.py` and `generate_questions_gpt.py`


## Common Use Cases

### 1. Generate Questions for New Papers

```bash
# Add papers to selected_papers.jsonl with relevance scores
python facets.py
python generate_similarity_aware_prompts.py
python generate_questions_gpt.py
```

### 2. Evaluate Question Quality

```bash
python analyze_question_paper_similarity.py
python evaluate_question_answerability.py  # Warning: very slow!
python analyze_similarity_answerability_overlap.py
```

### 3. Export for Model Training

```bash
python generate_relevance_indices.py
python create_qrels.py
python convert_papers_to_parquet.py
python convert_gpt_queries_to_parquet.py
```


## Troubleshooting

**Issue**: `OLLAMA_API_KEY not set`

- **Solution**: Set environment variable before running LLM scripts

  ```bash
  export OLLAMA_API_KEY="your_key_here"
  ```

**Issue**: SpaCy model not found

- **Solution**: Download the model

  ```bash
  python -m spacy download en_core_web_sm
  ```

**Issue**: Out of memory during similarity calculation

- **Solution**: Script automatically truncates papers >32K characters, but reduce batch size if needed

**Issue**: Rate limit errors with Ollama

- **Solution**: Scripts automatically sleep for 30 minutes and retry


## Notes

- All JSONL files use UTF-8 encoding
- Paper IDs must be unique across all datasets
- Question IDs follow format: `{paper_id}_Q{1|2|3}`
- Similarity scores are cosine similarity (range: 0-1)
- Answerability judgments are binary: "yes" or "no"
- Most scripts resume from where they left off (check for existing output files)


## Author & Citation

This pipeline was developed for RAG research on scientific papers from the ACL anthology. If you use this code, please cite the relevant papers and datasets (CL-SciSumm, ACL-OCL).
