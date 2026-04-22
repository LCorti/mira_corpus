import json
import math
import re
import spacy

from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

# ----------------------------
# 1) Setup
# ----------------------------
# Configuration for 10-paper corpus (adjust thresholds for larger corpora):
# - shared_thresh = 0.33 → DF ≥ 3 neighbors (yields ~4-6 shared terms)
# - unique_thresh = 0.11 → DF ≤ 1 neighbor (yields ~8-10 unique terms)
# - disallowed_thresh = 0.33 → DF ≥ 3 neighbors (yields ~2-4 disallowed terms)
#
# Global enrichment lists compensate for small corpus size by ensuring
# minimum vocabulary coverage for similarity-aware prompts.

nlp = spacy.load("en_core_web_sm", disable=["ner"])  # keep tagger/parser
STOP = set(spacy.lang.en.stop_words.STOP_WORDS)

ACRONYM_RE = re.compile(r"\b([A-Z]{2,})\b")  # LSA, WSJ, BNC, PP, etc.


def normalize_token(tok):
    # Keep acronyms and ProperNouns as-is; otherwise lowercase lemma
    if tok.is_stop or tok.is_punct or tok.is_space or tok.like_num:
        return None
    text = tok.text
    if ACRONYM_RE.fullmatch(text):
        return text  # keep uppercase
    if tok.pos_ in {"PROPN"} and text[0].isupper():
        return text  # keep case (WordNet)
    # general case
    lemma = tok.lemma_.lower()
    if len(lemma) < 3:
        return None
    return lemma


def noun_phrases(doc):
    for nc in doc.noun_chunks:
        # filter trivial chunks
        toks = [t for t in nc if not (t.is_stop or t.is_punct or t.is_space)]
        if 1 <= len(toks) <= 5:
            yield " ".join([t.text for t in toks])


def frequent_ngrams(doc, n=2, max_n=4):
    toks = [t for t in doc if normalize_token(t)]
    words = [t.text for t in toks]
    out = Counter()
    for m in range(n, max_n + 1):
        for i in range(len(words) - m + 1):
            span = words[i : i + m]
            # allow internal stopwords sparsely
            if sum(w.lower() in STOP for w in span) > m // 2:
                continue
            out[" ".join(span)] += 1
    return [p for p, c in out.items() if c >= 2]


def canonicalize_phrase(s: str):
    s = s.replace("-", "-").replace("–", "-").replace("_", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"-", " ", s)
    # keep case if acronym or ProperNoun; else lowercase
    if ACRONYM_RE.fullmatch(s) or any(w[0].isupper() for w in s.split()):
        return s
    return s.lower()


def extract_terms(text: str) -> Tuple[Counter, set]:
    """Return (term_counts, term_presence_set) with canonical forms."""
    doc = nlp(text)

    # unigram terms
    uni = [normalize_token(t) for t in doc]
    uni = [u for u in uni if u]

    # multiword candidates
    phrases = set(noun_phrases(doc)) | set(frequent_ngrams(doc, 2, 4))

    # fold acronyms (ABC -> expanded) if pattern X (ABC) appears
    acronym_map = {}
    for m in re.finditer(
        r"([A-Za-z][A-Za-z \-]{2,})\s*\(\s*([A-Z]{2,})\s*\)", text
    ):  # e.g., Latent Semantic Analysis (LSA)
        expanded = canonicalize_phrase(m.group(1))
        acronym = m.group(2)
        acronym_map[acronym] = expanded

    # canonicalize
    cand = [canonicalize_phrase(w) for w in uni] + [
        canonicalize_phrase(p) for p in phrases
    ]

    # apply acronym expansion mapping
    expanded = []
    for c in cand:
        if c in acronym_map:
            expanded.append(acronym_map[c])
        else:
            expanded.append(c)
    counts = Counter(expanded)
    presence = {t for t in counts if t and not t.isspace()}
    return counts, presence


# ----------------------------
# 2) Facet buckets
# ----------------------------

# Global shared terms that are universally safe for small corpora
GLOBAL_SHARED = [
    "word",
    "corpus",
    "language",
    "distributional",
    "automatic",
    "unsupervised",
    "semantic",
    "similarity",
]

# Global disallowed terms (frequent, low-information boilerplate)
GLOBAL_DISALLOWED = ["semantic", "similarity", "present", "approach"]

# Noise/boilerplate filter
NOISE = {
    "base",
    "present",
    "approach",
    "method",
    "model",
    "paper",
    "work",
    "result",
    "experiment",
    "system",
    "technique",
    "task",
    "study",
    "data",
    "language",
}


def build_facet_buckets(
    target_text: str,
    neighbor_texts: List[str],
    shared_thresh=0.33,
    unique_thresh=0.11,
    disallowed_thresh=0.33,
    top_unique=10,
    top_shared=10,
    top_disallowed=12,
    verbose=False,
):
    # extract presence sets
    t_counts, t_presence = extract_terms(target_text)
    n_presence_list = []
    for txt in neighbor_texts:
        _, prs = extract_terms(txt)
        n_presence_list.append(prs)

    k = max(1, len(n_presence_list))
    # neighbor DF
    df_nbr = Counter()
    for prs in n_presence_list:
        for t in prs:
            df_nbr[t] += 1

    # local TF-IDF to rank unique terms
    corpus = [target_text] + neighbor_texts
    vec = TfidfVectorizer(
        lowercase=False,  # we already normalized
        token_pattern=r"(?u)\b[\w\-]+\b",
        ngram_range=(1, 3),
        min_df=1,
        max_df=1.0,
    )
    X = vec.fit_transform(corpus)
    vocab = {v: k for k, v in vec.vocabulary_.items()}
    target_row = X[0]
    tfidf_scores = {}
    for idx in target_row.nonzero()[1]:
        tfidf_scores[vocab[idx]] = target_row[0, idx]

    # map tfidf tokens back to our canonical presence (approximate)
    def score_term(t):
        # sum scores of exact token and its whitespace-joined form
        s = tfidf_scores.get(t, 0.0) + tfidf_scores.get(t.replace(" ", ""), 0.0)
        return float(s)

    # Buckets
    shared_min = math.ceil(shared_thresh * k)
    unique_max = math.floor(unique_thresh * k)
    disallowed_min = math.ceil(disallowed_thresh * k)

    if verbose:
        print(f"\n[DEBUG] Thresholds with k={k} neighbors:")
        print(
            f"  - Shared: term must appear in >= {shared_min} neighbors (thresh={shared_thresh:.2f})"
        )
        print(
            f"  - Unique: term must appear in <= {unique_max} neighbors (thresh={unique_thresh:.2f})"
        )
        print(
            f"  - Disallowed: term must appear in >= {disallowed_min} neighbors AND not in target (thresh={disallowed_thresh:.2f})"
        )
        print(f"  - Target has {len(t_presence)} unique terms")

    SHARED = [t for t in t_presence if df_nbr[t] >= shared_min]
    UNIQUE = [t for t in t_presence if df_nbr[t] <= unique_max]
    DISALLOWED = [
        t for t in df_nbr if t not in t_presence and df_nbr[t] >= disallowed_min
    ]

    if verbose:
        print(f"  - Shared candidates before ranking: {len(SHARED)}")
        print(f"  - Unique candidates before ranking: {len(UNIQUE)}")
        print(f"  - Disallowed candidates before ranking: {len(DISALLOWED)}")

    # rank
    SHARED = sorted(SHARED, key=lambda t: (df_nbr[t], score_term(t)), reverse=True)[
        :top_shared
    ]
    UNIQUE = sorted(UNIQUE, key=lambda t: (score_term(t), -df_nbr[t]), reverse=True)[
        :top_unique
    ]
    DISALLOWED = sorted(DISALLOWED, key=lambda t: df_nbr[t], reverse=True)[
        :top_disallowed
    ]

    # Apply noise filter to all lists
    SHARED = [t for t in SHARED if t.lower() not in NOISE]
    UNIQUE = [t for t in UNIQUE if t.lower() not in NOISE]
    DISALLOWED = [t for t in DISALLOWED if t.lower() not in NOISE]

    # Enrich shared facets with global shared terms
    SHARED = list(set(SHARED) | set(GLOBAL_SHARED))
    SHARED = [t for t in SHARED if t.lower() not in NOISE]

    # Enrich disallowed with global clichés
    DISALLOWED = list(set(DISALLOWED) | set(GLOBAL_DISALLOWED))
    DISALLOWED = [t for t in DISALLOWED if t.lower() not in NOISE]

    if verbose:
        print("  - After enrichment & filtering:")
        print(
            f"    Shared: {len(SHARED)}, Unique: {len(UNIQUE)}, Disallowed: {len(DISALLOWED)}"
        )

    # Deduplicate across buckets (priority: shared > unique > disallowed)
    shared_lower = {t.lower() for t in SHARED}
    unique_lower = {t.lower() for t in UNIQUE}

    UNIQUE = [t for t in UNIQUE if t.lower() not in shared_lower]
    DISALLOWED = [
        t
        for t in DISALLOWED
        if t.lower() not in shared_lower and t.lower() not in unique_lower
    ]

    # Remove semantic/similarity from disallowed if they're in shared
    DISALLOW_DUPLICATES = {"semantic", "similarity"}
    DISALLOWED = [t for t in DISALLOWED if t.lower() not in DISALLOW_DUPLICATES]

    # Normalize casing (lowercase unless acronym or proper noun)
    def normalize_case(term):
        # Keep if all caps (acronym) or starts with capital (proper noun like WordNet)
        if ACRONYM_RE.fullmatch(term) or (
            term and term[0].isupper() and any(c.isupper() for c in term[1:])
        ):
            return term
        return term.lower()

    SHARED = [normalize_case(t) for t in SHARED]
    UNIQUE = [normalize_case(t) for t in UNIQUE]
    DISALLOWED = [normalize_case(t) for t in DISALLOWED]

    # Cap list lengths for clarity
    SHARED = SHARED[:8]
    UNIQUE = UNIQUE[:10]
    DISALLOWED = DISALLOWED[:10]

    if verbose:
        print("  - After deduplication & capping:")
        print(
            f"    Shared: {len(SHARED)}, Unique: {len(UNIQUE)}, Disallowed: {len(DISALLOWED)}"
        )

    return {
        "shared_facets": SHARED,
        "unique_facets": UNIQUE,
        "disallowed_terms": DISALLOWED,
        "facet_weights": {
            "shared_facets": "use sparingly (1-2 per question)",
            "unique_facets": "highlight in Q2, emphasize in Q3",
            "disallowed_terms": "avoid entirely",
        },
    }


def get_text(p):  # p is your record
    parts = [p.get("title", ""), p.get("abstract", "")]
    return " ".join(parts)


if __name__ == "__main__":
    # Load all papers and create a cache indexed by paper_id
    papers_cache = {}
    with open(Path("data", "selected_papers.jsonl"), "r") as f:
        for line in f:
            paper = json.loads(line)
            paper_id = paper.get("paper_id")
            if paper_id:
                papers_cache[paper_id] = paper

    if not papers_cache:
        print("No papers found in selected_papers.jsonl")
        exit(1)

    print(f"Processing {len(papers_cache)} papers...")

    # Store all results
    all_results = []
    skipped_papers = []

    # Process each paper as target
    for idx, (target_id, target_paper) in enumerate(papers_cache.items(), 1):
        print(f"\n[{idx}/{len(papers_cache)}] Processing target paper: {target_id}")

        # Get the relevance scores for the target paper
        relevance_scores = target_paper.get("relevance", {})

        if not relevance_scores:
            print("  !!! No relevance scores found, skipping")
            skipped_papers.append(
                {"paper_id": target_id, "reason": "no_relevance_scores"}
            )
            continue

        # Find neighbor papers based on relevance scores
        # Sort by similarity score (descending) and look them up in the cache
        neighbor_papers = []
        sorted_neighbors = sorted(
            relevance_scores.items(), key=lambda x: x[1], reverse=True
        )

        for neighbor_id, sim_score in sorted_neighbors:
            if neighbor_id != target_id and neighbor_id in papers_cache:
                neighbor_papers.append(papers_cache[neighbor_id])

        if not neighbor_papers:
            print("  !!! No neighbor papers found in cache, skipping")
            skipped_papers.append(
                {
                    "paper_id": target_id,
                    "reason": "no_neighbors_in_cache",
                    "relevance_count": len(relevance_scores),
                }
            )
            continue

        print(f"  Found {len(neighbor_papers)}/{len(relevance_scores) - 1} neighbors")

        # Extract text from target and neighbors
        target = get_text(target_paper)
        neighbors = [get_text(n) for n in neighbor_papers]

        # Build facet buckets (verbose only for first paper)
        buckets = build_facet_buckets(target, neighbors, verbose=(idx == 1))

        # Create result payload
        result = {
            "paper_id": target_id,
            "title": target_paper.get("title", ""),
            "neighbor_count": len(neighbor_papers),
            "shared_facets": buckets["shared_facets"],
            "unique_facets": buckets["unique_facets"],
            "disallowed_terms": buckets["disallowed_terms"],
            "facet_weights": buckets["facet_weights"],
        }

        all_results.append(result)

        # Print summary for this paper
        print(
            f"  [OK] Shared: {len(result['shared_facets'])}, "
            + f"Unique: {len(result['unique_facets'])}, "
            + f"Disallowed: {len(result['disallowed_terms'])}"
        )

    # Save all results to JSONL file
    output_file = Path("data", "facet_analysis_results.jsonl")
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    # Save summary statistics
    summary = {
        "total_papers": len(papers_cache),
        "processed_papers": len(all_results),
        "skipped_papers": len(skipped_papers),
        "skipped_details": skipped_papers,
        "average_neighbors": sum(r["neighbor_count"] for r in all_results)
        / len(all_results)
        if all_results
        else 0,
        "average_shared_facets": sum(len(r["shared_facets"]) for r in all_results)
        / len(all_results)
        if all_results
        else 0,
        "average_unique_facets": sum(len(r["unique_facets"]) for r in all_results)
        / len(all_results)
        if all_results
        else 0,
        "average_disallowed_terms": sum(len(r["disallowed_terms"]) for r in all_results)
        / len(all_results)
        if all_results
        else 0,
    }

    summary_file = Path("data", "facet_analysis_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total papers: {summary['total_papers']}")
    print(f"Successfully processed: {summary['processed_papers']}")
    print(f"Skipped: {summary['skipped_papers']}")
    print(f"\nAverage neighbors per paper: {summary['average_neighbors']:.1f}")
    print(f"Average shared facets: {summary['average_shared_facets']:.1f}")
    print(f"Average unique facets: {summary['average_unique_facets']:.1f}")
    print(f"Average disallowed terms: {summary['average_disallowed_terms']:.1f}")
    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Summary saved to: {summary_file}")

    # Print example payload for first paper
    if all_results:
        print("\n" + "=" * 60)
        print("EXAMPLE PAYLOAD (first paper)")
        print("=" * 60)
        example = all_results[0]
        print(f"\nPaper ID: {example['paper_id']}")
        print(f"Title: {example['title']}")
        print(f"\nShared Facets ({len(example['shared_facets'])}):")
        for facet in example["shared_facets"]:
            print(f"  • {facet}")
        print(f"\nUnique Facets ({len(example['unique_facets'])}):")
        for facet in example["unique_facets"]:
            print(f"  • {facet}")
        print(f"\nDisallowed Terms ({len(example['disallowed_terms'])}):")
        for term in example["disallowed_terms"]:
            print(f"  • {term}")
        print("\nFacet Weights:")
        for key, value in example["facet_weights"].items():
            print(f"  • {key}: {value}")
