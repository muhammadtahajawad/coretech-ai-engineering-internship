"""
CoreTech Innovations — Knowledge Base Analysis
Final Project: AI Engineering Internship — Task 11
Intern: Muhammad Taha | coretechio.com

Description:
    Standalone script to analyze and validate the CoreTech knowledge base.
    Shows dataset statistics, category distribution, embedding dimensions,
    keyword search vs semantic search comparison, and sample retrievals.
    Demonstrates the superiority of semantic search over keyword search.
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine
from rag_engine import (
    load_knowledge_base,
    build_embeddings,
    build_faiss_index,
    retrieve
)

# ─── STEP 1: LOAD AND ANALYZE KNOWLEDGE BASE ──────────────────────────────────

def analyze_knowledge_base(df: pd.DataFrame) -> None:
    """
    Analyze and display knowledge base statistics using pandas and numpy.

    Args:
        df (pd.DataFrame): Knowledge base DataFrame
    """
    print("=" * 60)
    print("  CoreTech Knowledge Base Analysis")
    print("=" * 60)

    print(f"\n  Total Records  : {len(df)}")
    print(f"  Columns        : {list(df.columns)}")

    # Category distribution using pandas value_counts
    print(f"\n  Records by Category:")
    category_counts = df["category"].value_counts()
    for cat, count in category_counts.items():
        bar = "█" * count
        print(f"  {cat:<20} : {count:>3}  {bar}")

    # Text length statistics using numpy
    lengths = np.array([len(str(t)) for t in df["text"]])
    print(f"\n  Text Length Statistics:")
    print(f"  Mean   : {np.mean(lengths):.0f} chars")
    print(f"  Max    : {np.max(lengths)} chars")
    print(f"  Min    : {np.min(lengths)} chars")
    print(f"  StdDev : {np.std(lengths):.0f} chars")


# ─── STEP 2: KEYWORD SEARCH (TF-IDF) ─────────────────────────────────────────

def keyword_search(query: str, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Perform keyword-based search using TF-IDF and cosine similarity.
    Used for comparison against semantic search.

    Args:
        query  (str)          : Search query
        df     (pd.DataFrame) : Knowledge base DataFrame
        top_k  (int)          : Number of results

    Returns:
        pd.DataFrame: Top K results with TF-IDF similarity scores
    """
    texts      = (df["title"] + ". " + df["text"]).tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_mat  = vectorizer.fit_transform(texts)
    query_vec  = vectorizer.transform([query])
    scores     = tfidf_cosine(query_vec, tfidf_mat).flatten()
    top_idx    = np.argsort(scores)[::-1][:top_k]
    results    = df.iloc[top_idx].copy()
    results["tfidf_score"] = np.round(scores[top_idx], 4)
    return results.reset_index(drop=True)


# ─── STEP 3: COMPARISON — KEYWORD VS SEMANTIC ────────────────────────────────

def compare_search_methods(queries: list, df: pd.DataFrame,
                            model: SentenceTransformer,
                            index) -> None:
    """
    Compare keyword search (TF-IDF) vs semantic search (sentence-transformers
    + FAISS) on the same queries to demonstrate semantic superiority.

    Args:
        queries (list)             : List of test queries
        df      (pd.DataFrame)     : Knowledge base DataFrame
        model   (SentenceTransformer): Embedding model
        index                      : FAISS vector store
    """
    print("\n" + "=" * 60)
    print("  Keyword Search vs Semantic Search Comparison")
    print("=" * 60)

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  {'─' * 55}")

        # Keyword search results
        kw_results = keyword_search(query, df, top_k=3)
        print(f"\n  [TF-IDF Keyword Search]")
        for i, row in kw_results.iterrows():
            print(f"  #{i+1} Score: {row['tfidf_score']:.4f} | {row['title']}")

        # Semantic search results
        sem_results = retrieve(query, df, model, index, top_k=3)
        print(f"\n  [Semantic Search — sentence-transformers + FAISS]")
        if sem_results.empty:
            print("  No results above threshold.")
        else:
            for i, row in sem_results.iterrows():
                print(f"  #{i+1} Score: {row['similarity_score']:.4f} | {row['title']}")

        print(f"\n  Insight: Semantic search captures meaning beyond exact keywords.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Load knowledge base
    df = load_knowledge_base("coretech_knowledge_base.csv")

    # Analyze dataset
    analyze_knowledge_base(df)

    # Load model and build index
    print("\n  Loading embedding model...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = build_embeddings(df, model)
    index      = build_faiss_index(embeddings)
    print(f"  FAISS index: {index.ntotal} vectors, {embeddings.shape[1]} dimensions")

    # Run comparison
    test_queries = [
        "how to contact the company",
        "pricing for small businesses",
        "security and data protection",
        "mobile application development"
    ]
    compare_search_methods(test_queries, df, model, index)

    print("\n  Knowledge base analysis complete.")


if __name__ == "__main__":
    main()
