"""
CoreTech Innovations — RAG Engine
Final Project: CoreTech RAG Knowledge Assistant
Intern: Muhammad Taha | AI Engineering Internship
Company: CoreTech Innovations (coretechio.com)

Description:
    Core RAG engine that handles:
    1. Loading and chunking the knowledge base
    2. Building semantic embeddings using sentence-transformers
    3. Storing vectors in a FAISS vector store
    4. Retrieving relevant chunks using cosine similarity
    5. Generating grounded answers with source attribution

Embedding Model: all-MiniLM-L6-v2 (sentence-transformers)
Vector Store   : FAISS (Facebook AI Similarity Search)
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Sentence transformer model — lightweight, fast, high quality embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Knowledge base file path
KB_FILE         = "coretech_knowledge_base.csv"

# Number of top chunks to retrieve per query
TOP_K           = 3

# Minimum similarity score threshold (cosine similarity 0-1)
MIN_SIMILARITY  = 0.20

# ─── STEP 1: LOAD KNOWLEDGE BASE ──────────────────────────────────────────────

def load_knowledge_base(filepath: str) -> pd.DataFrame:
    """
    Load the CoreTech knowledge base CSV into a pandas DataFrame.
    Each row represents one knowledge record with title and text.

    Args:
        filepath (str): Path to the knowledge base CSV

    Returns:
        pd.DataFrame: Loaded knowledge base
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["text"])
    df = df.reset_index(drop=True)
    return df


# ─── STEP 2: BUILD EMBEDDINGS ─────────────────────────────────────────────────

def build_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """
    Generate semantic embeddings for all knowledge base records
    using the sentence-transformers model (all-MiniLM-L6-v2).

    Sentence transformers convert text into dense vector representations
    that capture semantic meaning — unlike TF-IDF which only captures
    keyword frequency. Similar meanings produce similar vectors even
    when exact words differ.

    Args:
        df    (pd.DataFrame)      : Knowledge base DataFrame
        model (SentenceTransformer): Loaded embedding model

    Returns:
        np.ndarray: Matrix of embeddings, shape (n_records, embedding_dim)
    """
    # Combine title and text for richer embeddings
    texts = (df["title"] + ". " + df["text"]).tolist()

    # Generate embeddings — returns numpy array of shape (n, 384)
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # normalize for cosine similarity
    )

    return embeddings.astype(np.float32)


# ─── STEP 3: BUILD FAISS VECTOR STORE ────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS vector store (IndexFlatIP) from the knowledge base embeddings.

    FAISS (Facebook AI Similarity Search) is an industry-standard library
    for efficient similarity search over dense vectors. IndexFlatIP uses
    inner product (dot product) similarity — equivalent to cosine similarity
    when vectors are normalized.

    Args:
        embeddings (np.ndarray): Normalized embedding matrix

    Returns:
        faiss.IndexFlatIP: Built and populated FAISS index
    """
    # Get embedding dimension (384 for all-MiniLM-L6-v2)
    embedding_dim = embeddings.shape[1]

    # Create FAISS inner product index
    # IndexFlatIP = exact search using inner product (cosine sim for normalized vectors)
    index = faiss.IndexFlatIP(embedding_dim)

    # Add all knowledge base embeddings to the index
    index.add(embeddings)

    return index


# ─── STEP 4: RETRIEVE RELEVANT CHUNKS ─────────────────────────────────────────

def retrieve(query: str, df: pd.DataFrame, model: SentenceTransformer,
             index: faiss.IndexFlatIP, top_k: int = TOP_K) -> pd.DataFrame:
    """
    RETRIEVAL STEP — Find the most semantically relevant knowledge base
    records for the user query using FAISS similarity search.

    Process:
        1. Encode the query into a semantic embedding vector
        2. Search the FAISS index for the top_k nearest vectors
        3. Return matched records with similarity scores

    Args:
        query  (str)               : User question
        df     (pd.DataFrame)      : Knowledge base DataFrame
        model  (SentenceTransformer): Embedding model
        index  (faiss.IndexFlatIP) : FAISS vector store
        top_k  (int)               : Number of results to retrieve

    Returns:
        pd.DataFrame: Top K relevant records with similarity scores
    """
    if not query.strip():
        return pd.DataFrame()

    # Encode query into embedding vector
    # Shape: (1, 384) — normalized for cosine similarity
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    # Search FAISS index — returns distances and indices
    # distances: cosine similarity scores (higher = more similar)
    # indices  : positions in the knowledge base DataFrame
    distances, indices = index.search(query_embedding, top_k)

    # Flatten numpy arrays
    scores  = distances[0]
    idx     = indices[0]

    # Build results DataFrame with matched records
    results = df.iloc[idx].copy()
    results["similarity_score"] = np.round(scores, 4)

    # Filter by minimum similarity threshold
    results = results[results["similarity_score"] >= MIN_SIMILARITY]
    results = results.reset_index(drop=True)

    return results


# ─── STEP 5: GENERATE ANSWER ──────────────────────────────────────────────────

def generate_answer(query: str, retrieved: pd.DataFrame) -> dict:
    """
    GENERATION STEP — Synthesize a final grounded answer from
    the retrieved knowledge base chunks with source attribution.

    Args:
        query     (str)          : Original user question
        retrieved (pd.DataFrame) : Retrieved knowledge base records

    Returns:
        dict: {answer, sources, scores, num_chunks_used}
    """
    if retrieved.empty:
        return {
            "answer"         : (
                "I could not find relevant information for your question "
                "in the CoreTech knowledge base. Please try rephrasing, "
                "or contact CoreTech directly at hr@coretechio.com or "
                "+92 348 0394588."
            ),
            "sources"        : [],
            "scores"         : [],
            "num_chunks_used": 0
        }

    # Build grounded answer from retrieved chunks
    answer_parts  = []
    sources       = []
    scores        = []

    for _, row in retrieved.iterrows():
        answer_parts.append(row["text"])
        sources.append(f"{row['category']} — {row['title']}")
        scores.append(float(row["similarity_score"]))

    # Join retrieved chunks into a coherent answer
    answer = " ".join(answer_parts)

    return {
        "answer"         : answer,
        "sources"        : sources,
        "scores"         : scores,
        "num_chunks_used": len(retrieved)
    }


# ─── STEP 6: FULL RAG PIPELINE ────────────────────────────────────────────────

def rag_pipeline(query: str, df: pd.DataFrame, model: SentenceTransformer,
                 index: faiss.IndexFlatIP) -> dict:
    """
    Full RAG Pipeline — combines retrieval and generation.

    Args:
        query (str)               : User question
        df    (pd.DataFrame)      : Knowledge base DataFrame
        model (SentenceTransformer): Embedding model
        index (faiss.IndexFlatIP) : FAISS vector store

    Returns:
        dict: Complete RAG result with answer, sources, and scores
    """
    # Step 1: Retrieve relevant chunks from FAISS vector store
    retrieved = retrieve(query, df, model, index)

    # Step 2: Generate grounded answer from retrieved context
    result    = generate_answer(query, retrieved)

    # Add query to result
    result["query"] = query

    return result
