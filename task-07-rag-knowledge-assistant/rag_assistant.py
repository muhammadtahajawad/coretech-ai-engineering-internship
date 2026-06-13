"""
CoreTech Innovations — RAG-Based Knowledge Assistant
Task 07: RAG-Style Knowledge Assistant
Intern: Muhammad Taha | AI Engineering Internship
Company: CoreTech Innovations (coretechio.com)

What is RAG?
    Retrieval-Augmented Generation (RAG) is an AI architecture that:
    1. RETRIEVES relevant documents from a knowledge base for a given query
    2. AUGMENTS the query with the retrieved context
    3. GENERATES a final answer grounded in the retrieved content

    This system implements RAG using:
    - TF-IDF + Cosine Similarity for the retrieval step
    - Template-based answer generation for the generation step
    - Source attribution to show which document was used

Libraries:
    - numpy       : Score arrays, argsort ranking, statistics
    - pandas      : Document management and results as DataFrames
    - scikit-learn: TfidfVectorizer and cosine_similarity
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Folder containing the 5 knowledge base documents
DOCS_FOLDER   = "documents"

# Number of top chunks to retrieve per query
TOP_K         = 3

# Minimum similarity score to consider a chunk relevant
MIN_SCORE     = 0.02

# Size of each text chunk in characters
CHUNK_SIZE    = 400

# Overlap between chunks to avoid cutting context
CHUNK_OVERLAP = 80

# ─── STEP 1: LOAD DOCUMENTS ───────────────────────────────────────────────────

def load_documents(folder: str) -> dict:
    """
    Load all .txt knowledge base documents from the documents folder.
    Each file becomes one named document in the knowledge base.

    Args:
        folder (str): Path to the documents folder

    Returns:
        dict: {filename: content} mapping for all loaded documents
    """
    documents = {}

    if not os.path.exists(folder):
        print(f"[ERROR] Documents folder '{folder}' not found.")
        return documents

    # Iterate over all .txt files in the documents folder
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # Use filename without extension as the document name
            doc_name = filename.replace(".txt", "").replace("_", " ").title()
            documents[doc_name] = content
            print(f"  Loaded: {doc_name} ({len(content)} characters)")

    return documents


# ─── STEP 2: CHUNK DOCUMENTS ──────────────────────────────────────────────────

def chunk_documents(documents: dict) -> pd.DataFrame:
    """
    Split each document into overlapping chunks for finer-grained retrieval.
    Chunking ensures long documents don't get averaged into one vector —
    specific sections can be retrieved individually.

    Args:
        documents (dict): {doc_name: content} mapping

    Returns:
        pd.DataFrame: DataFrame with columns [chunk_id, doc_name, chunk_text]
    """
    rows = []
    chunk_id = 0

    for doc_name, content in documents.items():
        # Split into chunks with overlap
        start = 0
        while start < len(content):
            end   = start + CHUNK_SIZE
            chunk = content[start:end].strip()

            # Only keep chunks with meaningful content (more than 30 characters)
            if len(chunk) > 30:
                rows.append({
                    "chunk_id"  : chunk_id,
                    "doc_name"  : doc_name,
                    "chunk_text": chunk
                })
                chunk_id += 1

            # Move forward by chunk size minus overlap
            start += CHUNK_SIZE - CHUNK_OVERLAP

    # Build pandas DataFrame from the list of chunk records
    df = pd.DataFrame(rows)
    return df


# ─── STEP 3: BUILD TF-IDF INDEX ───────────────────────────────────────────────

def build_index(df: pd.DataFrame):
    """
    Build a TF-IDF index over all document chunks.
    This is the core of the retrieval system — it converts each
    chunk into a numerical vector for similarity comparison.

    Args:
        df (pd.DataFrame): DataFrame with chunk_text column

    Returns:
        tuple: (vectorizer, tfidf_matrix)
    """
    # TF-IDF Vectorizer with unigrams and bigrams, stop words removed
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1
    )

    # Fit on all chunk texts and transform to sparse TF-IDF matrix
    # Shape: (number of chunks, number of unique terms)
    tfidf_matrix = vectorizer.fit_transform(df["chunk_text"].tolist())

    return vectorizer, tfidf_matrix


# ─── STEP 4: RETRIEVE RELEVANT CHUNKS ─────────────────────────────────────────

def retrieve(query: str, df: pd.DataFrame, vectorizer,
             tfidf_matrix, top_k: int = TOP_K) -> pd.DataFrame:
    """
    RETRIEVAL STEP — Find the most relevant chunks for the user query.

    Process:
        1. Transform query into TF-IDF vector
        2. Compute Cosine Similarity against all chunk vectors
        3. Rank by score, return top K chunks with scores

    Args:
        query        (str)          : User question
        df           (pd.DataFrame) : Chunks DataFrame
        vectorizer                  : Fitted TfidfVectorizer
        tfidf_matrix                : Pre-built TF-IDF matrix
        top_k        (int)          : Number of chunks to retrieve

    Returns:
        pd.DataFrame: Top K relevant chunks with similarity scores
    """
    # Transform the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Compute Cosine Similarity between query and all document chunks
    scores       = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Use numpy argsort to rank chunks by score (descending)
    top_indices  = np.argsort(scores)[::-1][:top_k]

    # Build results DataFrame with retrieved chunks and their scores
    results      = df.iloc[top_indices].copy()
    results["similarity_score"] = np.round(scores[top_indices], 4)

    # Filter out chunks below the minimum relevance threshold
    results = results[results["similarity_score"] >= MIN_SCORE]
    results = results.reset_index(drop=True)

    return results


# ─── STEP 5: GENERATE ANSWER ──────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: pd.DataFrame) -> str:
    """
    GENERATION STEP — Synthesize a final answer from retrieved chunks.

    In a full RAG system this step would call an LLM (GPT, Claude, LLaMA).
    Here we implement template-based generation that:
    - Combines the top retrieved chunks as context
    - Identifies the primary source document
    - Formats a structured, readable answer with source attribution

    Args:
        query            (str)          : Original user question
        retrieved_chunks (pd.DataFrame) : Top retrieved chunks

    Returns:
        str: Generated answer with source attribution
    """
    if retrieved_chunks.empty:
        return (
            "I could not find relevant information for your query in the "
            "CoreTech knowledge base. Please try rephrasing your question "
            "or contact CoreTech directly at hr@coretechio.com."
        )

    # Identify the primary source document (highest scoring chunk)
    primary_source = retrieved_chunks.iloc[0]["doc_name"]
    primary_score  = retrieved_chunks.iloc[0]["similarity_score"]

    # Collect all unique source documents used
    all_sources = retrieved_chunks["doc_name"].unique().tolist()

    # Build the context string from top retrieved chunks
    context_parts = []
    for _, row in retrieved_chunks.iterrows():
        # Clean up chunk text — remove leading/trailing whitespace
        chunk = row["chunk_text"].strip()
        context_parts.append(chunk)

    # Join chunks into a single context block
    context = "\n\n".join(context_parts)

    # Generate the final answer using the retrieved context
    answer = (
        f"{context}\n\n"
        f"[Additional context from: {', '.join(all_sources)}]"
        if len(all_sources) > 1
        else context
    )

    return answer


# ─── STEP 6: FULL RAG PIPELINE ────────────────────────────────────────────────

def rag_query(query: str, df: pd.DataFrame, vectorizer,
              tfidf_matrix) -> dict:
    """
    Full RAG Pipeline — combines retrieval and generation into one call.

    Args:
        query        (str)          : User question
        df           (pd.DataFrame) : Chunks DataFrame
        vectorizer                  : Fitted TfidfVectorizer
        tfidf_matrix                : TF-IDF matrix

    Returns:
        dict: {query, retrieved_chunks, answer, sources, top_score}
    """
    # Step 1: Retrieve relevant chunks
    retrieved = retrieve(query, df, vectorizer, tfidf_matrix)

    # Step 2: Generate answer from retrieved context
    answer    = generate_answer(query, retrieved)

    # Step 3: Extract source documents used
    sources   = retrieved["doc_name"].unique().tolist() if not retrieved.empty else []

    # Step 4: Get top similarity score using numpy
    top_score = float(np.max(retrieved["similarity_score"])) if not retrieved.empty else 0.0

    return {
        "query"            : query,
        "retrieved_chunks" : retrieved,
        "answer"           : answer,
        "sources"          : sources,
        "top_score"        : round(top_score, 4)
    }


# ─── STEP 7: DISPLAY RAG RESULT ───────────────────────────────────────────────

def display_rag_result(result: dict) -> None:
    """
    Display the full RAG result in a clean, structured format.
    Shows: query, retrieved chunks with scores, sources used, final answer.

    Args:
        result (dict): Output from rag_query()
    """
    print(f"\n{'═' * 65}")
    print(f"  QUERY: {result['query']}")
    print(f"{'═' * 65}")

    # Show retrieved chunks with source attribution
    chunks = result["retrieved_chunks"]
    if chunks.empty:
        print("  No relevant chunks retrieved.")
    else:
        print(f"\n  RETRIEVED CHUNKS ({len(chunks)} found):")
        print(f"  {'─' * 60}")
        for i, row in chunks.iterrows():
            score     = row["similarity_score"]
            bar       = "█" * int(np.round(score * 10))
            relevance = "High" if score >= 0.3 else "Medium" if score >= 0.1 else "Low"
            print(f"\n  Chunk #{i+1}")
            print(f"  Source Document : {row['doc_name']}")
            print(f"  Similarity Score: {score:.4f}  [{bar:<10}]  {relevance}")
            preview = row["chunk_text"][:200] + ("..." if len(row["chunk_text"]) > 200 else "")
            print(f"  Content Preview : {preview}")

    # Show source attribution
    print(f"\n  {'─' * 60}")
    print(f"  SOURCES USED: {', '.join(result['sources']) if result['sources'] else 'None'}")
    print(f"  TOP SCORE   : {result['top_score']}")

    # Show the generated answer
    print(f"\n  GENERATED ANSWER:")
    print(f"  {'─' * 60}")
    # Indent answer lines for readability
    for line in result["answer"].split("\n"):
        print(f"  {line}")
    print(f"{'═' * 65}\n")


# ─── STEP 8: DEMO TEST QUERIES ────────────────────────────────────────────────

def run_demo(df: pd.DataFrame, vectorizer, tfidf_matrix) -> None:
    """
    Run predefined demo queries covering all 5 knowledge base documents.
    These demonstrate the full RAG pipeline on real CoreTech questions.
    """
    demo_queries = [
        "Who founded CoreTech Innovations and who leads the company?",
        "What web development and mobile app services does CoreTech offer?",
        "What is the project delivery process at CoreTech?",
        "How much does a web development project cost at CoreTech?",
        "How do I apply for a CoreTech internship?",
        "What is CoreTech's client retention rate?",
        "Does CoreTech offer cybersecurity services?",
        "How can I contact CoreTech Innovations?"
    ]

    print("\n" + "=" * 65)
    print("   RAG DEMO — 8 Test Queries Across All 5 Documents")
    print("=" * 65)

    for query in demo_queries:
        result = rag_query(query, df, vectorizer, tfidf_matrix)
        display_rag_result(result)


# ─── STEP 9: INTERACTIVE LOOP ─────────────────────────────────────────────────

def run_interactive(df: pd.DataFrame, vectorizer, tfidf_matrix) -> None:
    """
    Interactive command-line RAG assistant.
    User types questions and gets answers with source attribution.
    """
    print("\n" + "=" * 65)
    print("   CoreTech Innovations — RAG Knowledge Assistant")
    print("   Ask any question about CoreTech services and company.")
    print("   Commands: 'demo' | 'stats' | 'exit'")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("Your Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        if query.lower() in ["exit", "quit", "q"]:
            print("Thank you for using CoreTech RAG Assistant. Goodbye!")
            break

        if query.lower() == "demo":
            run_demo(df, vectorizer, tfidf_matrix)
            continue

        if query.lower() == "stats":
            # Show dataset statistics using pandas and numpy
            print(f"\n  Knowledge Base Statistics")
            print(f"  Total Chunks  : {len(df)}")
            doc_counts = df["doc_name"].value_counts()
            print(f"  Chunks per Document:")
            for doc, count in doc_counts.items():
                print(f"    {doc:<30}: {count} chunks")
            lengths = np.array([len(t) for t in df["chunk_text"]])
            print(f"  Avg Chunk Size: {np.mean(lengths):.0f} chars")
            print()
            continue

        # Run full RAG pipeline
        result = rag_query(query, df, vectorizer, tfidf_matrix)
        display_rag_result(result)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("   CoreTech Innovations — RAG Knowledge Assistant")
    print("   Retrieval-Augmented Generation System")
    print("   coretechio.com | Nawabshah & Hyderabad, Pakistan")
    print("=" * 65)

    # Step 1: Load all 5 knowledge base documents
    print("\n[1/4] Loading knowledge base documents...")
    documents = load_documents(DOCS_FOLDER)
    if not documents:
        print("[ERROR] No documents loaded. Check the 'documents' folder.")
        return
    print(f"      {len(documents)} documents loaded.")

    # Step 2: Chunk documents into smaller pieces
    print("[2/4] Chunking documents...")
    df = chunk_documents(documents)
    print(f"      {len(df)} chunks created from {len(documents)} documents.")

    # Step 3: Build TF-IDF index
    print("[3/4] Building TF-IDF index...")
    vectorizer, tfidf_matrix = build_index(df)
    print(f"      Index shape: {tfidf_matrix.shape} "
          f"({tfidf_matrix.shape[0]} chunks × {tfidf_matrix.shape[1]} terms)")

    # Step 4: Launch interactive assistant
    print("[4/4] System ready.\n")
    run_interactive(df, vectorizer, tfidf_matrix)


if __name__ == "__main__":
    main()
