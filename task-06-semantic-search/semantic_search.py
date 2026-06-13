"""
CoreTech Innovations — Semantic Search System
Task 06: Semantic Search for CoreTech Services
Intern: Muhammad Taha | AI Engineering Internship
Company: CoreTech Innovations (coretechio.com)

Description:
    This script builds a semantic search system over the CoreTech Innovations
    knowledge base using TF-IDF vectorization and Cosine Similarity scoring.
    Given a user query, it returns the top 3 most relevant records with scores.

Libraries Used:
    - pandas      : Load and manage the CSV knowledge base dataset
    - numpy       : Numerical operations on similarity score arrays
    - scikit-learn: TfidfVectorizer for text vectorization,
                    cosine_similarity for similarity computation
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Path to the knowledge base CSV file
KB_FILE       = "coretech_knowledge_base.csv"

# Number of top results to return per query
TOP_N         = 3

# Minimum similarity score threshold (0.0 to 1.0)
# Results below this score are considered not relevant
MIN_SCORE     = 0.02

# Column used as the searchable text corpus
TEXT_COLUMN   = "text"

# ─── STEP 1: LOAD KNOWLEDGE BASE ──────────────────────────────────────────────

def load_knowledge_base(filepath: str) -> pd.DataFrame:
    """
    Load the CoreTech knowledge base from a CSV file into a Pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded knowledge base with all records

    Raises:
        FileNotFoundError: If the CSV file does not exist
        ValueError: If the required 'text' column is missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Knowledge base file '{filepath}' not found. "
            f"Make sure coretech_knowledge_base.csv is in the same directory."
        )

    # Load CSV into DataFrame using pandas
    df = pd.DataFrame(pd.read_csv(filepath))

    # Validate required column exists
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COLUMN}' not found in {filepath}. "
            f"Expected columns: {list(df.columns)}"
        )

    # Drop any rows where text is empty or null
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df[df[TEXT_COLUMN].str.strip() != ""]
    df = df.reset_index(drop=True)

    return df


# ─── STEP 2: BUILD TF-IDF MATRIX ──────────────────────────────────────────────

def build_tfidf_matrix(df: pd.DataFrame):
    """
    Build a TF-IDF matrix from the knowledge base text corpus.

    TF-IDF (Term Frequency-Inverse Document Frequency) converts each text
    record into a numerical vector. Words that appear frequently in one
    document but rarely across others get higher scores, making them
    better discriminators for search.

    Args:
        df (pd.DataFrame): Knowledge base DataFrame

    Returns:
        tuple: (vectorizer, tfidf_matrix)
            - vectorizer    : Fitted TfidfVectorizer instance
            - tfidf_matrix  : Sparse matrix of shape (n_records, n_features)
    """
    # Initialize TF-IDF Vectorizer with English stop word removal
    # stop_words='english' removes common words like 'the', 'is', 'and'
    # ngram_range=(1, 2) captures both single words and 2-word phrases
    # max_df=0.95 ignores terms that appear in more than 95% of documents
    # min_df=1 requires a term to appear in at least 1 document
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1
    )

    # Fit the vectorizer on the corpus and transform to TF-IDF matrix
    # Shape: (number of records, number of unique terms)
    tfidf_matrix = vectorizer.fit_transform(df[TEXT_COLUMN].tolist())

    return vectorizer, tfidf_matrix


# ─── STEP 3: SEMANTIC SEARCH FUNCTION ────────────────────────────────────────

def semantic_search(query: str, df: pd.DataFrame, vectorizer: TfidfVectorizer,
                    tfidf_matrix, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Search the knowledge base for the most relevant records to a user query.

    Steps:
        1. Transform the query using the same TF-IDF vectorizer
        2. Compute Cosine Similarity between the query vector and all records
        3. Rank records by similarity score (descending)
        4. Return the top N results above the minimum score threshold

    Cosine Similarity measures the angle between two vectors in TF-IDF space.
    A score of 1.0 means identical direction (perfect match).
    A score of 0.0 means completely orthogonal (no shared terms).

    Args:
        query        (str)           : User search query
        df           (pd.DataFrame)  : Knowledge base DataFrame
        vectorizer   (TfidfVectorizer): Fitted TF-IDF vectorizer
        tfidf_matrix                 : Pre-built TF-IDF matrix of the corpus
        top_n        (int)           : Number of top results to return

    Returns:
        pd.DataFrame: Top N results with similarity scores, or empty DataFrame
    """
    if not query.strip():
        return pd.DataFrame()

    # Transform the user query into a TF-IDF vector using the fitted vectorizer
    query_vector = vectorizer.transform([query])

    # Compute Cosine Similarity between the query vector and all document vectors
    # Result shape: (1, n_records) — one score per knowledge base record
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Flatten the 2D numpy array to a 1D array of scores
    scores_array = similarity_scores.flatten()

    # Get indices of top N scores using numpy argsort (ascending), then reverse
    top_indices = np.argsort(scores_array)[::-1][:top_n]

    # Build results DataFrame with matched records and their scores
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = np.round(scores_array[top_indices], 4)

    # Filter out results below the minimum relevance threshold
    results = results[results['similarity_score'] >= MIN_SCORE]
    results = results.reset_index(drop=True)

    return results


# ─── STEP 4: DISPLAY RESULTS ─────────────────────────────────────────────────

def display_results(query: str, results: pd.DataFrame) -> None:
    """
    Print search results in a clean, readable format.

    Args:
        query   (str)          : The original user query
        results (pd.DataFrame) : Search results with similarity scores
    """
    print(f"\n{'─' * 62}")
    print(f"  Query: \"{query}\"")
    print(f"{'─' * 62}")

    if results.empty:
        print("  No relevant results found. Try rephrasing your query.")
        print(f"{'─' * 62}\n")
        return

    for i, row in results.iterrows():
        rank  = i + 1
        score = row['similarity_score']

        # Visual relevance indicator using numpy for score bucketing
        score_bar = "█" * int(np.round(score * 10))
        relevance = "High" if score >= 0.3 else "Medium" if score >= 0.1 else "Low"

        print(f"\n  Result #{rank}  |  Score: {score:.4f}  [{score_bar:<10}]  Relevance: {relevance}")
        print(f"  Category : {row.get('category', 'N/A')}")
        print(f"  Title    : {row.get('title', 'N/A')}")
        # Truncate long text for display (show first 200 characters)
        text_preview = str(row.get('text', ''))[:200]
        if len(str(row.get('text', ''))) > 200:
            text_preview += "..."
        print(f"  Preview  : {text_preview}")

    print(f"\n{'─' * 62}\n")


# ─── STEP 5: DATASET SUMMARY ─────────────────────────────────────────────────

def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the loaded knowledge base using pandas operations.

    Args:
        df (pd.DataFrame): Loaded knowledge base DataFrame
    """
    print("\n  Knowledge Base Summary")
    print(f"  {'─' * 40}")
    print(f"  Total Records  : {len(df)}")
    print(f"  Columns        : {list(df.columns)}")

    # Use pandas value_counts to show category distribution
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        print(f"\n  Records by Category:")
        for cat, count in category_counts.items():
            print(f"    {cat:<35} : {count}")

    # Use numpy to calculate average text length in characters
    text_lengths = np.array([len(str(t)) for t in df[TEXT_COLUMN]])
    print(f"\n  Avg Text Length : {np.mean(text_lengths):.0f} characters")
    print(f"  Max Text Length : {np.max(text_lengths)} characters")
    print(f"  Min Text Length : {np.min(text_lengths)} characters")
    print(f"  {'─' * 40}\n")


# ─── STEP 6: RUN TEST QUERIES ─────────────────────────────────────────────────

def run_demo_queries(df: pd.DataFrame, vectorizer, tfidf_matrix) -> None:
    """
    Run a set of predefined demo queries to demonstrate the search system.
    These test cases cover all major categories in the knowledge base.

    Args:
        df           (pd.DataFrame)   : Knowledge base DataFrame
        vectorizer   (TfidfVectorizer): Fitted TF-IDF vectorizer
        tfidf_matrix                  : Pre-built TF-IDF matrix
    """
    demo_queries = [
        "web development services for enterprise",
        "how to apply for internship",
        "who founded CoreTech Innovations",
        "cybersecurity and data protection",
        "ERP systems for manufacturing",
        "cloud modernization project",
        "contact information and location",
        "client retention and project delivery stats",
        "mobile app development iOS Android",
        "what makes CoreTech different"
    ]

    print("\n" + "=" * 62)
    print("   DEMO QUERIES — Semantic Search Test Cases")
    print("=" * 62)

    for query in demo_queries:
        results = semantic_search(query, df, vectorizer, tfidf_matrix)
        display_results(query, results)


# ─── STEP 7: INTERACTIVE SEARCH LOOP ─────────────────────────────────────────

def run_interactive_search(df: pd.DataFrame, vectorizer, tfidf_matrix) -> None:
    """
    Run an interactive command-line search loop.
    Users can enter queries and get real-time results until they type 'exit'.

    Args:
        df           (pd.DataFrame)   : Knowledge base DataFrame
        vectorizer   (TfidfVectorizer): Fitted TF-IDF vectorizer
        tfidf_matrix                  : Pre-built TF-IDF matrix
    """
    print("\n" + "=" * 62)
    print("   CoreTech Innovations — Semantic Search")
    print("   Type your query to search the knowledge base.")
    print("   Type 'demo' to run test queries.")
    print("   Type 'stats' to view dataset summary.")
    print("   Type 'exit' to quit.")
    print("=" * 62 + "\n")

    while True:
        try:
            query = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting search. Goodbye!")
            break

        # Handle control commands
        if not query:
            print("  Please enter a search query.\n")
            continue

        if query.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using CoreTech Innovations Semantic Search. Goodbye!")
            break

        if query.lower() == 'demo':
            run_demo_queries(df, vectorizer, tfidf_matrix)
            continue

        if query.lower() == 'stats':
            print_dataset_summary(df)
            continue

        # Run semantic search for user query
        results = semantic_search(query, df, vectorizer, tfidf_matrix)
        display_results(query, results)


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────

def main():
    """
    Main function: loads data, builds TF-IDF index, runs search system.
    """
    print("=" * 62)
    print("   CoreTech Innovations — Semantic Search System")
    print("   TF-IDF + Cosine Similarity")
    print("   coretechio.com | Nawabshah & Hyderabad, Pakistan")
    print("=" * 62)

    # Step 1: Load knowledge base CSV using pandas
    print("\n[1/3] Loading knowledge base...")
    try:
        df = load_knowledge_base(KB_FILE)
        print(f"      Loaded {len(df)} records from '{KB_FILE}'")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    # Step 2: Build TF-IDF matrix using scikit-learn
    print("[2/3] Building TF-IDF matrix...")
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)
    print(f"      Matrix shape: {tfidf_matrix.shape} "
          f"({tfidf_matrix.shape[0]} records × {tfidf_matrix.shape[1]} terms)")

    # Step 3: Launch interactive search
    print("[3/3] System ready.\n")

    # Run interactive search loop
    run_interactive_search(df, vectorizer, tfidf_matrix)


if __name__ == "__main__":
    main()
