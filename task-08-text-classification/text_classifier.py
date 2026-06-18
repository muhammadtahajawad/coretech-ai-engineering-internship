"""
CoreTech Innovations — Client Inquiry Text Classification System
Task 08: Client Inquiry Text Classification
Intern: Muhammad Taha | AI Engineering Internship
Company: CoreTech Innovations (coretechio.com)

Description:
    This script trains a text classification model to automatically
    categorize incoming client inquiry messages into 8 service categories.
    It uses TF-IDF for feature extraction and both Naive Bayes and
    Logistic Regression classifiers for comparison.

Libraries:
    - pandas      : Load and manage the dataset
    - numpy       : Numerical operations and array handling
    - scikit-learn: TF-IDF, Naive Bayes, Logistic Regression,
                    train_test_split, accuracy_score, classification_report
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ─── STEP 1: LOAD AND EXPLORE DATASET ────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the client inquiries CSV dataset into a pandas DataFrame.
    Validates required columns and prints a summary.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded and validated dataset
    """
    # Load CSV using pandas
    df = pd.read_csv(filepath)

    print("=== Dataset Loaded ===")
    print(f"  Total Records : {len(df)}")
    print(f"  Columns       : {list(df.columns)}")

    # Validate required columns exist
    required = ['message', 'category']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Drop null values
    df = df.dropna(subset=['message', 'category'])
    df = df.reset_index(drop=True)

    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """
    Explore and display dataset statistics using pandas and numpy.

    Args:
        df (pd.DataFrame): Client inquiries DataFrame
    """
    print("\n=== Dataset Exploration ===")

    # Category distribution using pandas value_counts
    print("\n  Category Distribution:")
    category_counts = df['category'].value_counts()
    for cat, count in category_counts.items():
        bar = "█" * count
        print(f"  {cat:<25} : {count:>3}  {bar}")

    # Message length statistics using numpy
    msg_lengths = np.array([len(str(m)) for m in df['message']])
    print(f"\n  Message Length Statistics:")
    print(f"  Average : {np.mean(msg_lengths):.1f} characters")
    print(f"  Max     : {np.max(msg_lengths)} characters")
    print(f"  Min     : {np.min(msg_lengths)} characters")
    print(f"  Std Dev : {np.std(msg_lengths):.1f} characters")

    # Show sample messages from each category
    print("\n  Sample Messages per Category:")
    for cat in df['category'].unique():
        sample = df[df['category'] == cat]['message'].iloc[0]
        print(f"  [{cat}]")
        print(f"    {sample}")


# ─── STEP 2: PREPARE FEATURES AND LABELS ─────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Extract features (X) and labels (y) from the DataFrame.
    Split into training and test sets using stratified sampling
    to ensure all categories are represented in both sets.

    Args:
        df (pd.DataFrame): Client inquiries DataFrame

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Extract feature column (message text) and label column (category)
    X = df['message'].values   # numpy array of message strings
    y = df['category'].values  # numpy array of category labels

    print("\n=== Data Preparation ===")
    print(f"  Total samples    : {len(X)}")
    print(f"  Unique categories: {len(np.unique(y))}")
    print(f"  Categories       : {list(np.unique(y))}")

    # Split dataset — 80% training, 20% test
    # stratify=y ensures proportional category representation in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\n  Training samples : {len(X_train)} (80%)")
    print(f"  Test samples     : {len(X_test)} (20%)")

    return X_train, X_test, y_train, y_test


# ─── STEP 3: TRAIN NAIVE BAYES CLASSIFIER ────────────────────────────────────

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Train a Multinomial Naive Bayes text classification model.

    Naive Bayes works by applying Bayes' theorem with the assumption
    that each word feature is independent. It works very well for
    text classification tasks and is fast to train.

    Pipeline:
        TF-IDF Vectorizer → Multinomial Naive Bayes Classifier

    Args:
        X_train, X_test : Message text arrays
        y_train, y_test : Category label arrays

    Returns:
        sklearn Pipeline: Trained Naive Bayes pipeline
    """
    print("\n" + "=" * 55)
    print("  MODEL 1: Multinomial Naive Bayes")
    print("=" * 55)

    # Build a pipeline: TF-IDF vectorization + Naive Bayes classification
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1
        )),
        ('classifier', MultinomialNB(alpha=0.1))
    ])

    # Train the model on training data
    nb_pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = nb_pipeline.predict(X_test)

    # Calculate and display accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy Score: {accuracy * 100:.2f}%")

    # Display full classification report (precision, recall, F1 per category)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Show correct vs incorrect predictions using numpy
    correct   = np.sum(y_pred == y_test)
    incorrect = np.sum(y_pred != y_test)
    print(f"  Correct Predictions  : {correct}")
    print(f"  Incorrect Predictions: {incorrect}")

    return nb_pipeline


# ─── STEP 4: TRAIN LOGISTIC REGRESSION CLASSIFIER ────────────────────────────

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train a Logistic Regression text classification model.

    Logistic Regression works by learning weights for each feature
    (TF-IDF term) per class, using gradient descent optimization.
    Often achieves higher accuracy than Naive Bayes on text tasks.

    Pipeline:
        TF-IDF Vectorizer → Logistic Regression Classifier

    Args:
        X_train, X_test : Message text arrays
        y_train, y_test : Category label arrays

    Returns:
        sklearn Pipeline: Trained Logistic Regression pipeline
    """
    print("\n" + "=" * 55)
    print("  MODEL 2: Logistic Regression")
    print("=" * 55)

    # Build a pipeline: TF-IDF vectorization + Logistic Regression
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        ))
    ])

    # Train the model on training data
    lr_pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = lr_pipeline.predict(X_test)

    # Calculate and display accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy Score: {accuracy * 100:.2f}%")

    # Display full classification report (precision, recall, F1 per category)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Show correct vs incorrect predictions using numpy
    correct   = np.sum(y_pred == y_test)
    incorrect = np.sum(y_pred != y_test)
    print(f"  Correct Predictions  : {correct}")
    print(f"  Incorrect Predictions: {incorrect}")

    return lr_pipeline


# ─── STEP 5: COMPARE MODELS ───────────────────────────────────────────────────

def compare_models(X_train, X_test, y_train, y_test,
                   nb_pipeline, lr_pipeline) -> str:
    """
    Compare Naive Bayes and Logistic Regression accuracy scores
    and recommend the better model.

    Args:
        X_train, X_test  : Message text arrays
        y_train, y_test  : Category label arrays
        nb_pipeline      : Trained Naive Bayes pipeline
        lr_pipeline      : Trained Logistic Regression pipeline

    Returns:
        str: Name of the better performing model
    """
    nb_acc = accuracy_score(y_test, nb_pipeline.predict(X_test))
    lr_acc = accuracy_score(y_test, lr_pipeline.predict(X_test))

    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(f"  Naive Bayes         : {nb_acc * 100:.2f}%")
    print(f"  Logistic Regression : {lr_acc * 100:.2f}%")

    # Use numpy to find the better model
    scores     = np.array([nb_acc, lr_acc])
    best_index = np.argmax(scores)
    best_model = ["Naive Bayes", "Logistic Regression"][best_index]

    print(f"\n  Best Model: {best_model} ({scores[best_index] * 100:.2f}%)")
    return best_model


# ─── STEP 6: TEST WITH NEW MESSAGES ──────────────────────────────────────────

def predict_new_messages(pipeline, model_name: str) -> None:
    """
    Test the trained model on brand new unseen client messages.
    Demonstrates real-world usage with varied phrasings.

    Args:
        pipeline   : Trained sklearn pipeline
        model_name : Name of the model for display
    """
    test_messages = [
        "I need a website for my business",
        "Can you build me an Android application?",
        "Our app design looks very outdated",
        "I want to run Facebook ads for my brand",
        "My website does not appear on Google",
        "We need a custom software system",
        "What services does CoreTech offer?",
        "The project you delivered was full of errors",
        "I want to develop an iOS and Android app together",
        "Can you help us rank higher on search engines?"
    ]

    print("\n" + "=" * 55)
    print(f"  TEST PREDICTIONS — {model_name}")
    print("=" * 55)

    for msg in test_messages:
        prediction   = pipeline.predict([msg])[0]
        # Get probability scores for confidence
        proba        = pipeline.predict_proba([msg])[0]
        confidence   = np.max(proba) * 100
        print(f"\n  Message    : {msg}")
        print(f"  Predicted  : {prediction}")
        print(f"  Confidence : {confidence:.1f}%")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  CoreTech Innovations — Text Classification System")
    print("  Client Inquiry Categorization using ML")
    print("  coretechio.com | Nawabshah & Hyderabad, Pakistan")
    print("=" * 55)

    # Step 1: Load dataset
    df = load_dataset("coretech_client_inquiries.csv")

    # Step 2: Explore dataset
    explore_dataset(df)

    # Step 3: Prepare train/test split
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 4: Train Naive Bayes
    nb_pipeline = train_naive_bayes(X_train, X_test, y_train, y_test)

    # Step 5: Train Logistic Regression
    lr_pipeline = train_logistic_regression(X_train, X_test, y_train, y_test)

    # Step 6: Compare models
    best_model_name = compare_models(
        X_train, X_test, y_train, y_test, nb_pipeline, lr_pipeline
    )

    # Step 7: Test with new messages using the best model
    best_pipeline = lr_pipeline if "Logistic" in best_model_name else nb_pipeline
    predict_new_messages(best_pipeline, best_model_name)

    print("\n" + "=" * 55)
    print("  Classification system complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
