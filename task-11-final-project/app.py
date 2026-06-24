"""
CoreTech Innovations — RAG Knowledge Assistant
Final Project: AI Engineering Internship — Task 11
Intern: Muhammad Taha | coretechio.com

Description:
    A production-quality Retrieval-Augmented Generation (RAG) system
    deployed as a Gradio web application. Users can ask any question
    about CoreTech Innovations and receive grounded answers retrieved
    from a curated knowledge base using semantic embeddings and FAISS.

Embedding Model : all-MiniLM-L6-v2 (sentence-transformers)
Vector Store    : FAISS (Facebook AI Similarity Search)
UI Framework    : Gradio
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer
from rag_engine import (
    load_knowledge_base,
    build_embeddings,
    build_faiss_index,
    rag_pipeline
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

KB_FILE         = "coretech_knowledge_base.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── INITIALIZE RAG SYSTEM ────────────────────────────────────────────────────

print("Initializing CoreTech RAG Knowledge Assistant...")

# Step 1: Load knowledge base into pandas DataFrame
print("[1/3] Loading knowledge base...")
df = load_knowledge_base(KB_FILE)
print(f"      Loaded {len(df)} knowledge records across {df['category'].nunique()} categories.")

# Step 2: Load sentence transformer embedding model
print("[2/3] Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"      Model loaded. Embedding dimension: 384")

# Step 3: Build embeddings and FAISS vector store
print("[3/3] Building FAISS vector store...")
embeddings = build_embeddings(df, model)
index      = build_faiss_index(embeddings)
print(f"      FAISS index built. {index.ntotal} vectors stored.")
print("System ready.\n")

# ─── SUGGESTED QUESTIONS ──────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "What services does CoreTech Innovations offer?",
    "Who founded CoreTech Innovations?",
    "How much does a web development project cost?",
    "Does CoreTech offer cybersecurity services?",
    "How do I apply for a CoreTech internship?",
    "What is CoreTech's client retention rate?",
    "How do I start a project with CoreTech?",
    "What is the project delivery process at CoreTech?",
    "Does CoreTech build mobile apps?",
    "What industries does CoreTech serve?"
]

# ─── CHAT FUNCTION ────────────────────────────────────────────────────────────

def chat(user_message: str, history: list) -> tuple:
    """
    Main chat function called by Gradio on every user message.
    Runs the full RAG pipeline and formats the response.

    Args:
        user_message (str) : User's question
        history      (list): Gradio chat history

    Returns:
        tuple: (updated history, empty string to clear input)
    """
    if not user_message.strip():
        history.append((user_message, "Please enter a question."))
        return history, ""

    # Run full RAG pipeline
    result = rag_pipeline(user_message, df, model, index)

    # Format the answer with source attribution
    answer  = result["answer"]
    sources = result["sources"]
    scores  = result["scores"]

    # Build response with sources shown below the answer
    if sources:
        source_lines = "\n".join([
            f"  • {src} (similarity: {score:.2f})"
            for src, score in zip(sources, scores)
        ])
        formatted_response = (
            f"{answer}\n\n"
            f"---\n"
            f"📚 **Sources Used ({result['num_chunks_used']} chunks retrieved):**\n"
            f"{source_lines}"
        )
    else:
        formatted_response = answer

    # Append to chat history
    history.append((user_message, formatted_response))
    return history, ""


def clear_chat() -> tuple:
    """Clear the chat history."""
    return [], ""


def use_suggestion(suggestion: str) -> str:
    """Fill the input box with a suggested question."""
    return suggestion


# ─── GRADIO UI ────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="CoreTech RAG Assistant",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    )
) as app:

    # ── Header ──
    gr.Markdown("""
    # 🤖 CoreTech Innovations — RAG Knowledge Assistant
    ### AI-Powered Question Answering using Semantic Search + FAISS Vector Store
    **Embedding Model:** `all-MiniLM-L6-v2` (sentence-transformers) &nbsp;|&nbsp;
    **Vector Store:** FAISS &nbsp;|&nbsp;
    **Knowledge Base:** 55 curated records across 6 categories
    ---
    """)

    with gr.Row():

        # ── Left Column: Chat ──
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="CoreTech RAG Assistant",
                height=480,
                bubble_full_width=False,
                show_label=True
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about CoreTech Innovations...",
                    label="Your Question",
                    scale=5,
                    lines=1
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # ── Right Column: Suggestions + Info ──
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Suggested Questions")
            for question in SUGGESTED_QUESTIONS:
                btn = gr.Button(question, size="sm", variant="secondary")
                btn.click(
                    fn=use_suggestion,
                    inputs=[],
                    outputs=[msg_input]
                ).then(
                    fn=lambda q=question: q,
                    inputs=[],
                    outputs=[msg_input]
                )

            gr.Markdown("""
            ---
            ### 📊 System Info
            - **Records:** 55 knowledge entries
            - **Categories:** Company Info, Services, Projects, Pricing, Process, FAQ
            - **Retrieval:** Top 3 semantic matches
            - **Model:** all-MiniLM-L6-v2
            ---
            ### 📞 Contact CoreTech
            - 🌐 coretechio.com
            - 📧 hr@coretechio.com
            - 📱 +92 348 0394588
            """)

    # ── Event Handlers ──
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_input]
    )

    # ── Footer ──
    gr.Markdown("""
    ---
    *Built by Muhammad Taha — AI Engineering Intern, CoreTech Innovations*
    *Powered by sentence-transformers, FAISS, and Gradio*
    """)

# ─── LAUNCH ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.launch()
