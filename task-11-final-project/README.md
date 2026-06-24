# Task 11: Final AI Engineering Project
# CoreTech Innovations RAG Knowledge Assistant

**Intern:** Muhammad Taha Jawad  
**Company:** CoreTech Innovations  
**Internship Track:** AI Engineering  
**Tools:** Python 3, sentence-transformers, FAISS, Gradio, pandas, numpy  
**Live Demo:** [huggingface.co/spaces/mtahajawad/coretech-rag-assistant](https://huggingface.co/spaces/mtahajawad/coretech-rag-assistant)

---

## Project Overview

This project is a production quality Retrieval Augmented Generation (RAG) system
built for CoreTech Innovations. Users ask questions about CoreTech services,
pricing, projects, and company info, the system retrieves the most semantically
relevant knowledge base chunks using real sentence embeddings and returns a
grounded, sourced answer through a professional Gradio web interface.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is an AI architecture that improves answer
quality by grounding responses in a real knowledge base rather than relying on
model memory alone. It has three steps:

1. **Retrieve**: Find the most semantically relevant knowledge chunks for the query
2. **Augment**: Add retrieved context to the generation process
3. **Generate**: Produce a final answer grounded in retrieved content with source attribution

---

## System Architecture
### Architecture Workflow

### Pipeline Architecture

1. **User Question** – Input query from the user.
2. **Embedding Generation** – `all-MiniLM-L6-v2` encodes the query into a 384-dimensional vector.
3. **Vector Search** – FAISS (`IndexFlatIP`) scans 55 knowledge vectors using inner product similarity.
4. **Retrieval** – Extracts the top 3 semantically relevant chunks with similarity scores.
5. **Generation** – Synthesizes the retrieved chunks into a grounded response.
6. **UI Deployment** – Displays the final answer and source attribution via Gradio.



### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Gradio app locally
```bash
python app.py
```

### 4. Run knowledge base analysis
```bash
python knowledge_base.py
```

### 5. Open in browser
http://localhost:7860


---

## Files in This Folder

| File | Description |
|---|---|
| `app.py` | Main Gradio web application — full RAG pipeline + UI |
| `rag_engine.py` | Core RAG engine — embeddings, FAISS, retrieval, generation |
| `knowledge_base.py` | Knowledge base analysis + keyword vs semantic comparison |
| `coretech_knowledge_base.csv` | 55-record curated knowledge base |
| `requirements.txt` | All Python dependencies |

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `sentence-transformers` | 2.7.0 | Semantic embedding model (all-MiniLM-L6-v2) |
| `faiss-cpu` | 1.8.0 | FAISS vector store for similarity search |
| `gradio` | 4.44.0 | Professional web UI and deployment |
| `pandas` | 2.2.2 | Knowledge base loading and management |
| `numpy` | 1.26.4 | Embedding arrays, similarity scores, statistics |
| `torch` | 2.3.0 | Backend for sentence-transformers |

---

## Real-World Business Application

This RAG system directly addresses a real CoreTech business need:

**Problem:** CoreTech's sales and support teams spend significant time answering
repetitive questions about services, pricing, and processes from potential clients.

**Solution:** This RAG assistant can be embedded on coretechio.com to:
- Answer client questions instantly 24/7 without human intervention
- Reduce support team workload by handling common inquiries automatically
- Provide sourced, accurate answers grounded in real company knowledge
- Route complex inquiries to the right team with context already extracted
- Scale to handle thousands of simultaneous client queries

**Estimated Impact:**
- 60-70% reduction in repetitive support queries handled manually
- Faster response time for potential clients (instant vs hours)
- Higher lead conversion through immediate, accurate information delivery

---

## Keyword vs Semantic Search Comparison

Run `knowledge_base.py` to see a full comparison. Sample result:

**Query:** `"how to reach CoreTech"`

| Method | Top Result | Score |
|---|---|---|
| TF-IDF Keyword | May return unrelated records | Low |
| Semantic Search | CoreTech Contact Information | High |

**Query:** `"cost of building a website"`

| Method | Top Result | Score |
|---|---|---|
| TF-IDF Keyword | May miss pricing records | Low |
| Semantic Search | Pricing Models + Starter Package | High |

---

## Screenshots

### Live Application
![App Screenshot](screenshots/app-screenshot.png)

### RAG Response with Source Attribution
![RAG Response](screenshots/rag-response.png)

### Knowledge Base Analysis
![KB Analysis](screenshots/kb-analysis.png)

---

## Live Demo

🚀 **[Launch Live App](https://huggingface.co/spaces/mtahajawad/coretech-rag-assistant)**

---

*Muhammad Taha Jawad — AI Engineering Intern, CoreTech Innovations*  
