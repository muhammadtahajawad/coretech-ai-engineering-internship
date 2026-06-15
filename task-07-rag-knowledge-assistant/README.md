# Task 07 — RAG-Based Knowledge Assistant

**Intern:** Muhammad Taha  
**Company:** CoreTech Innovations (coretechio.com)  
**Internship Track:** AI Engineering  
**Tools:** Python 3, pandas, numpy, scikit-learn, Google Colab  

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is an AI architecture that improves answer quality by grounding responses in a real knowledge base rather than relying on model memory alone.

**Three steps:**
1. **Retrieve**: Find the most relevant document chunks for the query using TF-IDF and Cosine Similarity
2. **Augment**:Add retrieved context to the query
3. **Generate**: Produce a final answer grounded in the retrieved content, with source attribution

---

## System Architecture
User Question  -> TF-IDF Vectorizer (query → vector) -> Cosine Similarity (query vector vs all chunk vectors) -> Top K Chunks Retrieved (ranked by similarity score) -> Answer Generation (context synthesis + source attribution)
Final Answer + Source Document(s) Shown

---

## Knowledge Base Documents (5 Files)

| File | Content |
|---|---|
| `company_profile.txt` | Founders, stats, mission, values, contact, ESG |
| `services.txt` | All 6 core services + additional service categories |
| `project_process.txt` | 6-phase delivery process from discovery to maintenance |
| `pricing_sample.txt` | 3 pricing models + sample tier estimates |
| `faqs.txt` | 15 frequently asked questions with answers |

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy

# Run the assistant
python rag_assistant.py

# Commands:
# demo  — run 8 test queries across all documents
# stats — view chunk statistics
# exit  — quit
```

---

## Files in This Folder

| File | Description |
|---|---|
| `rag_assistant.py` | Main RAG system — retrieval + generation + interactive loop |
| `documents/company_profile.txt` | Knowledge base document 1 |
| `documents/services.txt` | Knowledge base document 2 |
| `documents/project_process.txt` | Knowledge base document 3 |
| `documents/pricing_sample.txt` | Knowledge base document 4 |
| `documents/faqs.txt` | Knowledge base document 5 |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Document chunking DataFrame, results management |
| `numpy` | Score arrays, argsort ranking, statistics |
| `scikit-learn` | TfidfVectorizer, cosine_similarity |

---

## Screenshots



<img width="731" height="607" alt="Screenshot 2026-06-14 001106" src="https://github.com/user-attachments/assets/d05f91cc-3325-4167-9553-512e38d9dc28" />
<img width="974" height="493" alt="Screenshot 2026-06-14 000922" src="https://github.com/user-attachments/assets/3f368368-2f69-4b58-8e08-ec2d2f215ac3" />


---

*Muhammad Taha — AI Engineering Intern, CoreTech Innovations*  
