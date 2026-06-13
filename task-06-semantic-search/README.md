# Task 06: Semantic Search System  
**CoreTech Innovations — AI Engineering Internship**  
**Intern:** Muhammad Taha Jawad  

---

## Company: CoreTech Innovations  

## Internship Track: AI Engineering  

---

## Tools & Technologies
- Python 3  
- pandas  
- numpy  
- scikit-learn (TF-IDF, Cosine Similarity)  
- Google Colab  
- Jupyter Notebook  

---

## What is Semantic Search?

Semantic search retrieves results based on **meaning and context**, not exact keyword matching.

This system:
- Converts text into numerical vectors using **TF-IDF**
- Compares query and documents using **Cosine Similarity**
- Ranks results by relevance score

---

## System Architecture

User Query  
→ TfidfVectorizer.transform(query)  
→ Converts query into TF-IDF vector using trained vocabulary  
→ cosine_similarity(query_vector, tfidf_matrix)  
→ Computes similarity with all documents  
→ numpy.argsort(scores)[::-1][:3]  
→ Selects Top 3 most relevant results  
→ Returns ranked results with scores  

---

## How TF-IDF Works

- TF (Term Frequency): Frequency of a word in a document  
- IDF (Inverse Document Frequency): How rare the word is across all documents  
- TF-IDF = TF × IDF  

Rare but meaningful words get higher importance.

---

## How Cosine Similarity Works

Measures angle between two vectors:

- 1.0 → Perfect match  
- 0.5 → Partial match  
- 0.0 → No similarity  

Formula:  
cosine_similarity = (A · B) / (||A|| × ||B||)

---

## Project Files

| File | Description |
|------|-------------|
| coretech_knowledge_base.csv | 35-record knowledge base (services, FAQ, company info) |
| semantic_search.py | CLI-based semantic search engine |
| task_06_semantic_search.ipynb | Notebook implementation with test cases |

---

## Knowledge Base Overview

| Category     | Records | Content |
|--------------|--------|---------|
| Company Info | 9      | Mission, leadership, ESG, contact |
| Services     | 13     | Core services, finance, sales, support |
| Projects     | 3      | ApexOps, Meridian Health, onboarding |
| Internship   | 4      | Tracks, application process, interns |
| FAQ          | 6      | Clients, industries, cloud, features |

Total Records: 35  

---

## How to Run

### Option 1 — Google Colab
- Upload notebook and CSV file  
- Run all cells  

---

### Option 2 — Local Execution

Install dependencies:
- pip install scikit-learn pandas numpy  

Run system:
- python semantic_search.py  

Commands:
- demo  → Run test queries  
- stats → Dataset summary  
- exit  → Close program  

---

## Sample Output

Query: cybersecurity and data protection  

Result 1 | Score: 0.3637 | High Relevance  
Category: Services  
Title: Cybersecurity Service  
Preview: Enterprise-grade cybersecurity solutions...  

Result 2 | Score: 0.1450 | Medium Relevance  
Category: Company Info  
Title: CoreTech Why Choose Us  
Preview: Security and quality embedded early...  

Result 3 | Score: 0.0821 | Low Relevance  
Category: Services  
Title: Software Quality Services  
Preview: Testing and validation processes...  

---

## Libraries Used

- pandas → Data handling  
- numpy → Ranking & computations  
- scikit-learn → TF-IDF + Cosine Similarity  

---

## Environment

- OS: Windows 11  
- Python: 3.13  
- Platform: Google Colab / Jupyter Notebook  
- Repo: https://github.com/muhammadtahajawad/coretech-ai-engineering-internship  

---

## Author

Muhammad Taha Jawad  
AI Engineering Intern at CoreTech Innovations
