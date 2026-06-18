# Task 08: Client Inquiry Text Classification System

**Intern:** Muhammad Taha Jawad

**Company:** CoreTech Innovations 
**Internship Track:** AI Engineering  
**Tools:** Python 3, pandas, numpy, scikit-learn, Google Colab  

---

## What is Text Classification?

Text classification is a supervised machine learning task where a model
learns to assign predefined categories to text inputs. In this task,
client inquiry messages are automatically categorized into 8 service
categories so CoreTech can route them to the right team instantly.

---

---

## Dataset

| Property | Details |
|---|---|
| File | `coretech_client_inquiries.csv` |
| Total Records | 80 messages |
| Train / Test Split | 80% / 20% |
| Categories | 8 (10 messages each) |

### Categories

| Category | Description |
|---|---|
| Web Development | Website and web app requests |
| App Development | Mobile app requests (iOS/Android) |
| UI/UX Design | Design and prototyping requests |
| Digital Marketing | Campaigns and social media requests |
| SEO | Search engine optimization requests |
| Software Solutions | Custom ERP, CRM, and software requests |
| General Inquiry | Company info and general questions |
| Complaint | Negative feedback and complaints |

---

## Models Used

### 1. Multinomial Naive Bayes
- Applies Bayes theorem with word independence assumption
- Fast, efficient, works well for text classification
- Parameter: `alpha=0.1` (Laplace smoothing)

### 2. Logistic Regression
- Learns feature weights per class using gradient descent
- Generally achieves higher accuracy on text tasks
- Parameters: `max_iter=1000`, `solver=lbfgs`, `multi_class=multinomial`

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy

# Run the classifier
python text_classifier.py
```

### In Google Colab
```python
!pip install scikit-learn pandas numpy
exec(open('text_classifier.py').read())
```

---

## Files in This Folder

| File | Description |
|---|---|
| `coretech_client_inquiries.csv` | 80-message labeled dataset across 8 categories |
| `text_classifier.py` | Full classification pipeline with both models |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Load CSV, explore dataset, value_counts |
| `numpy` | Array ops, argmax, mean, std, prediction stats |
| `scikit-learn` | TF-IDF, Naive Bayes, Logistic Regression, metrics |

---

## Screenshots

[Add Colab output screenshots here]

---

*Muhammad Taha — AI Engineering Intern, CoreTech Innovations*  
