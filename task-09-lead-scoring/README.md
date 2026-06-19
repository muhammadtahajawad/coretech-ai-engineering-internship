# Task 09: AI Lead Scoring and Service Recommendation System

**Intern:** Muhammad Taha Jawad  
**Company:** CoreTech Innovations  
**Internship Track:** AI Engineering  
**Tools:** Python 3, pandas, numpy, Google Colab  

---

## What This System Does

This system scores incoming sales leads on a 0-100 scale using 5 weighted
factors, then outputs a priority label, a recommended CoreTech service,
and a short explanation helping the sales team prioritize outreach.

---

---

## Scoring Factors Explained

| Factor | Weight | Logic |
|---|---|---|
| Budget | 30% | Higher budget = higher score (normalized against max in dataset) |
| Timeline | 20% | Shorter timeline = higher urgency = higher score (inverted) |
| Lead Source | 15% | Referral > LinkedIn > Website > Cold Call > Social Media |
| Company Size | 20% | Enterprise > Mid-size > Small > Startup |
| Urgency | 15% | High > Medium > Low (self-reported by lead) |

---

## Priority Thresholds

| Score Range | Priority |
|---|---|
| 70 - 100 | High |
| 40 - 69 | Medium |
| 0 - 39 | Low |

---

## Files in This Folder

| File | Description |
|---|---|
| `coretech_leads.csv` | 52 sample leads across 12 industries |
| `lead_scoring.py` | Full scoring pipeline with test cases |

---

## How to Run

```bash
pip install pandas numpy
python lead_scoring.py
```

### In Google Colab
```python
exec(open('lead_scoring.py').read())
```

This generates `coretech_leads_scored.csv` with all output columns added.

---

# Sample Output
Lead #14: Capital Investment Bank

Industry        : Finance
Budget          : $100,000
Timeline        : 12 weeks
Lead Source     : Referral
Company Size    : Enterprise
Urgency         : High


Output 
Lead Score      : 89.5/100
Priority        : High
Recommendation  : ERP Systems
Explanation     : High priority (89.5/100) due to strong budget, came via
referral, enterprise-scale company, explicitly marked high urgency.


---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Load CSV, row iteration, sorting, value_counts, save results |
| `numpy` | Max/mean/std calculations, score normalization |

---

## Screenshots

## Scoring Summary:
<img width="793" height="512" alt="Screenshot 2026-06-18 160628" src="https://github.com/user-attachments/assets/7a2a9a5c-9c5c-4294-9bb4-8aa34f00ab0e" /> 

<img width="854" height="399" alt="image" src="https://github.com/user-attachments/assets/bf526426-6e2e-47b8-89ec-9909354a7c3c" /> 

# Test Case:

<img width="774" height="448" alt="image" src="https://github.com/user-attachments/assets/58f202d0-d269-449c-8093-962ce3f73366" />



---

**Muhammad Taha Jawad**
