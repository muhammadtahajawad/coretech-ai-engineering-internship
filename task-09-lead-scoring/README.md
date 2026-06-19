# Task 09: AI Lead Scoring and Service Recommendation System

**Intern:** Muhammad Taha Jawad 
**Company:** CoreTech Innovations  
**Internship Track:** AI Engineering  
**Tools:** Python 3, pandas, numpy, Google Colab  

---

## What This System Does

This system scores incoming sales leads on a 0-100 scale using 5 weighted
factors, then outputs a priority label, a recommended CoreTech service,
and a short explanation — helping the sales team prioritize outreach.

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


---

**Muhammad Taha Jawad**
