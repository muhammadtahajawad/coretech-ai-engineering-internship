# Task 10: AI Automation Workflow for Client Inquiries

**Intern:** Muhammad Taha Jawad  
**Company:** CoreTech Innovations  
**Internship Track:** AI Engineering   
**Tools:** Python 3, pandas, numpy, re (regex), Google Colab  

---

## What This System Does

This AI automation workflow processes raw client inquiry messages end-to-end:
reads the message, extracts structured fields, identifies the required service,
assigns a priority, generates a professional reply, and saves everything to CSV.

---

## Automation Pipeline
Raw Inquiry Message (CSV)  
│  
▼  
Extract: Name, Email, Budget, Timeline, Urgency  
(using regex patterns)  
│  
▼  
Identify Required Service  
(keyword matching against CoreTech service list)  
│  
▼  
Assign Priority (High / Medium / Low)  
(based on urgency + budget rules)  
│  
▼  
Generate Professional Reply  
(using adaptive email template)  
│  
▼  
Save All Results to CSV  
(coretech_inquiries_processed.csv)


---



## Extracted Fields

| Field | Method |
|---|---|
| Client Name | Regex — matches name introduction phrases |
| Client Email | Regex — standard email pattern |
| Budget | Regex — detects $, USD, dollar formats |
| Timeline | Regex — detects week-based timeline phrases |
| Urgency | Keyword matching — urgent/flexible/medium |
| Service | Keyword matching — maps to CoreTech services |

---

## Priority Rules

| Condition | Priority |
|---|---|
| Urgency = High OR Budget ≥ $50,000 | High |
| Urgency = Medium OR Budget ≥ $10,000 | Medium |
| Urgency = Low AND Budget < $10,000 | Low |

---

## Files in This Folder

| File | Description |
|---|---|
| `coretech_inquiries_input.csv` | 15 raw client inquiry messages |
| `ai_automation.py` | Full automation pipeline |
| `coretech_inquiries_processed.csv` | Auto-generated output with all fields |

---

## How to Run

```bash
pip install pandas numpy
python ai_automation.py
```

### In Google Colab
```python
exec(open('ai_automation.py').read())
```

Generates `coretech_inquiries_processed.csv` automatically.

---

## Output Columns in Processed CSV

| Column | Description |
|---|---|
| client_name | Extracted client name |
| client_email | Extracted email address |
| identified_service | Matched CoreTech service |
| budget_usd | Extracted budget in USD |
| timeline_weeks | Extracted timeline in weeks |
| urgency | Extracted urgency level |
| priority | Assigned priority label |
| generated_reply | Full professional email reply |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Load CSV, build results DataFrame, save output |
| `numpy` | argmax for best service match, statistics, extraction rates |
| `re` | Regex extraction of name, email, budget, timeline |

---

## Screenshots
**Processing Log showing all 15 inquiries**
<img width="725" height="319" alt="image" src="https://github.com/user-attachments/assets/6a5e2c67-76c7-4302-b951-7d587ae17ef7" />

*The summary stats (priority distribution, service counts, budget stats, extraction rates)*
<img width="683" height="468" alt="image" src="https://github.com/user-attachments/assets/7384ff16-dbe6-4bd2-86aa-b2589141f49e" />

**Test Case #1 (Ahmed Raza: Healthcare ERP)**
<img width="719" height="456" alt="image" src="https://github.com/user-attachments/assets/8f2e587f-5a2f-46a9-bc4f-dda1c1358585" />
<img width="814" height="398" alt="image" src="https://github.com/user-attachments/assets/97ef9080-abf7-4ddc-b5d3-d45e9acc66c5" />

**Test Case #4 (Hina Malik: Education Low priority)**
<img width="736" height="480" alt="image" src="https://github.com/user-attachments/assets/c6824921-d78d-4163-99bd-4ae566524857" />
<img width="733" height="438" alt="image" src="https://github.com/user-attachments/assets/b21b2835-c038-476b-83be-8849a582ff2d" />

**Test Case #12 (Saad Yousaf: Finance High budget)**
<img width="737" height="473" alt="image" src="https://github.com/user-attachments/assets/af380ad0-c992-4c00-93ea-eb5f85e9882b" />
<img width="751" height="459" alt="image" src="https://github.com/user-attachments/assets/3ffd15e6-b2b9-4d7f-a79e-eade2eba018c" />


---

*Muhammad Taha Jawad: AI Engineering Intern, CoreTech Innovations*
