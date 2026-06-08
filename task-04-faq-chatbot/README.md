> ⚠️ **Note:** Due to standard GitHub rendering timeouts handling heavy JSON metadata structures, the `.ipynb` notebook file view may occasionally flash a warning message above. Please use the official structural **Open in Colab** badge below to view code execution matrices directly, or inspect the verification traces in this document.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muhammadtahajawad/coretech-ai-engineering-internship/blob/main/task-04-faq-chatbot/task4_chatbot.ipynb)

# Task 4: CoreTech FAQ Dataset and Rule-Based Chatbot

**Intern Name:** Muhammad Taha Jawad  
**Target Organization:** CoreTech Innovations 
**Tools Used:** Python 3, Pandas, Regular Expressions, Git, Google Colab  

---

## 📋 Operational Project Summary
This module deploys a functional rule-based chatbot architecture designed to parse user intent and automatically route clients to core corporate answers. The setup includes an programmatically-generated dataset mapping company services, pipelines, and constraints.

### Key Architectural Enhancements
* **Token Intersection Matching:** Instead of relying on rigid, breakable string matching, the engine normalizes user input and calculates a token intersection weight score against keyword matrices.
* **Granular Exceptions Handling:** Gracefully catches missing dependencies and routes unmapped client phrases to a specific corporate advisory fallback loop.

---

## 📊 Dataset Distribution Verification (`coretech_faq.csv`)
The dataset context contains exactly 25 comprehensive categorical records. 

### 📋 Dataset Preview (Sample Rows)
Below is a structured preview of how the question, answer, and optimization keywords are mapped inside the target CSV file:

| Question | Answer | Keywords |
| :--- | :--- | :--- |
| What services does CoreTech Innovations offer? | CoreTech Innovations delivers enterprise web development, mobile applications, ERP systems, UI/UX design, cybersecurity, and digital marketing services. | services list offer what |
| What is CoreTech's historical client retention rate? | CoreTech Innovations proudly maintains a verified 94% client retention rate driven by transparent delivery. | retention rate statistic milestone |
| How does CoreTech handle cybersecurity protocols? | We embed security early into development lifecycles, ensuring strict compliance, threat mitigation, and data protection. | cybersecurity security protocol threat breach hack |
| Does CoreTech offer structured corporate internships? | Yes, we host structured 3-month AI Engineering internship cohorts focusing on real project exposure. | internship interns batch cohort hire |

---

## 🛠️ Execution & Chatbot Interface Verification
Below is the verification trace showing the rule-based keyword routing engine handling inputs and user exits correctly:

![Chatbot Execution Screen](chatbot_demo.png)
