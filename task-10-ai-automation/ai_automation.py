"""
CoreTech Innovations — AI Automation Workflow for Client Inquiries
Task 10: AI Automation Workflow for Client Inquiries
Intern: Muhammad Taha | AI Engineering Internship
Company: CoreTech Innovations (coretechio.com)

Description:
    This script builds a complete AI automation workflow that:
    1. Reads raw client inquiry messages from a CSV file
    2. Extracts structured fields: name, email, service, budget,
       timeline, urgency using regex and keyword matching
    3. Identifies the required CoreTech service
    4. Assigns a priority label (High / Medium / Low)
    5. Generates a professional reply email using a template
    6. Saves all processed results to a new CSV file

Libraries:
    - pandas : Load inquiries, build results DataFrame, save CSV
    - numpy  : Score calculations, priority thresholds, statistics
    - re     : Regex-based field extraction from raw message text
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import re
import numpy as np
import pandas as pd
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Input and output file paths
INPUT_FILE  = "coretech_inquiries_input.csv"
OUTPUT_FILE = "coretech_inquiries_processed.csv"

# Company info for reply template
COMPANY_NAME    = "CoreTech Innovations"
COMPANY_EMAIL   = "hr@coretechio.com"
COMPANY_PHONE   = "+92 348 0394588"
COMPANY_WEBSITE = "coretechio.com"

# ─── SERVICE KEYWORD MAP ──────────────────────────────────────────────────────

# Maps service keywords found in messages to CoreTech service names
SERVICE_KEYWORDS = {
    "Web Development"  : [
        "website", "web app", "web application", "landing page",
        "web portal", "web platform", "e-commerce website", "online store"
    ],
    "Mobile Apps"      : [
        "mobile app", "android app", "ios app", "flutter app",
        "react native", "smartphone app", "cross-platform app",
        "mobile application", "app for"
    ],
    "ERP Systems"      : [
        "erp", "enterprise resource planning", "erp system",
        "inventory management", "operations system", "business system",
        "management system"
    ],
    "UI/UX Design"     : [
        "ui design", "ux design", "user interface", "user experience",
        "wireframe", "prototype", "design system", "redesign"
    ],
    "Digital Marketing": [
        "digital marketing", "social media", "facebook ads", "google ads",
        "email marketing", "content marketing", "brand awareness", "campaigns"
    ],
    "Cybersecurity"    : [
        "cybersecurity", "cyber security", "security audit", "data protection",
        "network security", "cloud security", "vulnerability", "compliance"
    ],
    "Software Solutions": [
        "software solution", "custom software", "crm", "saas",
        "automation software", "dashboard", "data analytics", "software system"
    ]
}

# ─── URGENCY KEYWORD MAP ──────────────────────────────────────────────────────

# Maps urgency keywords to urgency levels
URGENCY_KEYWORDS = {
    "High"  : [
        "urgent", "urgently", "high urgency", "very urgent",
        "immediately", "asap", "critical", "extremely urgent",
        "high priority", "very high urgency"
    ],
    "Medium": [
        "medium urgency", "medium priority", "moderate",
        "normal priority", "standard"
    ],
    "Low"   : [
        "low urgency", "low priority", "not urgent",
        "flexible", "not very urgent", "no rush"
    ]
}

# Priority thresholds based on urgency and budget
PRIORITY_RULES = {
    "High"  : {"urgency": "High",   "min_budget": 0},
    "Medium": {"urgency": "Medium", "min_budget": 0},
    "Low"   : {"urgency": "Low",    "min_budget": 0}
}


# ─── STEP 1: LOAD INQUIRIES ───────────────────────────────────────────────────

def load_inquiries(filepath: str) -> pd.DataFrame:
    """
    Load raw client inquiry messages from the input CSV file.

    Args:
        filepath (str): Path to the inquiries CSV file

    Returns:
        pd.DataFrame: Loaded inquiries with id and raw_message columns
    """
    df = pd.read_csv(filepath)
    print(f"=== Inquiries Loaded ===")
    print(f"  Total Messages : {len(df)}")
    print(f"  Columns        : {list(df.columns)}")
    return df


# ─── STEP 2: EXTRACT CLIENT NAME ──────────────────────────────────────────────

def extract_name(message: str) -> str:
    """
    Extract the client name from the inquiry message using regex patterns.
    Tries multiple common introductory phrases.

    Args:
        message (str): Raw inquiry message

    Returns:
        str: Extracted client name or 'Unknown' if not found
    """
    # Regex patterns for common name introduction phrases
    patterns = [
        r"my name is ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"I am ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"this is ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"([A-Z][a-z]+ [A-Z][a-z]+) here",
        r"^([A-Z][a-z]+ [A-Z][a-z]+),",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "Unknown"


# ─── STEP 3: EXTRACT CLIENT EMAIL ─────────────────────────────────────────────

def extract_email(message: str) -> str:
    """
    Extract the client email address from the message using regex.

    Args:
        message (str): Raw inquiry message

    Returns:
        str: Extracted email address or 'Not provided' if not found
    """
    # Standard email regex pattern
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, message)
    return match.group(0) if match else "Not provided"


# ─── STEP 4: EXTRACT BUDGET ───────────────────────────────────────────────────

def extract_budget(message: str) -> float:
    """
    Extract the budget amount from the message using regex.
    Handles formats like $45000, $45,000, USD 45000, 45000 dollars.

    Args:
        message (str): Raw inquiry message

    Returns:
        float: Extracted budget in USD or 0.0 if not found
    """
    patterns = [
        r"\$([0-9,]+)",                    # $45000 or $45,000
        r"budget\s+(?:is\s+)?(?:around\s+)?\$?([0-9,]+)",  # budget is $45000
        r"([0-9,]+)\s+(?:USD|dollars)",   # 45000 USD
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            # Remove commas and convert to float
            budget_str = match.group(1).replace(",", "")
            try:
                return float(budget_str)
            except ValueError:
                continue

    return 0.0


# ─── STEP 5: EXTRACT TIMELINE ─────────────────────────────────────────────────

def extract_timeline(message: str) -> int:
    """
    Extract the project timeline in weeks from the message.
    Handles formats like '8 weeks', 'within 8 weeks', '8-week timeline'.

    Args:
        message (str): Raw inquiry message

    Returns:
        int: Timeline in weeks or 0 if not found
    """
    patterns = [
        r"(\d+)\s*weeks?",                  # 8 weeks
        r"within\s+(\d+)\s*weeks?",         # within 8 weeks
        r"(\d+)\s*week\s+timeline",         # 8 week timeline
        r"timeline\s+(?:is\s+)?(?:about\s+)?(\d+)\s*weeks?",  # timeline is 8 weeks
        r"delivery\s+in\s+(\d+)\s*weeks?",  # delivery in 10 weeks
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return 0


# ─── STEP 6: EXTRACT URGENCY ──────────────────────────────────────────────────

def extract_urgency(message: str) -> str:
    """
    Extract the urgency level from the message using keyword matching.

    Args:
        message (str): Raw inquiry message

    Returns:
        str: Urgency level (High / Medium / Low)
    """
    message_lower = message.lower()

    for level, keywords in URGENCY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in message_lower:
                return level

    # Default to Medium if no urgency keyword found
    return "Medium"


# ─── STEP 7: IDENTIFY SERVICE ─────────────────────────────────────────────────

def identify_service(message: str) -> str:
    """
    Identify the required CoreTech service from the inquiry message
    using keyword matching against the SERVICE_KEYWORDS dictionary.

    Args:
        message (str): Raw inquiry message

    Returns:
        str: Identified CoreTech service name
    """
    message_lower = message.lower()

    # Track match count per service for best-match selection
    match_counts = {}

    for service, keywords in SERVICE_KEYWORDS.items():
        count = 0
        for keyword in keywords:
            if keyword in message_lower:
                count += 1
        if count > 0:
            match_counts[service] = count

    if not match_counts:
        return "General Inquiry"

    # Return the service with the most keyword matches using numpy argmax
    services  = list(match_counts.keys())
    counts    = np.array(list(match_counts.values()))
    best_idx  = np.argmax(counts)

    return services[best_idx]


# ─── STEP 8: ASSIGN PRIORITY ──────────────────────────────────────────────────

def assign_priority(urgency: str, budget: float) -> str:
    """
    Assign a priority label based on urgency level and budget.

    Rules:
        High   : Urgency is High OR budget >= $50,000
        Medium : Urgency is Medium OR budget >= $10,000
        Low    : Urgency is Low AND budget < $10,000

    Args:
        urgency (str)  : Extracted urgency level
        budget  (float): Extracted budget in USD

    Returns:
        str: Priority label (High / Medium / Low)
    """
    if urgency == "High" or budget >= 50000:
        return "High"
    elif urgency == "Medium" or budget >= 10000:
        return "Medium"
    else:
        return "Low"


# ─── STEP 9: GENERATE REPLY ───────────────────────────────────────────────────

def generate_reply(name: str, service: str, budget: float,
                   timeline: int, priority: str) -> str:
    """
    Generate a professional reply email using a structured template.
    The template adapts based on service, priority, and budget.

    Args:
        name     (str)  : Client name
        service  (str)  : Identified service
        budget   (float): Client budget in USD
        timeline (int)  : Project timeline in weeks
        priority (str)  : Assigned priority label

    Returns:
        str: Complete professional reply email text
    """
    # Get current date for the email header
    today = datetime.now().strftime("%B %d, %Y")

    # Priority-specific opening line
    priority_lines = {
        "High"  : "We have flagged your inquiry as HIGH PRIORITY and a senior consultant will reach out within 24 hours.",
        "Medium": "We have reviewed your inquiry and a consultant will contact you within 2-3 business days.",
        "Low"   : "We have received your inquiry and will respond with a detailed proposal within 5 business days."
    }

    # Budget acknowledgment line
    if budget >= 50000:
        budget_line = f"Your indicated budget of ${budget:,.0f} aligns well with our enterprise delivery capabilities."
    elif budget >= 10000:
        budget_line = f"Your indicated budget of ${budget:,.0f} is a good fit for our mid-range service packages."
    else:
        budget_line = f"We will work with your budget of ${budget:,.0f} to find the most cost-effective solution."

    # Timeline acknowledgment
    if timeline > 0:
        timeline_line = f"We acknowledge your timeline of {timeline} weeks and will ensure our proposal respects this constraint."
    else:
        timeline_line = "Please share your preferred timeline so we can plan accordingly."

    # Build the complete email template
    reply = f"""Date: {today}

Dear {name},

Thank you for reaching out to {COMPANY_NAME}. We truly appreciate your interest in our services.

{priority_lines.get(priority, priority_lines['Medium'])}

Based on your inquiry, we understand that you are looking for our **{service}** service. {budget_line} {timeline_line}

At {COMPANY_NAME}, we pride ourselves on delivering:
- Architecture-first, security-embedded solutions
- Transparent milestone-based project execution
- Senior engineering attention on every engagement
- Long-term maintainability and client support

Our team will prepare a tailored proposal for your {service} project and present it during our consultation call.

Next Steps:
1. A {COMPANY_NAME} consultant will contact you shortly
2. We will schedule a 30-minute discovery call
3. A detailed proposal will be shared within 48 hours of the call

For any immediate queries, feel free to reach us:
- Email   : {COMPANY_EMAIL}
- Phone   : {COMPANY_PHONE}
- Website : {COMPANY_WEBSITE}

We look forward to building something great together.

Warm regards,
{COMPANY_NAME} — Business Development Team
{COMPANY_WEBSITE} | {COMPANY_EMAIL}"""

    return reply


# ─── STEP 10: PROCESS ALL INQUIRIES ──────────────────────────────────────────

def process_inquiries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full automation workflow on all inquiry messages.
    Extracts all fields, assigns priority, and generates replies.

    Args:
        df (pd.DataFrame): Raw inquiries DataFrame

    Returns:
        pd.DataFrame: Fully processed inquiries with all output fields
    """
    print(f"\n=== Processing {len(df)} Inquiries ===\n")

    results = []

    for _, row in df.iterrows():
        msg = str(row["raw_message"])

        # Extract all fields from the raw message
        name     = extract_name(msg)
        email    = extract_email(msg)
        budget   = extract_budget(msg)
        timeline = extract_timeline(msg)
        urgency  = extract_urgency(msg)
        service  = identify_service(msg)
        priority = assign_priority(urgency, budget)
        reply    = generate_reply(name, service, budget, timeline, priority)

        print(f"  [{row['id']:>2}] {name:<25} | Service: {service:<20} | Priority: {priority}")

        results.append({
            "id"               : row["id"],
            "client_name"      : name,
            "client_email"     : email,
            "identified_service": service,
            "budget_usd"       : budget,
            "timeline_weeks"   : timeline,
            "urgency"          : urgency,
            "priority"         : priority,
            "generated_reply"  : reply,
            "raw_message"      : msg
        })

    # Build results DataFrame using pandas
    result_df = pd.DataFrame(results)
    return result_df


# ─── STEP 11: DISPLAY SUMMARY ─────────────────────────────────────────────────

def display_summary(result_df: pd.DataFrame) -> None:
    """
    Display processing summary statistics using pandas and numpy.

    Args:
        result_df (pd.DataFrame): Processed inquiries DataFrame
    """
    print(f"\n=== Processing Summary ===")

    # Priority distribution using pandas value_counts
    print(f"\n  Priority Distribution:")
    priority_counts = result_df["priority"].value_counts()
    for p in ["High", "Medium", "Low"]:
        count = priority_counts.get(p, 0)
        print(f"  {p:<8}: {count}")

    # Service distribution
    print(f"\n  Identified Services:")
    service_counts = result_df["identified_service"].value_counts()
    for service, count in service_counts.items():
        print(f"  {service:<25}: {count}")

    # Budget statistics using numpy
    budgets = result_df["budget_usd"].values
    budgets_nonzero = budgets[budgets > 0]
    print(f"\n  Budget Statistics (from {len(budgets_nonzero)} inquiries with budget):")
    print(f"  Average : ${np.mean(budgets_nonzero):>10,.0f}")
    print(f"  Max     : ${np.max(budgets_nonzero):>10,.0f}")
    print(f"  Min     : ${np.min(budgets_nonzero):>10,.0f}")

    # Extraction success rate using numpy
    email_found    = np.sum(result_df["client_email"] != "Not provided")
    name_found     = np.sum(result_df["client_name"] != "Unknown")
    budget_found   = np.sum(result_df["budget_usd"] > 0)
    timeline_found = np.sum(result_df["timeline_weeks"] > 0)

    print(f"\n  Extraction Success Rates:")
    print(f"  Name     : {name_found}/{len(result_df)} ({name_found/len(result_df)*100:.0f}%)")
    print(f"  Email    : {email_found}/{len(result_df)} ({email_found/len(result_df)*100:.0f}%)")
    print(f"  Budget   : {budget_found}/{len(result_df)} ({budget_found/len(result_df)*100:.0f}%)")
    print(f"  Timeline : {timeline_found}/{len(result_df)} ({timeline_found/len(result_df)*100:.0f}%)")


def display_test_case(result_df: pd.DataFrame, inquiry_id: int) -> None:
    """
    Display full processing output for a single inquiry as a test case.

    Args:
        result_df  (pd.DataFrame): Processed inquiries DataFrame
        inquiry_id (int)         : ID of the inquiry to display
    """
    row = result_df[result_df["id"] == inquiry_id]
    if row.empty:
        print(f"Inquiry {inquiry_id} not found.")
        return

    r = row.iloc[0]
    print(f"\n{'='*60}")
    print(f"  TEST CASE — Inquiry #{inquiry_id}")
    print(f"{'='*60}")
    print(f"  Raw Message   : {r['raw_message'][:100]}...")
    print(f"  ── Extracted Fields ──")
    print(f"  Client Name   : {r['client_name']}")
    print(f"  Client Email  : {r['client_email']}")
    print(f"  Service       : {r['identified_service']}")
    print(f"  Budget        : ${r['budget_usd']:,.0f}")
    print(f"  Timeline      : {r['timeline_weeks']} weeks")
    print(f"  Urgency       : {r['urgency']}")
    print(f"  Priority      : {r['priority']}")
    print(f"  ── Generated Reply ──")
    print(r["generated_reply"])
    print(f"{'='*60}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CoreTech Innovations — AI Automation Workflow")
    print("  Client Inquiry Processing System")
    print("  coretechio.com | Nawabshah & Hyderabad, Pakistan")
    print("=" * 60)

    # Step 1: Load inquiries
    df = load_inquiries(INPUT_FILE)

    # Step 2: Process all inquiries through full automation pipeline
    result_df = process_inquiries(df)

    # Step 3: Display summary statistics
    display_summary(result_df)

    # Step 4: Show 3 test cases with full output
    print(f"\n=== Test Cases ===")
    display_test_case(result_df, inquiry_id=1)   # Healthcare ERP high priority
    display_test_case(result_df, inquiry_id=4)   # Education startup low priority
    display_test_case(result_df, inquiry_id=12)  # Finance enterprise high budget

    # Step 5: Save processed results to output CSV using pandas
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Results saved to '{OUTPUT_FILE}'")
    print(f"  Total records processed: {len(result_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
