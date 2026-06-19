"""
CoreTech Innovations — AI Lead Scoring and Service Recommendation System
Task 09: AI Lead Scoring and Service Recommendation System
Intern: Muhammad Taha | AI Engineering Internship
Company: CoreTech Innovations (coretechio.com)

Description:
    This script scores incoming sales leads on a 0-100 scale based on
    5 weighted factors: budget, timeline, lead source, company size,
    and urgency. It outputs a priority label, a service recommendation,
    and a short explanation for each lead.

Libraries:
    - pandas : Load and manage the leads dataset
    - numpy  : Score calculations, normalization, statistics
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

# ─── CONFIGURATION — SCORING WEIGHTS ──────────────────────────────────────────

# Each factor contributes a weighted portion of the total 100-point score
WEIGHTS = {
    "budget"       : 30,   # Budget size — higher budget = higher score
    "timeline"     : 20,   # Shorter timeline = more urgent = higher score
    "lead_source"  : 15,   # Referrals score higher than cold calls
    "company_size" : 20,   # Larger companies score higher
    "urgency"      : 15    # Explicit urgency level stated by lead
}

# Lead source quality ranking (referrals are warmest leads)
SOURCE_SCORES = {
    "Referral"     : 1.0,
    "LinkedIn"     : 0.8,
    "Website"      : 0.6,
    "Cold Call"    : 0.4,
    "Social Media" : 0.3
}

# Company size ranking (larger companies = bigger potential contracts)
SIZE_SCORES = {
    "Enterprise" : 1.0,
    "Mid-size"   : 0.7,
    "Small"      : 0.4,
    "Startup"    : 0.25
}

# Urgency level ranking
URGENCY_SCORES = {
    "High"   : 1.0,
    "Medium" : 0.6,
    "Low"    : 0.3
}

# Service recommendation rules based on budget tier and industry
SERVICE_MAP = {
    "Healthcare"            : "ERP Systems",
    "Finance"                : "Cybersecurity",
    "Manufacturing"          : "ERP Systems",
    "Retail"                 : "Web Development",
    "Technology"              : "Software Solutions",
    "Education"               : "Mobile Apps",
    "Logistics"                : "ERP Systems",
    "Construction"             : "Web Development",
    "Real Estate"               : "Web Development",
    "Hospitality"                : "Mobile Apps",
    "Agriculture"                : "Software Solutions",
    "Professional Services"       : "Web Development"
}


# ─── STEP 1: LOAD DATASET ─────────────────────────────────────────────────────

def load_leads(filepath: str) -> pd.DataFrame:
    """
    Load the leads dataset from CSV into a pandas DataFrame.

    Args:
        filepath (str): Path to coretech_leads.csv

    Returns:
        pd.DataFrame: Loaded leads dataset
    """
    df = pd.read_csv(filepath)
    print(f"=== Leads Dataset Loaded ===")
    print(f"  Total Leads : {len(df)}")
    print(f"  Columns     : {list(df.columns)}")
    return df


# ─── STEP 2: SCORE INDIVIDUAL FACTORS ────────────────────────────────────────

def score_budget(budget: float, max_budget: float) -> float:
    """
    Score the budget factor on a 0-1 scale using min-max normalization.
    Higher budgets score closer to 1.0.

    Args:
        budget     (float): Lead's budget in USD
        max_budget (float): Maximum budget across all leads (for normalization)

    Returns:
        float: Normalized score between 0 and 1
    """
    # Normalize budget relative to the highest budget in the dataset
    return min(budget / max_budget, 1.0)


def score_timeline(timeline_weeks: float, max_weeks: float) -> float:
    """
    Score the timeline factor — SHORTER timelines score HIGHER
    because they indicate more immediate buying intent.

    Args:
        timeline_weeks (float): Lead's expected timeline in weeks
        max_weeks       (float): Maximum timeline across dataset

    Returns:
        float: Normalized score between 0 and 1 (inverted)
    """
    # Invert the score: shorter timeline = higher urgency = higher score
    normalized = timeline_weeks / max_weeks
    return 1.0 - normalized


def score_lead_source(source: str) -> float:
    """
    Score the lead source based on a predefined quality ranking.

    Args:
        source (str): Lead source channel

    Returns:
        float: Score between 0 and 1
    """
    return SOURCE_SCORES.get(source, 0.5)


def score_company_size(size: str) -> float:
    """
    Score the company size based on a predefined ranking.

    Args:
        size (str): Company size category

    Returns:
        float: Score between 0 and 1
    """
    return SIZE_SCORES.get(size, 0.5)


def score_urgency(urgency: str) -> float:
    """
    Score the stated urgency level.

    Args:
        urgency (str): Urgency level (High/Medium/Low)

    Returns:
        float: Score between 0 and 1
    """
    return URGENCY_SCORES.get(urgency, 0.5)


# ─── STEP 3: CALCULATE TOTAL LEAD SCORE ───────────────────────────────────────

def calculate_lead_score(row: pd.Series, max_budget: float, max_weeks: float) -> float:
    """
    Calculate the total weighted lead score (0-100) for a single lead.

    Formula:
        total_score = sum(factor_score * factor_weight) for all 5 factors

    Args:
        row        (pd.Series): A single lead record
        max_budget (float)    : Max budget in dataset (for normalization)
        max_weeks  (float)    : Max timeline in dataset (for normalization)

    Returns:
        float: Total lead score from 0 to 100
    """
    # Calculate normalized score (0-1) for each factor
    budget_score   = score_budget(row["budget_usd"], max_budget)
    timeline_score = score_timeline(row["timeline_weeks"], max_weeks)
    source_score   = score_lead_source(row["lead_source"])
    size_score     = score_company_size(row["company_size"])
    urgency_score  = score_urgency(row["urgency"])

    # Apply weights to each factor and sum for total score (0-100)
    total_score = (
        budget_score   * WEIGHTS["budget"] +
        timeline_score * WEIGHTS["timeline"] +
        source_score   * WEIGHTS["lead_source"] +
        size_score     * WEIGHTS["company_size"] +
        urgency_score  * WEIGHTS["urgency"]
    )

    return round(total_score, 2)


# ─── STEP 4: ASSIGN PRIORITY LABEL ────────────────────────────────────────────

def assign_priority(score: float) -> str:
    """
    Assign a priority label based on the total lead score.

    Thresholds:
        High   : score >= 70
        Medium : 40 <= score < 70
        Low    : score < 40

    Args:
        score (float): Total lead score (0-100)

    Returns:
        str: Priority label
    """
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


# ─── STEP 5: RECOMMEND SERVICE ────────────────────────────────────────────────

def recommend_service(industry: str, budget: float) -> str:
    """
    Recommend a CoreTech service based on industry and budget tier.

    Args:
        industry (str)  : Lead's industry
        budget   (float): Lead's budget in USD

    Returns:
        str: Recommended CoreTech service
    """
    # Industry-based recommendation as the primary signal
    base_recommendation = SERVICE_MAP.get(industry, "Software Solutions")

    # Override for very high budgets — recommend full ERP regardless of industry
    if budget >= 80000:
        return "ERP Systems"

    # Override for very low budgets — recommend UI/UX as an entry point
    if budget < 5000:
        return "UI/UX Design"

    return base_recommendation


# ─── STEP 6: GENERATE EXPLANATION ─────────────────────────────────────────────

def generate_explanation(row: pd.Series, score: float, priority: str) -> str:
    """
    Generate a short, human-readable explanation for the lead score.

    Args:
        row      (pd.Series): Lead record
        score    (float)    : Calculated lead score
        priority (str)      : Assigned priority label

    Returns:
        str: Short explanation sentence
    """
    reasons = []

    # Budget reasoning
    if row["budget_usd"] >= 50000:
        reasons.append("strong budget")
    elif row["budget_usd"] < 5000:
        reasons.append("limited budget")

    # Timeline reasoning
    if row["timeline_weeks"] <= 8:
        reasons.append("urgent timeline")
    elif row["timeline_weeks"] >= 20:
        reasons.append("long timeline")

    # Source reasoning
    if row["lead_source"] == "Referral":
        reasons.append("came via referral")
    elif row["lead_source"] == "Social Media":
        reasons.append("low-intent source")

    # Company size reasoning
    if row["company_size"] == "Enterprise":
        reasons.append("enterprise-scale company")
    elif row["company_size"] == "Startup":
        reasons.append("early-stage startup")

    # Urgency reasoning
    if row["urgency"] == "High":
        reasons.append("explicitly marked high urgency")

    reason_text = ", ".join(reasons) if reasons else "average lead signals across all factors"

    return f"{priority} priority ({score}/100) due to {reason_text}."


# ─── STEP 7: FULL SCORING PIPELINE ────────────────────────────────────────────

def score_all_leads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full lead scoring pipeline on the entire dataset.

    Args:
        df (pd.DataFrame): Raw leads dataset

    Returns:
        pd.DataFrame: Leads with score, priority, recommendation, explanation
    """
    # Get max values for normalization using numpy
    max_budget = np.max(df["budget_usd"].values)
    max_weeks  = np.max(df["timeline_weeks"].values)

    print(f"\n=== Scoring Parameters ===")
    print(f"  Max Budget   : ${max_budget:,.0f}")
    print(f"  Max Timeline : {max_weeks} weeks")
    print(f"  Weights      : {WEIGHTS}")

    # Calculate score for each lead row by row
    scores       = []
    priorities   = []
    recommendations = []
    explanations = []

    for _, row in df.iterrows():
        score    = calculate_lead_score(row, max_budget, max_weeks)
        priority = assign_priority(score)
        service  = recommend_service(row["industry"], row["budget_usd"])
        explanation = generate_explanation(row, score, priority)

        scores.append(score)
        priorities.append(priority)
        recommendations.append(service)
        explanations.append(explanation)

    # Add results as new columns using pandas
    result_df = df.copy()
    result_df["lead_score"]       = scores
    result_df["priority"]         = priorities
    result_df["recommended_service"] = recommendations
    result_df["explanation"]      = explanations

    return result_df


# ─── STEP 8: DISPLAY RESULTS ──────────────────────────────────────────────────

def display_summary(result_df: pd.DataFrame) -> None:
    """
    Display summary statistics of the scored leads using pandas and numpy.

    Args:
        result_df (pd.DataFrame): Scored leads dataset
    """
    print(f"\n=== Scoring Summary ===")

    # Priority distribution using pandas value_counts
    priority_counts = result_df["priority"].value_counts()
    print(f"\n  Priority Distribution:")
    for p in ["High", "Medium", "Low"]:
        count = priority_counts.get(p, 0)
        bar   = "█" * count
        print(f"  {p:<8}: {count:>3}  {bar}")

    # Score statistics using numpy
    scores_array = result_df["lead_score"].values
    print(f"\n  Score Statistics:")
    print(f"  Average Score : {np.mean(scores_array):.2f}")
    print(f"  Max Score     : {np.max(scores_array):.2f}")
    print(f"  Min Score     : {np.min(scores_array):.2f}")
    print(f"  Std Dev       : {np.std(scores_array):.2f}")

    # Service recommendation distribution
    print(f"\n  Service Recommendation Distribution:")
    service_counts = result_df["recommended_service"].value_counts()
    for service, count in service_counts.items():
        print(f"  {service:<25}: {count}")


def display_top_leads(result_df: pd.DataFrame, n: int = 10) -> None:
    """
    Display the top N highest-scoring leads with full details.

    Args:
        result_df (pd.DataFrame): Scored leads dataset
        n         (int)         : Number of top leads to show
    """
    print(f"\n=== Top {n} Highest Scoring Leads ===\n")

    # Sort by lead_score descending using pandas
    top_leads = result_df.sort_values("lead_score", ascending=False).head(n)

    for _, row in top_leads.iterrows():
        print(f"  Company    : {row['company_name']}")
        print(f"  Industry   : {row['industry']}")
        print(f"  Score      : {row['lead_score']}/100")
        print(f"  Priority   : {row['priority']}")
        print(f"  Service    : {row['recommended_service']}")
        print(f"  Reason     : {row['explanation']}")
        print(f"  {'-' * 55}")


def display_single_lead(result_df: pd.DataFrame, lead_id: int) -> None:
    """
    Display full scoring details for one specific lead by ID.
    Used as a test case to demonstrate individual lead output.

    Args:
        result_df (pd.DataFrame): Scored leads dataset
        lead_id   (int)         : ID of the lead to display
    """
    lead = result_df[result_df["id"] == lead_id]
    if lead.empty:
        print(f"  Lead ID {lead_id} not found.")
        return

    row = lead.iloc[0]
    print(f"\n  ── Lead #{lead_id}: {row['company_name']} ──")
    print(f"  Industry        : {row['industry']}")
    print(f"  Budget          : ${row['budget_usd']:,.0f}")
    print(f"  Timeline        : {row['timeline_weeks']} weeks")
    print(f"  Lead Source     : {row['lead_source']}")
    print(f"  Company Size    : {row['company_size']}")
    print(f"  Urgency         : {row['urgency']}")
    print(f"  ── Output ──")
    print(f"  Lead Score      : {row['lead_score']}/100")
    print(f"  Priority        : {row['priority']}")
    print(f"  Recommendation  : {row['recommended_service']}")
    print(f"  Explanation     : {row['explanation']}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CoreTech Innovations — AI Lead Scoring System")
    print("  coretechio.com | Nawabshah & Hyderabad, Pakistan")
    print("=" * 60)

    # Step 1: Load leads dataset
    df = load_leads("coretech_leads.csv")

    # Step 2: Score all leads
    result_df = score_all_leads(df)

    # Step 3: Display summary statistics
    display_summary(result_df)

    # Step 4: Display top 10 highest priority leads
    display_top_leads(result_df, n=10)

    # Step 5: Test cases — show 3 individual lead breakdowns
    print(f"\n=== Test Cases — Individual Lead Breakdown ===")
    display_single_lead(result_df, lead_id=14)   # High budget enterprise
    display_single_lead(result_df, lead_id=13)   # Low budget startup
    display_single_lead(result_df, lead_id=1)    # Healthcare referral

    # Step 6: Save scored results to a new CSV
    result_df.to_csv("coretech_leads_scored.csv", index=False)
    print(f"\n  Scored results saved to 'coretech_leads_scored.csv'")
    print("=" * 60)


if __name__ == "__main__":
    main()
