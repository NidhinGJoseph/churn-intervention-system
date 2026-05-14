# =============================================================================
# intervention.py
# -----------------------------------------------------------------------------
# PURPOSE:
#   Transforms the raw optimizer output into a clean, actionable
#   business report that non-technical users can act on directly.
#
# WHAT THIS FILE DOES:
#   1. Adds priority tier         — High / Medium / Low risk segments
#   2. Adds offer type            — what kind of offer to send
#   3. Adds recommended action    — plain English for account managers
#   4. Adds risk segment label    — for dashboard filtering
#   5. Formats and cleans output  — round numbers, rename columns
#
# WHAT THIS FILE DOES NOT DO:
#   - No model predictions        (that's predict.py)
#   - No budget optimization      (that's optimizer.py)
#   - No evaluation metrics       (that's evaluation.py)
#
# USED BY:
#   - predict.py → called after optimize_intervention()
#                  to format the final output before saving
# =============================================================================

import pandas as pd
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# BUSINESS RULES
# =============================================================================
# These thresholds define how we classify customers into priority tiers
# and what offer they receive.
# Backed by EDA churn rate findings.

# Priority tier thresholds based on churn probability
HIGH_RISK_THRESHOLD   = 0.70   # churn_prob >= 0.70 → High priority
MEDIUM_RISK_THRESHOLD = 0.50   # churn_prob >= 0.50 → Medium priority
                                # churn_prob < 0.50  → Low priority

# Offer type rules based on customer value and contract type
# High LTV + Month-to-month → most valuable to retain → best offer
HIGH_LTV_THRESHOLD = 3500      # LTV above this = high value customer


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def build_intervention_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the optimizer output into a clean business intervention list.

    HOW TO USE FROM predict.py:
        from src.intervention import build_intervention_list

        optimizer_output = optimize_intervention(df_clean, churn_prob)
        final_report     = build_intervention_list(optimizer_output)
        final_report.to_csv(INTERVENTION_PATH, index=False)

    Args:
        df (pd.DataFrame): Raw output from optimizer.py
                           Must contain: churn_prob, LTV, MonthlyCharges,
                           tenure, intervention_cost, expected_profit,
                           Contract, InternetService

    Returns:
        pd.DataFrame: Formatted intervention list ready for business users
    """

    logger.info("Building intervention list...")

    if df.empty:
        logger.warning("Empty DataFrame received — no interventions to format")
        return df

    df = df.copy()

    # -------------------------------------------------------------------------
    # STEP 1: ASSIGN PRIORITY TIER
    # -------------------------------------------------------------------------
    df["priority_tier"] = _assign_priority_tier(df["churn_prob"])

    # -------------------------------------------------------------------------
    # STEP 2: ASSIGN OFFER TYPE
    # -------------------------------------------------------------------------
    df["offer_type"] = _assign_offer_type(df)

    # -------------------------------------------------------------------------
    # STEP 3: ASSIGN RECOMMENDED ACTION
    # -------------------------------------------------------------------------
    df["recommended_action"] = _assign_recommended_action(df)

    # -------------------------------------------------------------------------
    # STEP 4: ASSIGN RISK SEGMENT
    # -------------------------------------------------------------------------
    df["risk_segment"] = _assign_risk_segment(df)

    # -------------------------------------------------------------------------
    # STEP 5: ADD READABLE BUSINESS METRICS
    # -------------------------------------------------------------------------

    # Expected monthly saving = expected revenue saved per month
    # We divide LTV-based profit by remaining tenure to get monthly view
    # This makes the number more intuitive for business users
    df["expected_monthly_saving"] = np.where(
        df["tenure"] < 72,
        df["expected_profit"] / np.maximum(1, 72 - df["tenure"]),
        df["expected_profit"]
    ).round(2)

    # Churn probability as a readable percentage
    df["churn_probability_pct"] = (df["churn_prob"] * 100).round(1)

    # -------------------------------------------------------------------------
    # STEP 6: SORT BY PRIORITY
    # -------------------------------------------------------------------------
    # Sort order: High first, then Medium, then Low
    # Within each tier, sort by expected profit descending
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    df["priority_sort"] = df["priority_tier"].map(priority_order)

    df = df.sort_values(
        by=["priority_sort", "expected_profit"],
        ascending=[True, False]
    ).drop(columns=["priority_sort"])

    # -------------------------------------------------------------------------
    # STEP 7: ROUND ALL NUMERIC COLUMNS
    # -------------------------------------------------------------------------
    numeric_cols = [
        "churn_prob", "LTV", "intervention_cost",
        "expected_revenue_saved", "expected_profit",
        "risk_adjusted_profit", "MonthlyCharges"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    # -------------------------------------------------------------------------
    # STEP 8: SELECT AND ORDER FINAL COLUMNS
    # -------------------------------------------------------------------------
    # Business-facing columns first, technical columns last
    # Account managers see the action columns first
    business_cols = [
        "priority_tier",            # High / Medium / Low
        "risk_segment",             # Descriptive segment label
        "offer_type",               # What offer to send
        "recommended_action",       # Plain English instruction
        "churn_probability_pct",    # e.g. 78.3%
        "MonthlyCharges",           # Customer's current monthly bill
        "tenure",                   # How long they've been a customer
        "intervention_cost",        # Cost of the offer
        "expected_revenue_saved",   # Total revenue saved if retention succeeds
        "expected_monthly_saving",  # Expected monthly revenue saved
        "expected_profit",          # Total expected profit from retention
        "LTV",                      # Customer lifetime value
        "Contract",                 # Contract type
        "InternetService",          # Internet service type
    ]

    # Only keep columns that exist in the DataFrame
    final_cols = [col for col in business_cols if col in df.columns]
    df = df[final_cols]

    # -------------------------------------------------------------------------
    # LOG SUMMARY
    # -------------------------------------------------------------------------
    logger.info(f"Intervention list built: {len(df)} customers")
    logger.info(f"\nPriority breakdown:")

    for tier in ["High", "Medium", "Low"]:
        count = (df["priority_tier"] == tier).sum()
        if count > 0:
            logger.info(f"  {tier} priority: {count} customers")

    logger.info(f"\nOffer type breakdown:")
    for offer in df["offer_type"].unique():
        count = (df["offer_type"] == offer).sum()
        logger.info(f"  {offer}: {count} customers")

    return df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _assign_priority_tier(churn_prob: pd.Series) -> pd.Series:
    """
    Assigns priority tier based on churn probability.

    WHY THESE THRESHOLDS?
        High   (≥70%): Extremely likely to churn — act immediately
        Medium (≥50%): Likely to churn — act within the week
        Low    (<50%): At risk but less urgent — standard outreach

    Account managers use this to triage their workload.
    High priority customers get personal calls.
    Low priority customers get automated emails.
    """

    return pd.cut(
        churn_prob,
        bins=[0, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    ).astype(str)


def _assign_offer_type(df: pd.DataFrame) -> pd.Series:
    """
    Assigns the type of retention offer based on customer profile.

    OFFER LOGIC (in priority order):
        1. High LTV + Month-to-month → "Premium Discount"
           Most valuable customer, most at risk of leaving — best offer
           20% discount + loyalty reward

        2. Month-to-month + High churn → "Contract Upgrade Incentive"
           Encourage them to commit to a 1-year contract
           Reduces future churn risk significantly

        3. Fiber optic customer → "Service Enhancement Offer"
           Fiber customers churn at 41.9% — often due to service issues
           Offer free tech support or speed upgrade

        4. Everyone else → "Loyalty Discount"
           Standard 20% discount offer
    """

    conditions = [
        # Condition 1: High value + flexible contract
        (df["LTV"] >= HIGH_LTV_THRESHOLD) &
        (df["Contract"] == "Month-to-month"),

        # Condition 2: Month-to-month + high churn risk
        (df["Contract"] == "Month-to-month") &
        (df["churn_prob"] >= HIGH_RISK_THRESHOLD),

        # Condition 3: Fiber optic customer
        (df["InternetService"] == "Fiber optic"),
    ]

    choices = [
        "Premium Discount",           # Best offer for high-value customers
        "Contract Upgrade Incentive", # Encourage commitment
        "Service Enhancement Offer",  # Address fiber pain points
    ]

    # np.select applies conditions in order — first match wins
    return pd.Series(
        np.select(conditions, choices, default="Loyalty Discount"),
        index=df.index
    )


def _assign_recommended_action(df: pd.DataFrame) -> pd.Series:
    """
    Generates a plain English action instruction per customer.

    WHY THIS EXISTS:
        Data scientists understand churn_prob=0.82 and LTV=3200.
        Account managers need to know: "Call this customer and offer X."

        This bridges the gap between model output and business action.
        Each instruction is specific enough to act on immediately.
    """

    actions = []

    for _, row in df.iterrows():

        tier       = row.get("priority_tier", "Low")
        offer      = row.get("offer_type", "Loyalty Discount")
        prob_pct   = row.get("churn_prob", 0) * 100
        monthly    = row.get("MonthlyCharges", 0)
        discount   = monthly * 0.20

        if tier == "High":
            action = (
                f"URGENT: Call within 24hrs. "
                f"Churn risk: {prob_pct:.0f}%. "
                f"Offer: {offer} (₹{discount:.0f}/month saving)."
            )
        elif tier == "Medium":
            action = (
                f"Contact within 3 days. "
                f"Churn risk: {prob_pct:.0f}%. "
                f"Offer: {offer} (₹{discount:.0f}/month saving)."
            )
        else:
            action = (
                f"Standard outreach this week. "
                f"Churn risk: {prob_pct:.0f}%. "
                f"Offer: {offer} (₹{discount:.0f}/month saving)."
            )

        actions.append(action)

    return pd.Series(actions, index=df.index)


def _assign_risk_segment(df: pd.DataFrame) -> pd.Series:
    """
    Assigns a descriptive risk segment label combining multiple signals.

    WHY THIS EXISTS:
        Priority tier tells you urgency (High/Medium/Low).
        Risk segment tells you WHY the customer is at risk.
        This helps account managers personalize their outreach.

    SEGMENTS:
        High Value At Risk    — high LTV + high churn prob
        New Customer At Risk  — low tenure + high churn prob
        Price Sensitive       — high monthly charges + high churn prob
        Service Dissatisfied  — fiber optic + high churn prob
        Standard At Risk      — everyone else who passed filters
    """

    conditions = [
        # High value customer who might leave
        (df["LTV"] >= HIGH_LTV_THRESHOLD) &
        (df["churn_prob"] >= MEDIUM_RISK_THRESHOLD),

        # New customer still evaluating
        (df["tenure"] <= 12) &
        (df["churn_prob"] >= MEDIUM_RISK_THRESHOLD),

        # Likely price sensitive
        (df["MonthlyCharges"] >= 80) &
        (df["churn_prob"] >= MEDIUM_RISK_THRESHOLD),

        # Likely dissatisfied with service quality
        (df["InternetService"] == "Fiber optic") &
        (df["churn_prob"] >= MEDIUM_RISK_THRESHOLD),
    ]

    segments = [
        "High Value At Risk",
        "New Customer At Risk",
        "Price Sensitive",
        "Service Dissatisfied",
    ]

    return pd.Series(
        np.select(conditions, segments, default="Standard At Risk"),
        index=df.index
    )