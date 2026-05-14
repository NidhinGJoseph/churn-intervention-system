# =============================================================================
# optimizer.py
# -----------------------------------------------------------------------------
# PURPOSE:
#   Budget-constrained intervention optimizer.
#   Decides WHICH customers to target to maximize expected saved revenue
#   under a fixed monthly budget.
#
# WHAT THIS FILE DOES:
#   1. Applies hard business filters
#   2. Calculates LTV per customer
#   3. Calculates retention uplift per customer
#   4. Calculates expected profit per customer
#   5. Applies risk adjustment
#   6. Greedily selects best customers within budget
#
# USED BY:
#   - predict.py → calls optimize_intervention() after probabilities generated
# =============================================================================

import pandas as pd
import numpy as np

from src.config import (
    OFFER_COST_PERCENT,        # 20% discount on monthly charges
    OFFER_ACCEPTANCE_RATE,     # 15% of targeted customers accept the offer
    BASE_RETENTION_UPLIFT,     # Base probability offer retains the customer
    MONTHLY_BUDGET,            # Total budget available per month
    MAX_TENURE,                # Maximum tenure for LTV calculation
    MONTHLY_DISCOUNT_RATE,     # Discount rate for future cashflows
    MIN_CHURN_PROB,            # Minimum churn probability to target
    MIN_MONTHLY_CHARGES,       # Minimum monthly charges to target
    MIN_TENURE_MONTHS,         # Minimum tenure to target
    EXCLUDE_INTERNET_SERVICE,  # Internet service segments to exclude
    EXCLUDE_CONTRACT_TYPE      # Contract types to exclude
)

from src.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# LTV CALCULATION
# =============================================================================

def calculate_ltv(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Discounted Lifetime Value (LTV) per customer.

    WHY LTV AND NOT JUST MONTHLY CHARGES?
        Monthly charges = what a customer pays NOW.
        LTV = what a customer is worth over their REMAINING lifetime.
        A customer paying ₹80/month with 50 months left is worth far more
        than one paying ₹80/month with 5 months left.

    FORMULA:
        LTV = sum of discounted future monthly cashflows
        Each future month discounted: value / (1 + rate)^t
        Reflects that money today is worth more than money tomorrow.

    Args:
        df: DataFrame with tenure and MonthlyCharges columns

    Returns:
        pd.Series: LTV value per customer
    """

    remaining_months = np.maximum(0, MAX_TENURE - df["tenure"])

    ltv_values = []

    for idx in range(len(df)):

        monthly_charge = df.iloc[idx]["MonthlyCharges"]
        remaining      = int(remaining_months.iloc[idx])

        # Discounted cashflow for each future month
        discounted_cashflows = [
            monthly_charge / ((1 + MONTHLY_DISCOUNT_RATE) ** t)
            for t in range(1, remaining + 1)
        ]

        ltv_values.append(np.sum(discounted_cashflows))

    return pd.Series(ltv_values, index=df.index)


# =============================================================================
# RETENTION UPLIFT CALCULATION
# =============================================================================

def calculate_uplift(df: pd.DataFrame) -> pd.Series:
    """
    Estimates retention effectiveness per customer.

    WHY DIFFERENT UPLIFT PER CUSTOMER?
        Month-to-month customers are more flexible — easier to retain
        because they haven't committed to a long contract.
        One-year contract customers are already somewhat committed.

    HOW IT WORKS:
        Base uplift × contract multiplier × churn probability scaling
        Clipped between 0.05 and 0.30 to avoid unrealistic values.

    Args:
        df: DataFrame with Contract and churn_prob columns

    Returns:
        pd.Series: Uplift probability per customer
    """

    # Month-to-month customers get 1.5x base uplift — more responsive to offers
    uplift = np.where(
        df["Contract"] == "Month-to-month",
        BASE_RETENTION_UPLIFT * 1.5,
        BASE_RETENTION_UPLIFT
    )

    # Scale by churn probability — higher risk customers benefit more
    uplift = uplift * (0.5 + df["churn_prob"])

    # Clip to realistic range: min 5%, max 30%
    uplift = np.clip(uplift, 0.05, 0.30)

    return uplift


# =============================================================================
# INTERVENTION COST CALCULATION
# =============================================================================

def calculate_intervention_cost(df: pd.DataFrame) -> pd.Series:
    """
    Cost of the retention offer per customer.

    FORMULA:
        Cost = MonthlyCharges × OFFER_COST_PERCENT
        Example: ₹80/month × 0.20 = ₹16 offer cost

    WHY DYNAMIC COST?
        Scales naturally with customer value.
        High-value customers get bigger offers, low-value get smaller ones.
    """

    return df["MonthlyCharges"] * OFFER_COST_PERCENT


# =============================================================================
# MAIN OPTIMIZER
# =============================================================================

def optimize_intervention(
    df: pd.DataFrame,
    churn_prob: np.ndarray
) -> pd.DataFrame:
    """
    Selects the optimal set of customers to target under a fixed budget.

    THE CORE BUSINESS PROBLEM:
        We can't afford to offer discounts to every at-risk customer.
        We pick the subset that maximizes expected saved revenue
        while staying within budget — a variant of the knapsack problem.

    APPROACH — Greedy by Expected Value:
        1. Filter out customers not worth targeting
        2. Calculate expected profit per customer
        3. Sort by risk-adjusted profit (best first)
        4. Greedily add customers until budget exhausted

    Args:
        df:         Clean DataFrame from data_preprocessing.py
        churn_prob: Churn probabilities from the model

    Returns:
        pd.DataFrame: Selected customers with all business metrics
    """

    logger.info("Starting optimization...")

    df = df.copy()

    # -------------------------------------------------------------------------
    # STEP 1: ATTACH CHURN PROBABILITIES
    # -------------------------------------------------------------------------
    df["churn_prob"] = churn_prob

    # -------------------------------------------------------------------------
    # STEP 2: APPLY HARD BUSINESS FILTERS
    # -------------------------------------------------------------------------
    # Every filter is backed by EDA findings — see docs/eda_findings.md
    df = df[
        (df["churn_prob"] >= MIN_CHURN_PROB) &
        (df["MonthlyCharges"] >= MIN_MONTHLY_CHARGES) &
        (df["tenure"] >= MIN_TENURE_MONTHS) &
        (~df["InternetService"].isin(EXCLUDE_INTERNET_SERVICE)) &
        (~df["Contract"].isin(EXCLUDE_CONTRACT_TYPE))
    ]

    logger.info(f"After business filters: {len(df)}")

    if len(df) == 0:
        logger.warning("No customers passed business filters")
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # STEP 3: CALCULATE LTV
    # -------------------------------------------------------------------------
    df["LTV"] = calculate_ltv(df)

    # -------------------------------------------------------------------------
    # STEP 4: CALCULATE INTERVENTION ECONOMICS
    # -------------------------------------------------------------------------
    df["uplift"]            = calculate_uplift(df)
    df["intervention_cost"] = calculate_intervention_cost(df)

    # -------------------------------------------------------------------------
    # STEP 5: CALCULATE EXPECTED VALUE
    # -------------------------------------------------------------------------
    # Formula:
    #   Expected Revenue Saved
    #     = P(churn) × P(offer accepted) × P(offer retains customer) × LTV
    #
    # OFFER_ACCEPTANCE_RATE: not every customer accepts the offer (15%)
    # uplift: probability the offer actually retains the customer
    # Without OFFER_ACCEPTANCE_RATE we assume 100% acceptance → inflated ROI

    df["expected_revenue_saved"] = (
        df["churn_prob"] *
        OFFER_ACCEPTANCE_RATE *
        df["uplift"] *
        df["LTV"]
    )

    df["expected_profit"] = (
        df["expected_revenue_saved"] -
        df["intervention_cost"]
    )

    # -------------------------------------------------------------------------
    # STEP 6: RISK ADJUSTMENT
    # -------------------------------------------------------------------------
    # Penalize uncertain predictions (churn_prob near 0.5)
    # Reward confident predictions (churn_prob near 0 or 1)
    #
    # churn_prob = 0.90 → confidence = |0.90 - 0.5| × 2 = 0.80
    # churn_prob = 0.55 → confidence = |0.55 - 0.5| × 2 = 0.10

    confidence = np.abs(df["churn_prob"] - 0.5) * 2

    df["risk_adjusted_profit"] = df["expected_profit"] * confidence

    # -------------------------------------------------------------------------
    # STEP 7: KEEP ONLY PROFITABLE CUSTOMERS
    # -------------------------------------------------------------------------
    df = df[df["risk_adjusted_profit"] > 0]

    logger.info(f"Profitable customers: {len(df)}")

    if len(df) == 0:
        logger.warning("No profitable customers found")
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # STEP 8: SORT BY BEST VALUE FIRST
    # -------------------------------------------------------------------------
    df = df.sort_values(by="risk_adjusted_profit", ascending=False)

    # -------------------------------------------------------------------------
    # STEP 9: APPLY BUDGET CONSTRAINT
    # -------------------------------------------------------------------------
    df["cumulative_cost"] = df["intervention_cost"].cumsum()
    df = df[df["cumulative_cost"] <= MONTHLY_BUDGET]

    # -------------------------------------------------------------------------
    # STEP 10: LOG FINAL BUSINESS METRICS
    # -------------------------------------------------------------------------
    total_cost   = df["intervention_cost"].sum()
    total_profit = df["expected_profit"].sum()
    roi          = total_profit / total_cost if total_cost > 0 else 0

    logger.info(f"Selected under budget: {len(df)}")
    logger.info(f"Total Cost:            ₹{total_cost:,.2f}")
    logger.info(f"Total Expected Profit: ₹{total_profit:,.2f}")
    logger.info(f"ROI:                   {roi:.2f}x")

    if roi > 10:
        logger.warning(
            "ROI still high — consider lowering OFFER_ACCEPTANCE_RATE "
            "or BASE_RETENTION_UPLIFT in config.py"
        )

    # -------------------------------------------------------------------------
    # RETURN FINAL OUTPUT
    # -------------------------------------------------------------------------
    columns_to_return = [
        "churn_prob",
        "uplift",
        "LTV",
        "intervention_cost",
        "expected_revenue_saved",
        "expected_profit",
        "risk_adjusted_profit",
        "MonthlyCharges",
        "tenure",
        "Contract",
        "InternetService"
    ]

    existing_columns = [col for col in columns_to_return if col in df.columns]

    return df[existing_columns]