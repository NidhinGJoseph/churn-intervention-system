# =============================================================================
# optimizer.py (FINAL - CONSISTENT + BUSINESS-ALIGNED)
# =============================================================================

import pandas as pd
import numpy as np

from src.config import (
    OFFER_COST_PERCENT,
    RETENTION_UPLIFT,
    MONTHLY_BUDGET,
    MAX_TENURE,
    MIN_MONTHLY_CHARGES,
    MIN_TENURE_MONTHS,
    MIN_CHURN_PROB,
    EXCLUDE_CONTRACT_TYPE,
    EXCLUDE_INTERNET_SERVICE
)
from src.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# LTV CALCULATION
# =============================================================================
def calculate_ltv(df: pd.DataFrame) -> pd.Series:
    """
    LTV = MonthlyCharges × remaining tenure
    """

    remaining_months = np.maximum(0, MAX_TENURE - df["tenure"])
    return df["MonthlyCharges"] * remaining_months


# =============================================================================
# OPTIMIZER
# =============================================================================
def optimize_intervention(df: pd.DataFrame, churn_prob: np.ndarray) -> pd.DataFrame:

    logger.info("Starting optimization...")

    df = df.copy()

    # -------------------------------------------------------------------------
    # STEP 1: Attach model predictions
    # -------------------------------------------------------------------------
    df["churn_prob"] = churn_prob

    # -------------------------------------------------------------------------
    # STEP 2: Calculate LTV
    # -------------------------------------------------------------------------
    df["LTV"] = calculate_ltv(df)

    # -------------------------------------------------------------------------
    # STEP 3: Calculate intervention cost (dynamic)
    # -------------------------------------------------------------------------
    df["intervention_cost"] = df["MonthlyCharges"] * OFFER_COST_PERCENT

    # -------------------------------------------------------------------------
    # STEP 4: Expected value calculation
    # -------------------------------------------------------------------------
    df["expected_revenue_saved"] = (
        df["churn_prob"] *
        df["LTV"] *
        RETENTION_UPLIFT
    )

    df["expected_profit"] = (
        df["expected_revenue_saved"] - df["intervention_cost"]
    )

    # -------------------------------------------------------------------------
    # STEP 5: BUSINESS FILTERS (config-driven)
    # -------------------------------------------------------------------------
    df = df[
        (df["MonthlyCharges"] >= MIN_MONTHLY_CHARGES) &
        (df["tenure"] >= MIN_TENURE_MONTHS) &
        (df["churn_prob"] >= MIN_CHURN_PROB) &
        (~df["Contract"].isin(EXCLUDE_CONTRACT_TYPE)) &
        (~df["InternetService"].isin(EXCLUDE_INTERNET_SERVICE)) &
        (df["LTV"] > 0)
    ]

    logger.info(f"After business filters: {len(df)}")

    # -------------------------------------------------------------------------
    # STEP 6: Keep only profitable customers
    # -------------------------------------------------------------------------
    df = df[df["expected_profit"] > 0]

    logger.info(f"Profitable customers: {len(df)}")

    # -------------------------------------------------------------------------
    # STEP 7: Rank by profit
    # -------------------------------------------------------------------------
    df = df.sort_values(by="expected_profit", ascending=False)

    # -------------------------------------------------------------------------
    # STEP 8: Apply budget constraint
    # -------------------------------------------------------------------------
    df["cumulative_cost"] = df["intervention_cost"].cumsum()

    df = df[df["cumulative_cost"] <= MONTHLY_BUDGET]

    logger.info(f"Selected under budget: {len(df)}")

    # -------------------------------------------------------------------------
    # STEP 9: BUSINESS METRICS (CRITICAL)
    # -------------------------------------------------------------------------
    total_profit = df["expected_profit"].sum()
    total_cost = df["intervention_cost"].sum()

    roi = total_profit / total_cost if total_cost > 0 else 0

    logger.info(f"Total Cost: ₹{total_cost:.2f}")
    logger.info(f"Total Expected Profit: ₹{total_profit:.2f}")
    logger.info(f"ROI: {roi:.2f}")

    # -------------------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------------------
    return df[
        [
            "churn_prob",
            "LTV",
            "intervention_cost",
            "expected_revenue_saved",
            "expected_profit",
            "MonthlyCharges",
            "tenure"
        ]
    ]