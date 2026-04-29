# =============================================================================
# optimizer.py
# =============================================================================

import pandas as pd
import numpy as np

from src.config import (
    INTERVENTION_COST,
    RETENTION_UPLIFT,
    MONTHLY_BUDGET
)
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_ltv(df: pd.DataFrame) -> pd.Series:
    """
    Simple LTV approximation:
    MonthlyCharges × remaining months
    """
    return df["MonthlyCharges"] * (72 - df["tenure"])


def optimize_intervention(df: pd.DataFrame, churn_prob: np.ndarray) -> pd.DataFrame:
    """
    Select customers to target based on profit.

    Args:
        df: original dataframe
        churn_prob: predicted churn probabilities

    Returns:
        DataFrame with selected customers
    """

    logger.info("Starting optimization...")

    df = df.copy()

    # -------------------------------------------------------------------------
    # STEP 1: Add model output
    # -------------------------------------------------------------------------
    df["churn_prob"] = churn_prob

    # -------------------------------------------------------------------------
    # STEP 2: Calculate LTV
    # -------------------------------------------------------------------------
    df["LTV"] = calculate_ltv(df)

    # -------------------------------------------------------------------------
    # STEP 3: Expected value
    # -------------------------------------------------------------------------
    df["expected_revenue_saved"] = (
        df["churn_prob"] * df["LTV"] * RETENTION_UPLIFT
    )

    df["expected_profit"] = (
        df["expected_revenue_saved"] - INTERVENTION_COST
    )

    # -------------------------------------------------------------------------
    # STEP 4: Filter only profitable customers
    # -------------------------------------------------------------------------
    df = df[df["expected_profit"] > 0]

    logger.info(f"Profitable customers: {len(df)}")

    # -------------------------------------------------------------------------
    # STEP 5: Rank by profit
    # -------------------------------------------------------------------------
    df = df.sort_values(by="expected_profit", ascending=False)

    # -------------------------------------------------------------------------
    # STEP 6: Apply budget constraint
    # -------------------------------------------------------------------------
    df["cumulative_cost"] = np.arange(1, len(df) + 1) * INTERVENTION_COST

    df = df[df["cumulative_cost"] <= MONTHLY_BUDGET]

    logger.info(f"Selected customers under budget: {len(df)}")

    # -------------------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------------------
    return df[
        [
            "churn_prob",
            "LTV",
            "expected_revenue_saved",
            "expected_profit"
        ]
    ]