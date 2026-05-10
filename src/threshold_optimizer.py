# =============================================================================
# threshold_optimizer.py
# =============================================================================

import pandas as pd
import numpy as np

from src.optimizer import optimize_intervention
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_thresholds(df: pd.DataFrame, churn_prob: np.ndarray):
    """
    Evaluate profit across different churn probability thresholds.
    """

    logger.info("Evaluating thresholds...")

    thresholds = np.linspace(0.05, 0.95, 19)

    results = []

    for t in thresholds:

        # Apply threshold
        mask = churn_prob >= t

        df_subset = df[mask]
        prob_subset = churn_prob[mask]

        if len(df_subset) == 0:
            continue

        selected = optimize_intervention(df_subset, prob_subset)

        total_profit = selected["expected_profit"].sum()
        total_cost = selected["intervention_cost"].sum()

        roi = total_profit / total_cost if total_cost > 0 else 0

        results.append({
            "threshold": t,
            "customers": len(selected),
            "total_profit": total_profit,
            "total_cost": total_cost,
            "roi": roi
        })

    results_df = pd.DataFrame(results)

    return results_df


def get_best_threshold(results_df: pd.DataFrame):
    """
    Select threshold that maximizes profit.
    """

    best_row = results_df.loc[results_df["total_profit"].idxmax()]

    logger.info(f"Best threshold: {best_row['threshold']:.2f}")
    logger.info(f"Max profit: ₹{best_row['total_profit']:.2f}")

    return best_row