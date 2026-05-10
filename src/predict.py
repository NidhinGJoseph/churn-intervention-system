# =============================================================================
# predict.py (DECISION ENGINE)
# =============================================================================

import joblib
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess
from src.features import build_features

from src.optimizer import optimize_intervention
from src.threshold_optimizer import evaluate_thresholds, get_best_threshold

from src.config import MODEL_PATH, MODELS_DIR, INTERVENTION_PATH
from src.utils import get_logger

logger = get_logger(__name__)


def predict():

    logger.info("=" * 60)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD MODEL + ENCODER
    # -------------------------------------------------------------------------
    logger.info("Loading model and encoder...")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(MODELS_DIR / "encoder.pkl")

    # -------------------------------------------------------------------------
    # STEP 2: LOAD DATA
    # -------------------------------------------------------------------------
    df_raw = load_raw_data()
    df_clean = preprocess(df_raw)

    logger.info(f"Data loaded: {df_clean.shape}")

    # -------------------------------------------------------------------------
    # STEP 3: BUILD FEATURES
    # -------------------------------------------------------------------------
    X, _, _ = build_features(df_clean, encoder=encoder)

    # -------------------------------------------------------------------------
    # STEP 4: PREDICT CHURN PROBABILITY
    # -------------------------------------------------------------------------
    churn_prob = model.predict_proba(X)[:, 1]

    logger.info("Churn probabilities generated")

    # -------------------------------------------------------------------------
    # STEP 5: FIND BEST THRESHOLD
    # -------------------------------------------------------------------------
    threshold_results = evaluate_thresholds(df_clean, churn_prob)

    best = get_best_threshold(threshold_results)
    best_threshold = best["threshold"]

    logger.info(f"Optimal threshold selected: {best_threshold:.2f}")

    # -------------------------------------------------------------------------
    # STEP 6: FINAL OPTIMIZATION
    # -------------------------------------------------------------------------
    mask = churn_prob >= best_threshold

    df_selected = df_clean[mask]
    prob_selected = churn_prob[mask]

    final_df = optimize_intervention(df_selected, prob_selected)

    # -------------------------------------------------------------------------
    # STEP 7: SAVE OUTPUT
    # -------------------------------------------------------------------------
    INTERVENTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(INTERVENTION_PATH, index=False)

    logger.info(f"Saved intervention list → {INTERVENTION_PATH}")

    # -------------------------------------------------------------------------
    # STEP 8: BUSINESS SUMMARY (THIS IS WHAT MATTERS)
    # -------------------------------------------------------------------------
    total_profit = final_df["expected_profit"].sum()
    total_cost = final_df["intervention_cost"].sum()
    roi = total_profit / total_cost if total_cost > 0 else 0

    logger.info("=" * 60)
    logger.info("FINAL BUSINESS METRICS")
    logger.info(f"Customers Targeted: {len(final_df)}")
    logger.info(f"Total Cost: ₹{total_cost:.2f}")
    logger.info(f"Total Expected Profit: ₹{total_profit:.2f}")
    logger.info(f"ROI: {roi:.2f}")
    logger.info("=" * 60)

    return final_df


if __name__ == "__main__":
    predict()