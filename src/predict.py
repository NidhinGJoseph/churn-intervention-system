# =============================================================================
# predict.py
# =============================================================================

import joblib
import pandas as pd

from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess
from src.features import build_features
from src.optimizer import optimize_intervention
from src.intervention import build_intervention_list

from src.config import MODEL_PATH, MODELS_DIR, INTERVENTION_PATH
from src.utils import get_logger

logger = get_logger(__name__)


def predict() -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("=" * 60)

    # STEP 1: Load model and encoder
    logger.info("Step 1: Loading model and encoder...")
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(MODELS_DIR / "encoder.pkl")

    # STEP 2: Load and preprocess data
    logger.info("Step 2: Loading and preprocessing data...")
    df_raw   = load_raw_data()
    df_clean = preprocess(df_raw)
    logger.info(f"  Data shape: {df_clean.shape}")

    # STEP 3: Build features
    logger.info("Step 3: Building features...")
    X, _, _ = build_features(df_clean, encoder=encoder)
    logger.info(f"  Feature matrix: {X.shape}")

    # STEP 4: Generate churn probabilities
    logger.info("Step 4: Generating churn probabilities...")
    churn_prob = model.predict_proba(X)[:, 1]
    logger.info(f"  Avg churn probability: {churn_prob.mean():.3f}")
    logger.info(f"  Customers above 30% threshold: {(churn_prob >= 0.30).sum():,}")

    # STEP 5: Optimize intervention
    logger.info("Step 5: Running optimizer...")
    final_df = optimize_intervention(df_clean, churn_prob)

    if final_df.empty:
        logger.warning("No customers selected — intervention list is empty")
        return final_df

    # STEP 6: Format intervention list
    # Adds priority tier, offer type, recommended action, risk segment
    logger.info("Step 6: Formatting intervention list...")
    final_df = build_intervention_list(final_df)

    # STEP 7: Save output
    logger.info("Step 7: Saving intervention list...")
    INTERVENTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(INTERVENTION_PATH, index=False)
    logger.info(f"  Saved → {INTERVENTION_PATH}")

    # STEP 8: Business summary
    total_cost   = final_df["intervention_cost"].sum()
    total_profit = final_df["expected_profit"].sum()
    roi          = total_profit / total_cost if total_cost > 0 else 0

    # expected_revenue_saved may not exist after intervention formatting
    total_revenue_saved = (
    final_df["expected_revenue_saved"].sum()
    if "expected_revenue_saved" in final_df.columns
    else total_profit
    )
                
    logger.info("=" * 60)
    logger.info("FINAL BUSINESS METRICS")
    logger.info("=" * 60)
    logger.info(f"  Customers Targeted:        {len(final_df):,}")
    logger.info(f"  Budget Used:               ₹{total_cost:,.2f} / ₹5,000.00")
    logger.info(f"  Expected Revenue Saved:    ₹{total_revenue_saved:,.2f}")
    logger.info(f"  Expected Profit:           ₹{total_profit:,.2f}")
    logger.info(f"  ROI:                       {roi:.2f}x")
    logger.info("=" * 60)

    return final_df


if __name__ == "__main__":
    predict()