# =============================================================================
# predict.py
# =============================================================================

import joblib
import pandas as pd

from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess
from src.features import build_features
from src.optimizer import optimize_intervention

from src.config import (
    MODEL_PATH,
    MODELS_DIR,
    INTERVENTION_PATH
)

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
    # STEP 2: LOAD RAW DATA
    # -------------------------------------------------------------------------
    df_raw = load_raw_data()
    logger.info(f"Raw data: {df_raw.shape}")

    # -------------------------------------------------------------------------
    # STEP 3: PREPROCESS
    # -------------------------------------------------------------------------
    df_clean = preprocess(df_raw)
    logger.info(f"Clean data: {df_clean.shape}")

    # -------------------------------------------------------------------------
    # STEP 4: BUILD FEATURES (IMPORTANT: use SAME encoder)
    # -------------------------------------------------------------------------
    X, _, _ = build_features(df_clean, encoder=encoder)
    logger.info(f"Feature matrix: {X.shape}")

    # -------------------------------------------------------------------------
    # STEP 5: PREDICT CHURN PROBABILITY
    # -------------------------------------------------------------------------
    logger.info("Predicting churn probability...")

    churn_prob = model.predict_proba(X)[:, 1]

    # -------------------------------------------------------------------------
    # STEP 6: RUN OPTIMIZER
    # -------------------------------------------------------------------------
    logger.info("Running optimizer...")

    intervention_df = optimize_intervention(df_clean, churn_prob)

    logger.info(f"Final intervention list: {intervention_df.shape}")

    # -------------------------------------------------------------------------
    # STEP 7: SAVE OUTPUT
    # -------------------------------------------------------------------------
    INTERVENTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    intervention_df.to_csv(INTERVENTION_PATH, index=False)

    logger.info(f"Saved intervention list → {INTERVENTION_PATH}")

    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 60)

    return intervention_df


if __name__ == "__main__":
    predict()