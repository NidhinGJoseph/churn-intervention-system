# =============================================================================
# train.py (CORRECTED)
# =============================================================================

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

from sklearn.calibration import CalibratedClassifierCV

from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess
from src.features import build_features
from src.model import get_candidate_models
from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    MODELS_DIR
)
from src.utils import get_logger

logger = get_logger(__name__)


def train():

    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD DATA
    # -------------------------------------------------------------------------
    df_raw = load_raw_data()
    logger.info(f"Raw shape: {df_raw.shape}")

    # -------------------------------------------------------------------------
    # STEP 2: PREPROCESS
    # -------------------------------------------------------------------------
    df_clean = preprocess(df_raw)
    logger.info(f"Clean shape: {df_clean.shape}")

    # -------------------------------------------------------------------------
    # STEP 3: FEATURES
    # -------------------------------------------------------------------------
    X, y, encoder = build_features(df_clean)
    logger.info(f"Features: {X.shape}")
    logger.info(f"Churn rate: {y.mean()*100:.2f}%")

    # -------------------------------------------------------------------------
    # STEP 4: TRAIN / TEST SPLIT
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 🔥 CRITICAL: Create validation set for calibration
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    logger.info(f"Train main: {X_train_main.shape}")
    logger.info(f"Validation: {X_val.shape}")
    logger.info(f"Test: {X_test.shape}")

    # -------------------------------------------------------------------------
    # STEP 5: TRAIN MODELS
    # -------------------------------------------------------------------------
    models = get_candidate_models()
    results = {}

    for name, pipeline in models.items():

        logger.info(f"\nTraining {name}...")

        # Cross-validation on TRAIN MAIN only
        cv_scores = cross_val_score(
            pipeline,
            X_train_main,
            y_train_main,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )

        logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on TRAIN MAIN
        pipeline.fit(X_train_main, y_train_main)

        # Validate BEFORE calibration (optional check)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)

        logger.info(f"Validation AUC (pre-calibration): {val_auc:.4f}")

        # ---------------------------------------------------------------------
        # CALIBRATION (CORRECT)
        # ---------------------------------------------------------------------
        calibrated_model = CalibratedClassifierCV(
            pipeline,
            method="sigmoid",
            cv=5
        )

        calibrated_model.fit(X_train_main, y_train_main)

        # ---------------------------------------------------------------------
        # TEST EVALUATION
        # ---------------------------------------------------------------------
        y_test_prob = calibrated_model.predict_proba(X_test)[:, 1]
        y_test_pred = calibrated_model.predict(X_test)

        test_auc = roc_auc_score(y_test, y_test_prob)

        logger.info(f"Test AUC (calibrated): {test_auc:.4f}")
        logger.info("\n" + classification_report(y_test, y_test_pred))