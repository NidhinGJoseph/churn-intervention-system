# =============================================================================
# train.py
# -----------------------------------------------------------------------------
# PURPOSE:
#   Orchestrates the full training pipeline from raw data to saved model.
#   After training, automatically runs evaluation and saves all plots.
#
# HOW TO RUN:
#   python -m src.train
#
# OUTPUT:
#   models/churn_model.pkl     → saved best model
#   models/encoder.pkl         → saved encoder
#   outputs/plots/             → all evaluation plots
# =============================================================================

import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess
from src.features import build_features
from src.model import get_candidate_models
from src.evaluation import evaluate_model          # ← added
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
    # encoder=None → training time → fit a new encoder
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
        stratify=y           # Preserves churn ratio in both splits
    )

    # Validation split for sanity check during training
    # Does NOT affect test set — test set stays completely untouched
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    logger.info(f"Train: {X_train_main.shape} | "
                f"Val: {X_val.shape} | "
                f"Test: {X_test.shape}")

    # -------------------------------------------------------------------------
    # STEP 5: TRAIN AND COMPARE ALL CANDIDATE MODELS
    # -------------------------------------------------------------------------
    models  = get_candidate_models()
    results = {}

    for name, pipeline in models.items():

        logger.info(f"\n--- Training {name} ---")

        # Cross-validation on training data
        # Gives reliable performance estimate across 5 different splits
        cv_scores = cross_val_score(
            pipeline,
            X_train_main,
            y_train_main,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        logger.info(f"CV AUC:         {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train on full training set
        pipeline.fit(X_train_main, y_train_main)

        # Validation check — quick sanity before calibration
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        val_auc    = roc_auc_score(y_val, y_val_prob)
        logger.info(f"Validation AUC: {val_auc:.4f}")

        # ---------------------------------------------------------------------
        # CALIBRATION
        # ---------------------------------------------------------------------
        # CalibratedClassifierCV with cv=5 refits the base model 5 times
        # and learns probability calibration on each fold.
        # Ensures predicted probabilities are reliable inputs for optimizer.
        calibrated_model = CalibratedClassifierCV(
            pipeline,
            method="sigmoid",
            cv=5
        )
        calibrated_model.fit(X_train_main, y_train_main)

        # ---------------------------------------------------------------------
        # TEST SET EVALUATION
        # ---------------------------------------------------------------------
        y_test_prob = calibrated_model.predict_proba(X_test)[:, 1]
        y_test_pred = calibrated_model.predict(X_test)

        test_auc = roc_auc_score(y_test, y_test_prob)

        logger.info(f"Test AUC:       {test_auc:.4f}")
        logger.info(f"\n{classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn'])}")

        results[name] = {
            "model":    calibrated_model,
            "test_auc": test_auc
        }

    # -------------------------------------------------------------------------
    # STEP 6: SELECT BEST MODEL
    # -------------------------------------------------------------------------
    logger.info("\n--- Model Comparison ---")
    for name, result in results.items():
        logger.info(f"  {name:<25} Test AUC: {result['test_auc']:.4f}")

    best_name  = max(results, key=lambda x: results[x]["test_auc"])
    best_model = results[best_name]["model"]

    logger.info(f"\n✅ Best model: {best_name} "
                f"(Test AUC: {results[best_name]['test_auc']:.4f})")

    # -------------------------------------------------------------------------
    # STEP 7: EVALUATE BEST MODEL
    # -------------------------------------------------------------------------
    # Runs full evaluation: ROC curve, PR curve, confusion matrix,
    # lift curve, calibration curve, feature importance
    # All plots saved to outputs/plots/
    logger.info("\n--- Running Full Evaluation ---")

    evaluate_model(
        best_model,
        X_test,
        y_test,
        feature_names=list(X_test.columns)
    )

    # -------------------------------------------------------------------------
    # STEP 8: SAVE MODEL AND ENCODER
    # -------------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"Saved model   → {MODEL_PATH}")

    encoder_path = MODELS_DIR / "encoder.pkl"
    joblib.dump(encoder, encoder_path)
    logger.info(f"Saved encoder → {encoder_path}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return best_model, encoder


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    train()