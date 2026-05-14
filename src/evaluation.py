# =============================================================================
# evaluation.py
# -----------------------------------------------------------------------------
# PURPOSE:
#   Evaluates model performance beyond basic metrics.
#   Produces visualizations that tell the business story of the model.
#
# WHAT THIS FILE PRODUCES:
#   1. ROC-AUC curve         — visual model performance
#   2. Precision-Recall curve — better metric for imbalanced data
#   3. Confusion matrix       — where model is right and wrong
#   4. Lift curve             — proves system beats random targeting
#   5. Calibration curve      — proves probabilities are trustworthy
#   6. Feature importance     — which features drove predictions
#
# USED BY:
#   - train.py → called after model training to evaluate and save plots
#
# HOW TO RUN STANDALONE:
#   python src/evaluation.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_curve,                  # Points for ROC curve
    roc_auc_score,              # Area under ROC curve
    precision_recall_curve,     # Points for Precision-Recall curve
    average_precision_score,    # Area under Precision-Recall curve
    confusion_matrix,           # TP, FP, TN, FN counts
    classification_report       # Precision, recall, F1 per class
)

from sklearn.calibration import calibration_curve
# calibration_curve: compares predicted probabilities vs actual frequencies
# Used to verify: "when model says 70% churn, do ~70% actually churn?"

from src.config import OUTPUTS_DIR, RANDOM_STATE
from src.utils import get_logger

logger = get_logger(__name__)

# Set consistent plot style for all charts
sns.set_theme(style="whitegrid", palette="muted")
PLOT_DIR = OUTPUTS_DIR / "plots"


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list = None
) -> dict:
    """
    Runs full model evaluation and saves all plots to outputs/plots/.

    HOW TO USE FROM train.py:
        from src.evaluation import evaluate_model
        metrics = evaluate_model(calibrated_model, X_test, y_test)

    Args:
        model:         Trained + calibrated sklearn pipeline
        X_test:        Test feature matrix
        y_test:        True labels (0/1)
        feature_names: List of feature column names for importance plot

    Returns:
        dict: All computed metrics
    """

    logger.info("Starting model evaluation...")

    # Create plots directory if it doesn't exist
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate predictions
    # predict_proba returns [P(No Churn), P(Churn)]
    # [:, 1] = P(Churn) — the probability we care about
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # -------------------------------------------------------------------------
    # COMPUTE ALL METRICS
    # -------------------------------------------------------------------------
    metrics = _compute_metrics(y_test, y_pred, y_prob)

    # -------------------------------------------------------------------------
    # GENERATE ALL PLOTS
    # -------------------------------------------------------------------------
    _plot_roc_curve(y_test, y_prob, metrics["roc_auc"])
    _plot_precision_recall_curve(y_test, y_prob, metrics["avg_precision"])
    _plot_confusion_matrix(y_test, y_pred)
    _plot_lift_curve(y_test, y_prob)
    _plot_calibration_curve(y_test, y_prob)

    if feature_names is not None:
        _plot_feature_importance(model, feature_names)

    logger.info(f"All plots saved to: {PLOT_DIR}")
    logger.info(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    logger.info(f"Avg Precision: {metrics['avg_precision']:.4f}")

    return metrics


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def _compute_metrics(y_test, y_pred, y_prob) -> dict:
    """
    Computes all evaluation metrics and logs them.

    WHY THESE METRICS?
        ROC-AUC:   Measures ranking ability — how well does the model
                   separate churners from non-churners?
                   1.0 = perfect, 0.5 = random guessing

        Avg Precision: Better than ROC-AUC for imbalanced datasets.
                   Summarizes the Precision-Recall curve.
                   More sensitive to how well we catch actual churners.

        Classification Report: Shows precision, recall, F1 per class.
                   Recall for class 1 (churners) is most important —
                   we want to catch as many actual churners as possible.
    """

    roc_auc       = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    cm            = confusion_matrix(y_test, y_pred)
    report        = classification_report(
                        y_test, y_pred,
                        target_names=["No Churn", "Churn"]
                    )

    logger.info("\nClassification Report:")
    logger.info(f"\n{report}")

    # Extract TP, FP, TN, FN from confusion matrix
    # Confusion matrix layout for binary classification:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"True Positives (caught churners):    {tp}")
    logger.info(f"False Positives (wrong alarms):      {fp}")
    logger.info(f"True Negatives (correct non-churn):  {tn}")
    logger.info(f"False Negatives (missed churners):   {fn}")

    return {
        "roc_auc":       roc_auc,
        "avg_precision": avg_precision,
        "confusion_matrix": cm,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


# =============================================================================
# PLOT 1: ROC CURVE
# =============================================================================

def _plot_roc_curve(y_test, y_prob, roc_auc: float):
    """
    Plots the ROC (Receiver Operating Characteristic) curve.

    WHAT IT SHOWS:
        X-axis: False Positive Rate (% of non-churners wrongly flagged)
        Y-axis: True Positive Rate (% of actual churners caught)

        The diagonal line = random guessing (AUC = 0.5)
        Our curve should be well above the diagonal.

        AUC = 0.85 means: if we randomly pick one churner and one
        non-churner, the model correctly ranks the churner higher 85% of the time.
    """

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.3f})")

    # Diagonal = random guessing baseline
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--",
             lw=1, label="Random Baseline (AUC = 0.50)")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — Churn Prediction Model", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve.png", dpi=150)
    plt.close()

    logger.info("  Saved: roc_curve.png")


# =============================================================================
# PLOT 2: PRECISION-RECALL CURVE
# =============================================================================

def _plot_precision_recall_curve(y_test, y_prob, avg_precision: float):
    """
    Plots the Precision-Recall curve.

    WHY THIS MATTERS MORE THAN ROC FOR THIS PROJECT:
        Our dataset is imbalanced (74% No, 26% Yes).
        ROC-AUC can look good even on imbalanced data because
        it accounts for True Negatives — but we don't care much
        about correctly identifying non-churners.

        Precision-Recall focuses only on the positive class (churners):
        Precision = of all customers we flag, how many actually churn?
        Recall    = of all actual churners, how many did we catch?

        For our retention system, we want HIGH RECALL —
        we'd rather flag some non-churners than miss actual churners.
    """

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    # Baseline = random classifier = churn rate in dataset (~26%)
    baseline = y_test.mean()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="tomato", lw=2,
             label=f"PR Curve (Avg Precision = {avg_precision:.3f})")
    plt.axhline(y=baseline, color="gray", linestyle="--", lw=1,
                label=f"Random Baseline ({baseline:.2f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve — Churn Prediction Model",
              fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()

    logger.info("  Saved: precision_recall_curve.png")


# =============================================================================
# PLOT 3: CONFUSION MATRIX
# =============================================================================

def _plot_confusion_matrix(y_test, y_pred):
    """
    Plots a heatmap of the confusion matrix.

    HOW TO READ IT:
        Rows    = Actual class (what really happened)
        Columns = Predicted class (what model said)

        Top-left  (TN): Correctly predicted NO churn — good
        Top-right (FP): Predicted churn but customer stayed — wasted offer
        Bot-left  (FN): Predicted no churn but customer left — missed opportunity
        Bot-right (TP): Correctly predicted churn — exactly what we want

    FOR OUR BUSINESS PROBLEM:
        False Negatives (FN) are most costly — these are churners we miss
        and never offer a retention deal to.
        We want to minimize FN even at the cost of some FP.
    """

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,         # Show numbers in each cell
        fmt="d",            # Integer format
        cmap="Blues",       # Blue color scale
        xticklabels=["Predicted No Churn", "Predicted Churn"],
        yticklabels=["Actual No Churn", "Actual Churn"]
    )

    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    logger.info("  Saved: confusion_matrix.png")


# =============================================================================
# PLOT 4: LIFT CURVE — THE MOST IMPORTANT PLOT
# =============================================================================

def _plot_lift_curve(y_test, y_prob):
    """
    Plots the Lift Curve.

    THIS IS THE MOST IMPORTANT PLOT FOR THIS PROJECT.

    WHAT IT SHOWS:
        How much better is our model vs randomly targeting customers?

        X-axis: % of customers targeted (sorted by churn score, highest first)
        Y-axis: Lift = how many times better than random

        Example: Lift of 2.5 at 20% means:
            If we target the top 20% of customers by churn score,
            we catch 2.5x more actual churners than if we randomly
            targeted 20% of customers.

    WHY IT MATTERS FOR INTERVIEWS:
        This directly answers: "How much value does your ML system add
        over a simpler approach?" — the core business question.

    HOW IT'S CALCULATED:
        1. Sort customers by predicted churn probability (highest first)
        2. For each % of customers targeted, calculate cumulative churn rate
        3. Divide by the overall churn rate (random baseline)
        4. Plot the ratio — that's your lift
    """

    # Combine predictions into a DataFrame for easy sorting
    lift_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_prob": y_prob
    })

    # Sort by predicted probability descending
    # (highest churn risk customers first)
    lift_df = lift_df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    # Calculate cumulative churn rate as we target more customers
    lift_df["cumulative_churners"] = lift_df["y_true"].cumsum()
    lift_df["cumulative_pct"]      = (lift_df.index + 1) / len(lift_df)

    # Total churn rate in the dataset (random baseline)
    baseline_rate = lift_df["y_true"].mean()

    # Lift = cumulative churn rate / baseline churn rate
    # If lift = 2.5, we're catching 2.5x more churners than random
    lift_df["lift"] = (
        lift_df["cumulative_churners"] /
        ((lift_df.index + 1) * baseline_rate)
    )

    plt.figure(figsize=(10, 6))

    # Model lift curve
    plt.plot(
        lift_df["cumulative_pct"] * 100,
        lift_df["lift"],
        color="steelblue", lw=2,
        label="Model Lift"
    )

    # Random baseline — lift = 1.0 means no better than random
    plt.axhline(y=1.0, color="gray", linestyle="--", lw=1,
                label="Random Baseline (Lift = 1.0)")

    # Mark the 20% and 30% targeting points — commonly used in business
    for pct in [20, 30]:
        idx   = int(len(lift_df) * pct / 100)
        lift  = lift_df.iloc[idx]["lift"]
        plt.axvline(x=pct, color="tomato", linestyle=":", alpha=0.6)
        plt.annotate(
            f"Top {pct}%\nLift={lift:.2f}x",
            xy=(pct, lift),
            xytext=(pct + 2, lift + 0.1),
            fontsize=9, color="tomato"
        )

    plt.xlabel("% of Customers Targeted", fontsize=12)
    plt.ylabel("Lift", fontsize=12)
    plt.title("Lift Curve — Model vs Random Targeting",
              fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "lift_curve.png", dpi=150)
    plt.close()

    logger.info("  Saved: lift_curve.png")

    # Log lift at key targeting percentages
    for pct in [10, 20, 30]:
        idx  = int(len(lift_df) * pct / 100)
        lift = lift_df.iloc[idx]["lift"]
        logger.info(f"  Lift at top {pct}%: {lift:.2f}x")


# =============================================================================
# PLOT 5: CALIBRATION CURVE
# =============================================================================

def _plot_calibration_curve(y_test, y_prob):
    """
    Plots the Calibration Curve.

    WHAT IT SHOWS:
        X-axis: Mean predicted probability (what model says)
        Y-axis: Fraction of actual positives (what actually happened)

        A perfectly calibrated model follows the diagonal.
        If the model says 0.7 probability of churn, ~70% of those
        customers should actually churn.

    WHY IT MATTERS FOR OUR OPTIMIZER:
        Our optimizer makes financial decisions based on churn probabilities.
        If probabilities are wrong (e.g. model says 0.9 but real rate is 0.5),
        the optimizer will make poor targeting decisions.
        Calibration ensures probabilities are trustworthy inputs.
    """

    # n_bins = 10 means we group predictions into 10 probability buckets
    fraction_of_positives, mean_predicted_prob = calibration_curve(
        y_test, y_prob, n_bins=10
    )

    plt.figure(figsize=(8, 6))

    # Perfect calibration diagonal
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--",
             lw=1, label="Perfect Calibration")

    # Model calibration curve
    plt.plot(mean_predicted_prob, fraction_of_positives,
             color="steelblue", lw=2, marker="o",
             label="Model Calibration")

    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Actual Positives", fontsize=12)
    plt.title("Calibration Curve — Are Probabilities Reliable?",
              fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "calibration_curve.png", dpi=150)
    plt.close()

    logger.info("  Saved: calibration_curve.png")


# =============================================================================
# PLOT 6: FEATURE IMPORTANCE
# =============================================================================

def _plot_feature_importance(model, feature_names: list):
    """
    Plots feature importances from the trained model.

    WHY THIS MATTERS:
        Shows which features drove the model's predictions.
        Connects back to EDA findings — we expect Contract, tenure,
        PaymentMethod, InternetService to be top features.

        If unexpected features appear at the top, it could signal
        data leakage or a feature engineering issue.

    HOW IT WORKS:
        For XGBoost/tree models: uses built-in feature_importances_
        For Logistic Regression: uses absolute coefficient values
        Both tell us "how much does each feature contribute to predictions"

    Args:
        model:         Trained pipeline (scaler + model)
        feature_names: List of feature column names
    """

    try:
        # Get the actual model from inside the pipeline
        # pipeline.named_steps["model"] extracts the classifier
        clf = model.named_steps["model"] if hasattr(model, "named_steps") \
              else model.estimator.named_steps["model"]

        # Try XGBoost/Random Forest feature importances first
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_

        # Fall back to Logistic Regression coefficients
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])

        else:
            logger.warning("Model does not support feature importance extraction")
            return

        # Create DataFrame for easy plotting
        importance_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(20)
        # Show top 20 features only — more than that gets cluttered

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df,
            x="importance",
            y="feature",
            hue="feature",
            palette="Blues_r",
            legend=False    # Darker = more important
        )

        plt.title("Top 20 Feature Importances",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "feature_importance.png", dpi=150)
        plt.close()

        logger.info("  Saved: feature_importance.png")

        # Log top 10 features
        logger.info("  Top 10 Features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"    {row['feature']:<40} {row['importance']:.4f}")

    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":

    import joblib
    from src.data_loader import load_raw_data
    from src.data_preprocessing import preprocess
    from src.features import build_features
    from src.config import MODEL_PATH, MODELS_DIR, TEST_SIZE
    from sklearn.model_selection import train_test_split

    # Load model and encoder
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(MODELS_DIR / "encoder.pkl")

    # Load and prepare data
    df_raw   = load_raw_data()
    df_clean = preprocess(df_raw)
    X, y, _  = build_features(df_clean, encoder=encoder)

    # Recreate the same test split used during training
    # IMPORTANT: same random_state and stratify as train.py
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Run evaluation
    metrics = evaluate_model(model, X_test, y_test, feature_names=list(X.columns))

    print(f"\nROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"\nPlots saved to: {PLOT_DIR}")