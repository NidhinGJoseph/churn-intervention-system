# =============================================================================
# features.py
# -----------------------------------------------------------------------------
# PURPOSE:
#   Transform the clean DataFrame into a model-ready feature matrix.
#   Every decision here is backed by EDA findings.
#
# WHAT THIS FILE DOES:
#   1. Create tenure_group feature    (EDA: clear churn drop-off per bucket)
#   2. Create monthly_ratio feature   (spending intensity signal)
#   3. One-hot encode categoricals    (sklearn OneHotEncoder — production ready)
#   4. Separate X (features) and y (target)
#
# WHAT THIS FILE DOES NOT DO:
#   - No data quality fixes           (that's data_preprocessing.py)
#   - No scaling                      (that's inside Pipeline in model.py)
#   - No model training               (that's train.py)
#
# USED BY:
#   - train.py    → calls build_features() to get X, y for training
#   - predict.py  → calls build_features() to get X for prediction
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder   
# OneHotEncoder converts categorical columns to binary columns
# Example: Contract [Month-to-month, One year, Two year]
#          → Contract_Month-to-month, Contract_One year, Contract_Two year
#          Each column is 0 or 1

from src.config import TARGET_COLUMN
from src.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# COLUMN DEFINITIONS
# -----------------------------------------------------------------------------
# We define which columns are categorical and which are numeric HERE
# so that if the dataset changes, we only update one place.
# =============================================================================

# Categorical columns that need to be one-hot encoded
# These are all string columns with a fixed set of categories
CATEGORICAL_COLS = [
    "Contract",          # Month-to-month, One year, Two year
    "InternetService",   # DSL, Fiber optic, No
    "PaymentMethod",     # Electronic check, Mailed check, etc.
    "TechSupport",       # Yes, No, No internet service
    "OnlineSecurity",    # Yes, No, No internet service
    "OnlineBackup",      # Yes, No, No internet service
    "DeviceProtection",  # Yes, No, No internet service
    "StreamingTV",       # Yes, No, No internet service
    "StreamingMovies",   # Yes, No, No internet service
    "MultipleLines",     # Yes, No, No phone service
    "PhoneService",      # Yes, No
    "PaperlessBilling",  # Yes, No
    "Partner",           # Yes, No
    "Dependents",        # Yes, No
]

# Numeric columns — passed through as-is (scaling happens in Pipeline)
NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "monthly_ratio",     # engineered feature we create below
]


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def build_features(df: pd.DataFrame, encoder=None):
    """
    Builds the final feature matrix from the clean DataFrame.

    WHY encoder=None?
        During TRAINING:  encoder=None → we create and fit a new encoder
                          on the training data
        During PREDICTION: encoder=fitted_encoder → we reuse the same
                          encoder fitted on training data

        This is critical to avoid data leakage:
        The encoder must ONLY learn category mappings from training data.
        If we refit on test/new data, we'd be cheating.

    HOW TO USE:

        # During training:
        X_train, y_train, encoder = build_features(df_train)

        # During prediction on new data:
        X_new, _, _ = build_features(df_new, encoder=encoder)

    Args:
        df (pd.DataFrame): Clean DataFrame from data_preprocessing.py
        encoder: Fitted OneHotEncoder or None

    Returns:
        X (pd.DataFrame): Feature matrix ready for model
        y (pd.Series):    Target variable (1=churned, 0=not churned)
                          Returns None if TARGET_COLUMN not in df
        encoder:          Fitted OneHotEncoder (reuse this at predict time)
    """

    logger.info("Building features...")

    # Always work on a copy — never modify the input DataFrame
    df = df.copy()

    # -------------------------------------------------------------------------
    # STEP 1: ENGINEER NEW FEATURES
    # -------------------------------------------------------------------------
    df = _create_tenure_group(df)
    df = _create_monthly_ratio(df)

    # -------------------------------------------------------------------------
    # STEP 2: SEPARATE TARGET FROM FEATURES
    # We do this before encoding so the target column is never accidentally
    # passed through the encoder
    # -------------------------------------------------------------------------
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN]               # y = what we want to predict
        df = df.drop(columns=[TARGET_COLUMN])  # Remove target from features
    else:
        # At prediction time, there may be no target column — that's fine
        y = None

    # -------------------------------------------------------------------------
    # STEP 3: ENCODE CATEGORICAL COLUMNS
    # -------------------------------------------------------------------------
    df, encoder = _encode_categoricals(df, encoder)

    # -------------------------------------------------------------------------
    # STEP 4: SELECT FINAL FEATURE COLUMNS
    # After encoding, categorical columns become many binary columns.
    # We combine numeric columns + encoded columns into final X.
    # -------------------------------------------------------------------------

    # Get the names of encoded columns from the encoder
    encoded_col_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    # get_feature_names_out() returns names like:
    # ["Contract_Month-to-month", "Contract_One year", "Contract_Two year", ...]

    # Only keep numeric cols that actually exist in df
    # (monthly_ratio might not exist if TotalCharges was 0 for all rows)
    existing_numeric = [col for col in NUMERIC_COLS if col in df.columns]

    # Final feature columns = numeric + encoded categorical
    final_cols = existing_numeric + encoded_col_names

    X = df[final_cols]

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"  Numeric features:     {len(existing_numeric)}")
    logger.info(f"  Encoded features:     {len(encoded_col_names)}")

    return X, y, encoder


# =============================================================================
# STEP 1A: CREATE TENURE GROUP
# =============================================================================

def _create_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buckets tenure (months) into 4 groups based on EDA findings.

    WHY THIS FEATURE (from EDA):
        Raw tenure is a continuous number (0-72).
        But our EDA showed churn drops sharply at specific thresholds:
            0-12  months: 47.7% churn  ← very high risk
            13-24 months: 28.7% churn
            25-48 months: 20.4% churn
            49-72 months:  9.5% churn  ← very loyal

        Bucketing captures this non-linear relationship more clearly
        than the raw number alone.

        This is called "binning" or "discretization" in feature engineering.

    Args:
        df (pd.DataFrame): Clean DataFrame

    Returns:
        pd.DataFrame: DataFrame with new tenure_group column added
    """

    logger.info("  Creating tenure_group feature...")

    # Define the bin edges and their labels
    # pd.cut() assigns each tenure value to a bucket
    bins   = [0, 12, 24, 48, 72]
    labels = ["0-12m", "13-24m", "25-48m", "49-72m"]

    # include_lowest=True means the first bin includes 0
    # right=True means bins are (0,12], (12,24], (24,48], (48,72]
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    # Convert to string so OneHotEncoder can handle it
    # pd.cut returns a "Categorical" dtype which can confuse sklearn
    df["tenure_group"] = df["tenure_group"].astype(str)

    # Add tenure_group to categorical columns list so it gets encoded
    if "tenure_group" not in CATEGORICAL_COLS:
        CATEGORICAL_COLS.append("tenure_group")

    logger.info(f"  tenure_group distribution:\n{df['tenure_group'].value_counts().sort_index()}")

    return df


# =============================================================================
# STEP 1B: CREATE MONTHLY RATIO
# =============================================================================

def _create_monthly_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates monthly_ratio = MonthlyCharges / TotalCharges.

    WHY THIS FEATURE:
        This captures "spending intensity" — how much of a customer's
        total spend is happening RIGHT NOW vs historically.

        High ratio → customer is paying more now than their historical average
                   → could signal a recent price increase → churn risk
        Low ratio  → customer has been paying consistently for a long time
                   → loyal, stable customer

        Example:
            Customer A: MonthlyCharges=$80, TotalCharges=$80  → ratio=1.0
                        (brand new customer, first month)
            Customer B: MonthlyCharges=$80, TotalCharges=$960 → ratio=0.083
                        (paying $80/month for 12 months, very stable)

    HANDLING DIVISION BY ZERO:
        New customers have TotalCharges=0 (we filled NaN with 0).
        Dividing by 0 gives infinity (inf) in pandas.
        We replace inf with 0 — these customers have no history yet.

    Args:
        df (pd.DataFrame): Clean DataFrame

    Returns:
        pd.DataFrame: DataFrame with new monthly_ratio column added
    """

    logger.info("  Creating monthly_ratio feature...")

    # Divide MonthlyCharges by TotalCharges
    # np.where handles division by zero:
    #   if TotalCharges == 0 → return 0
    #   otherwise → return MonthlyCharges / TotalCharges
    df["monthly_ratio"] = np.where(
        df["TotalCharges"] == 0,          # condition: is TotalCharges zero?
        0,                                 # if yes → return 0
        df["MonthlyCharges"] / df["TotalCharges"]  # if no → divide
    )

    logger.info(f"  monthly_ratio — mean: {df['monthly_ratio'].mean():.4f}, "
                f"max: {df['monthly_ratio'].max():.4f}")

    return df


# =============================================================================
# STEP 3: ENCODE CATEGORICAL COLUMNS
# =============================================================================

def _encode_categoricals(df: pd.DataFrame, encoder=None):
    """
    One-hot encodes all categorical columns using sklearn OneHotEncoder.

    WHY OneHotEncoder INSTEAD OF pd.get_dummies()?
        pd.get_dummies() is fine for notebooks but has a critical flaw:
        it re-encodes based on whatever categories exist in the current data.
        If a category is missing in new data, the columns won't match.

        OneHotEncoder solves this:
        - Fit once on training data → learns all possible categories
        - Transform any new data → always produces the same columns
        - Can be saved and reloaded alongside the model

    HOW ONE-HOT ENCODING WORKS:
        Contract column has 3 values: Month-to-month, One year, Two year
        After encoding it becomes 3 binary columns:
            Contract_Month-to-month  Contract_One year  Contract_Two year
        Customer 1 (Month-to-month):    1                  0                0
        Customer 2 (One year):          0                  1                0
        Customer 3 (Two year):          0                  0                1

    WHY drop="first"?
        If we keep all 3 columns, they're perfectly correlated:
        knowing 2 columns tells you the 3rd.
        Dropping the first category prevents this "multicollinearity".
        This matters especially for Logistic Regression.

    Args:
        df (pd.DataFrame): Clean DataFrame with categorical columns
        encoder: Fitted OneHotEncoder or None

    Returns:
        df (pd.DataFrame): DataFrame with encoded columns added,
                           original categorical columns removed
        encoder: Fitted OneHotEncoder
    """

    logger.info("  Encoding categorical columns...")

    # Only encode columns that actually exist in the DataFrame
    existing_cats = [col for col in CATEGORICAL_COLS if col in df.columns]

    if encoder is None:
        # TRAINING TIME: Create and fit a new encoder on this data
        encoder = OneHotEncoder(
            drop="first",        # Drop first category to avoid multicollinearity
            sparse_output=False, # Return a regular numpy array, not a sparse matrix
                                 # sparse_output=False makes it easier to work with
            handle_unknown="ignore"  # If new categories appear at predict time,
                                     # ignore them instead of crashing
        )

        # fit_transform() = fit (learn categories) + transform (encode) in one step
        # We only do this during training
        encoded_array = encoder.fit_transform(df[existing_cats])
        logger.info("  Encoder fitted on training data")

    else:
        # PREDICTION TIME: Use the already-fitted encoder
        # transform() only — never refit on new data
        encoded_array = encoder.transform(df[existing_cats])
        logger.info("  Using pre-fitted encoder")

    # Get the column names for the encoded columns
    # e.g. ["Contract_One year", "Contract_Two year", ...]
    encoded_col_names = encoder.get_feature_names_out(existing_cats).tolist()

    # Create a DataFrame from the encoded array with proper column names
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_col_names,
        index=df.index    # Keep the same row index as original df
                          # Important for joining back correctly
    )

    # Remove original categorical columns from df
    df = df.drop(columns=existing_cats)

    # Add encoded columns to df
    df = pd.concat([df, encoded_df], axis=1)
    # axis=1 means join side by side (columns), not top to bottom (rows)

    logger.info(f"  Encoded {len(existing_cats)} categorical columns "
                f"→ {len(encoded_col_names)} binary columns")

    return df, encoder


# =============================================================================
# QUICK TEST
# Run directly to test feature engineering works:
#   python src/features.py
# =============================================================================

if __name__ == "__main__":

    from src.data_loader import load_raw_data
    from src.data_preprocessing import preprocess

    # Load and preprocess
    df_raw   = load_raw_data()
    df_clean = preprocess(df_raw)

    # Build features
    X, y, encoder = build_features(df_clean)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape:         {y.shape}")
    print(f"Churn rate:           {y.mean()*100:.1f}%")
    print(f"\nFeature columns:\n{list(X.columns)}")
    print(f"\nSample X:\n{X.head()}")