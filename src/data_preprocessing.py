# =============================================================================
# data_preprocessing.py
# PURPOSE:
#   Clean raw Telco churn data before feature engineering.
# =============================================================================

import pandas as pd

from src.config import TARGET_COLUMN, COLUMNS_TO_DROP


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full preprocessing pipeline.
    """
    df = df.copy()

    df = _fix_total_charges(df)
    df = _handle_missing_total_charges(df)
    df = _drop_columns(df)
    df = _encode_target(df)

    return df


# -----------------------------------------------------------------------------
# STEP 1: Fix TotalCharges dtype
# -----------------------------------------------------------------------------
def _fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    # Convert to numeric (blank values → NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


# -----------------------------------------------------------------------------
# STEP 2: Handle missing TotalCharges
# -----------------------------------------------------------------------------
def _handle_missing_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    # Validate assumption: NaNs should only exist for tenure = 0
    invalid_rows = df[(df["TotalCharges"].isna()) & (df["tenure"] != 0)]

    if len(invalid_rows) > 0:
        raise ValueError("Unexpected NaNs in TotalCharges for non-zero tenure")

    # Fill NaN with 0 (new customers not yet billed)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


# -----------------------------------------------------------------------------
# STEP 3: Drop unnecessary columns
# -----------------------------------------------------------------------------
def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    # customerID → identifier (no predictive value)
    # gender → negligible signal (based on EDA)
    drop_cols = COLUMNS_TO_DROP + ["gender"]

    existing = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing)

    return df


# -----------------------------------------------------------------------------
# STEP 4: Encode target variable
# -----------------------------------------------------------------------------
def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    # Convert Yes/No → 1/0
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_raw_data

    df_raw = load_raw_data()
    df_clean = preprocess(df_raw)

    print("Preprocessing complete")
    print(df_clean.head())
    print("\nShape:", df_clean.shape)