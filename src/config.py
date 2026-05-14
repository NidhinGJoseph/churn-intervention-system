# =============================================================================
# config.py
# -----------------------------------------------------------------------------
# Single source of truth for the entire project.
# One definition per setting — no duplicates.
# =============================================================================

from pathlib import Path

# -------------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------------------
# DIRECTORIES
# -------------------------------------------------------------------------
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# -------------------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------------------
RAW_DATA_PATH       = DATA_DIR / "raw" / "telco_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "telco_processed.csv"
MODEL_PATH          = MODELS_DIR / "churn_model.pkl"
INTERVENTION_PATH   = OUTPUTS_DIR / "intervention_list.csv"

# -------------------------------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------------------------------
TARGET_COLUMN   = "Churn"
COLUMNS_TO_DROP = ["customerID"]
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# -------------------------------------------------------------------------
# CUSTOMER ECONOMICS
# -------------------------------------------------------------------------

# Maximum tenure in months for LTV calculation
# IBM Telco dataset caps at 72 months
MAX_TENURE = 72

# Monthly discount rate for future cashflow discounting
# 1% per month ≈ 12% annually — standard telecom assumption
MONTHLY_DISCOUNT_RATE = 0.01

# -------------------------------------------------------------------------
# RETENTION ASSUMPTIONS
# -------------------------------------------------------------------------

# Base probability that a retention offer successfully retains the customer
# 15% is a realistic industry estimate for telecom
BASE_RETENTION_UPLIFT = 0.15

# Probability a targeted customer actually accepts the offer
# Not every at-risk customer accepts — 15% is conservative and realistic
# This is the key variable that prevents unrealistically high ROI
OFFER_ACCEPTANCE_RATE = 0.15

# -------------------------------------------------------------------------
# INTERVENTION COST
# -------------------------------------------------------------------------

# Offer = 20% discount on monthly charges
# Example: Customer pays ₹80/month → offer costs ₹80 × 0.20 = ₹16
OFFER_COST_PERCENT = 0.20

# -------------------------------------------------------------------------
# BUDGET CONSTRAINT
# -------------------------------------------------------------------------
MONTHLY_BUDGET = 5000.0

# -------------------------------------------------------------------------
# BUSINESS TARGETING FILTERS
# -------------------------------------------------------------------------

# Minimum churn probability — below 30% not worth targeting
MIN_CHURN_PROB = 0.30

# Minimum monthly charges — below ₹30 revenue saved doesn't justify cost
MIN_MONTHLY_CHARGES = 30.0

# Minimum tenure — below 3 months too new, needs onboarding not retention
MIN_TENURE_MONTHS = 3

# Segments to exclude entirely based on EDA findings
EXCLUDE_INTERNET_SERVICE = ["No"]       # 7.4% churn — not worth it
EXCLUDE_CONTRACT_TYPE    = ["Two year"] # 2.8% churn — already committed

# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"