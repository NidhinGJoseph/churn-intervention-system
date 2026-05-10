# =============================================================================
# config.py (FINAL - CONSISTENT + PRODUCTION READY)
# =============================================================================

from pathlib import Path

# -------------------------------------------------------------------------
# ROOT
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

MODEL_PATH        = MODELS_DIR / "churn_model.pkl"
INTERVENTION_PATH = OUTPUTS_DIR / "intervention_list.csv"

# -------------------------------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------------------------------
TARGET_COLUMN   = "Churn"
COLUMNS_TO_DROP = ["customerID"]

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# -------------------------------------------------------------------------
# BUSINESS RULES (FILTERING)
# -------------------------------------------------------------------------
MIN_CHURN_PROB      = 0.30
MIN_MONTHLY_CHARGES = 30.0
MIN_TENURE_MONTHS   = 3

EXCLUDE_INTERNET_SERVICE = ["No"]
EXCLUDE_CONTRACT_TYPE    = ["Two year"]

# -------------------------------------------------------------------------
# ECONOMICS (CRITICAL)
# -------------------------------------------------------------------------

# Cost = % of MonthlyCharges (dynamic per customer)
OFFER_COST_PERCENT = 0.20

# Probability intervention works
RETENTION_UPLIFT = 0.30

# Max lifecycle assumption
MAX_TENURE = 72

# Budget constraint
MONTHLY_BUDGET = 5000.0

# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"