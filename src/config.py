# =============================================================================
# config.py (CLEAN + CONSISTENT)
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
# BUSINESS LOGIC (FOR OPTIMIZER)
# -------------------------------------------------------------------------
MIN_CHURN_PROB      = 0.30
MIN_MONTHLY_CHARGES = 30.0
MIN_TENURE_MONTHS   = 3

OFFER_COST_PERCENT = 0.20
RETENTION_UPLIFT   = 0.30
MAX_TENURE         = 72
MONTHLY_BUDGET     = 5000.0

EXCLUDE_INTERNET_SERVICE = ["No"]
EXCLUDE_CONTRACT_TYPE    = ["Two year"]

# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"