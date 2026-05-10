# =============================================================================
# utils.py
# -----------------------------------------------------------------------------
# PURPOSE:
#   Sets up logging for the project so we can track what's happening
#   instead of using print() everywhere.
# =============================================================================

import logging
from src.config import LOG_LEVEL, LOG_FORMAT

def get_logger(name: str) -> logging.Logger:
    """
    Creates a logger for any file that needs one.

    HOW TO USE IN OTHER FILES:
        from src.utils import get_logger
        logger = get_logger(__name__)
        logger.info("Data loaded successfully")

    Args:
        name: pass __name__ from the calling file
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL))

    return logger