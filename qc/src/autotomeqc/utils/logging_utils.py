import logging
import os
import sys

def setup_logging(default_level="INFO"):
    """
    Centralized logging configuration.
    Call this once at the very start of any entry point.
    """
    # Get level from env, or fallback to the provided default
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )
    
    # Return the logger for the module that called this
    return logging.getLogger("autotomeqc")