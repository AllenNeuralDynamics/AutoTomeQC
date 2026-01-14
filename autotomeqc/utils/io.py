import json
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Initialize the logger for this module
logger = logging.getLogger(__name__)

def save_json_results(data: Dict[str, Any], path: Path) -> None:
    """Saves a dictionary to a JSON file, creating parent directories if needed."""
    try:
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"QC metrics saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise

def save_debug_image(image: Optional[np.ndarray], path: Path) -> None:
    """Saves an image to disk, creating parent directories if needed."""
    if image is None:
        logger.warning("No image data provided to save_debug_image.")
        return
    try:
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        logger.info(f"Debug image saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to save image to {path}: {e}")