# autotomeqc/utils/io.py
from datetime import datetime
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

def save_failure_report(output_dir: Path, filename: str, reason: str, ts: float) -> None:
    """
    Generates and saves a standardized JSON failure report.
    
    Args:
        output_dir (Path): The directory to save the JSON file in.
        filename (str): The base name of the file (without extension).
        reason (str): The failure reason to log.
    """
    # Generate current timestamp for the report
    ts_dt = datetime.fromtimestamp(ts)
    timestamp_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")

    output = {
        "filename": filename,
        "timestamp": timestamp_str,     # Added to match success schema
        "qc_summary": "FAIL",
        "error_reason": reason,
        "segmentation_conf": 0.0,       # Optional: defaults to 0.0 on failure
        "criteria": {}                  # Empty because no checks ran
    }

    # Construct the full path consistent with pipeline logic
    full_path = Path(output_dir) / f"{filename}_qc.json"

    # Reuse the existing save logic
    save_json_results(output, full_path)