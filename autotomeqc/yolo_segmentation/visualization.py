import logging
import cv2
import numpy as np
import os
import random
from autotomeqc.config.config_loader import TEST_OUT_DIR

def cropped_segmented(frame: np.ndarray, detections: dict, filename="") -> np.ndarray:
    """
    Processing logic:
    1. Finds the 'loop' detection (global context).
    2. Finds the 'best' section (highest confidence > 0.8).
    3. Masks that specific section (blacking out background).
    4. Crops to the 'loop' bounding box.
    5. Returns the processed image for QC.
    """
    if not detections:
        return None

    # Find the 'loop' detection (Global for the frame)
    loop_detection = next((d for d in detections if d['class_name'] == 'loop'), None)

    # Find ALL 'section' detections with high confidence
    valid_sections = [
        d for d in detections
        if d['class_name'] == 'section' and d.get('confidence', 0.0) > 0.8
    ]

    if not valid_sections:
        logging.warning(f"[{filename}] No valid 'section' found (conf > 0.8). Skipping.")
        return None

    # Select the Best Section
    # Since the pipeline currently handles one image at a time, we pick the
    # section with the highest confidence score to be the "Main" section.
    best_section = max(valid_sections, key=lambda x: x.get('confidence', 0.0))
    
    # Processing
    process_frame = frame.copy()
    mask_poly = best_section.get("mask", [])

    if mask_poly and len(mask_poly) > 0:
        h, w = process_frame.shape[:2]
        polygon_mask = np.zeros((h, w), dtype=np.uint8)

        polygons = []
        # Handle different polygon nesting formats
        if isinstance(mask_poly[0][0], (list, tuple, np.ndarray)):
            for poly in mask_poly:
                polygons.append(np.array(poly, dtype=np.int32))
        else:
            polygons.append(np.array(mask_poly, dtype=np.int32))

        # Fill the polygon on the mask
        cv2.fillPoly(polygon_mask, polygons, 255)

        # Apply mask: Keep only the section, black out everything else
        process_frame = cv2.bitwise_and(process_frame, process_frame, mask=polygon_mask)
    else:
        logging.warning(f"[{filename}] Best section has no mask polygon. Skipping masking.")

    # Crop to 'loop' BBox ---
    # We crop to the loop because the section is inside the loop
    if loop_detection:
        bbox = loop_detection.get('bbox', [])
        margin = loop_detection.get('loop_bbox_margin', 30)

        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(process_frame.shape[1], x2 + margin)
            y2 = min(process_frame.shape[0], y2 + margin)

            if x2 > x1 and y2 > y1:
                process_frame = process_frame[y1:y2, x1:x2]

    # Resize (Standardize input for QC models) ---
    # TODO move to config
    process_frame = cv2.resize(process_frame, (640, 640))

    return process_frame