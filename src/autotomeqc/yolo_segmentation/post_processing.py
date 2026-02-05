# autotome/yolo_segmentation/post_processing.py
import logging
from typing import Optional
import cv2
import numpy as np
from autotomeqc.config.config_loader import CONFIG

logger = logging.getLogger(__name__)

def get_best_section_detection(detections: list) -> Optional[dict]:
    # Find ALL 'section' detections with high confidence
    valid_sections = [
        d for d in detections
        if d['class_name'] == 'section'
    ]
    if not valid_sections:
        return None

    # Select the Best Section
    best_section = max(valid_sections, key=lambda x: x.get('confidence', 0.0))
    return best_section

def validate_detections(detections: list[dict]) -> tuple[bool, Optional[str]]:
    """
    Validates detections against AutoTomeQC logic cases (1-5).
    Returns: (is_valid, error_reason)
    """
    loop_detection = next((d for d in detections if d['class_name'] == 'loop'), None)
    all_sections = [d for d in detections if d['class_name'] == 'section']
    allow_no_loop = CONFIG["qc"].get("yolo_post_processing", {}).get("allow_no_loop", True)

    # Case 1: No Section detected in the whole frame
    if not all_sections:
        return False, "No section detected"

    # Case 2: No Loop logic
    if not loop_detection:
        if not allow_no_loop:
            return False, "No loop detected"
        return True, None  # Proceed in Global Mode (Section only) for debuging purposes

    # --- Identify Sections relative to the Loop ---
    sections_in_loop = []
    sections_outside_loop = []
    loop_mask = loop_detection.get('mask', [])
    loop_bbox = loop_detection.get('bbox', [0,0,0,0])
    for s in all_sections:
        # Check overlap ratio
        overlap = get_overlap_ratio(
            section_poly=s.get('mask', []),
            loop_poly=loop_mask,
            section_bbox=s.get('bbox', [0,0,0,0]),
            loop_bbox=loop_bbox
        )
        if overlap > 0.5:
            sections_in_loop.append(s)
        else:
            sections_outside_loop.append(s)

    # Case 3: Loop present but section is outside
    if len(sections_in_loop) == 0 and len(sections_outside_loop) > 0:
        return False, "Section detected outside loop"

    # Case 4: Multiple Sections in Loop
    if len(sections_in_loop) > 1:
        return False, f"Multiple sections ({len(sections_in_loop)}) detected in loop"

    # Case 5: Success (Exactly one section in loop)
    return True, None

def get_overlap_ratio(section_poly: list, loop_poly: list, section_bbox: list, loop_bbox: list) -> float:
    # Quick BBox Check (Cheap)
    # If the rectangles don't even touch, the ratio is definitely 0.0
    x1_s, y1_s, x2_s, y2_s = section_bbox
    x1_l, y1_l, x2_l, y2_l = loop_bbox
    
    if x1_s > x2_l or x2_s < x1_l or y1_s > y2_l or y2_s < y1_l:
        return 0.0

    # Precise Mask Check (Only if BBoxes overlap)
    try:
        output_dim = CONFIG["qc"].get("yolo_post_processing", {}).get("out_dim", [640, 640])
        img_dim = (output_dim[0], output_dim[1])  # (w, h)
        mask_s = np.zeros((img_dim[1], img_dim[0]), dtype=np.uint8)
        mask_l = np.zeros((img_dim[1], img_dim[0]), dtype=np.uint8)
        
        cv2.fillPoly(mask_s, [np.array(section_poly, dtype=np.int32)], 255)
        cv2.fillPoly(mask_l, [np.array(loop_poly, dtype=np.int32)], 255)
        
        intersection = cv2.bitwise_and(mask_s, mask_l)
        area_s = np.sum(mask_s > 0)
        area_int = np.sum(intersection > 0)
        
        return float(area_int / area_s) if area_s > 0 else 0.0
    except Exception:
        return 0.0
    
def cropped_segmented(frame: np.ndarray, detections: list, filename="") -> Optional[np.ndarray]:
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

    best_section = get_best_section_detection(detections)
    if best_section is None:
        logger.warning(f"[{filename}] No valid 'section' found. Skipping.")
        return None

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
        cv2.fillPoly(polygon_mask, polygons, color=255)  # type: ignore

        # Apply mask: Keep only the section, black out everything else
        process_frame = cv2.bitwise_and(process_frame, process_frame, mask=polygon_mask)
    else:
        logger.warning(f"[{filename}] Best section has no mask polygon. Skipping masking.")

    # Crop to 'loop' BBox ---
    if loop_detection:
        bbox = loop_detection.get('bbox', [])
        margin = CONFIG["qc"].get("yolo_post_processing", {}).get("loop_bbox_margin", 30)

        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(process_frame.shape[1], x2 + margin)
            y2 = min(process_frame.shape[0], y2 + margin)

            if x2 > x1 and y2 > y1:
                process_frame = process_frame[y1:y2, x1:x2]

    # Resize (Standardize input for QC models)
    output_dim = CONFIG["qc"].get("yolo_post_processing", {}).get("out_dim", [640, 640])
    process_frame = cv2.resize(process_frame, tuple(output_dim))
    return process_frame