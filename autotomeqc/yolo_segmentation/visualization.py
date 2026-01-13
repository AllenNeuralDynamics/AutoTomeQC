import cv2
import numpy as np
import os
import random
from autotomeqc.config.config_loader import TEST_OUT_DIR

# Cache for consistent class colors
CLASS_COLORS = {}

def get_color_for_class(class_id):
    if class_id not in CLASS_COLORS:
        CLASS_COLORS[class_id] = tuple(random.randint(60, 255) for _ in range(3))
    return CLASS_COLORS[class_id]


def cropped_segmented(frame, detections, filename=""):
    """
    1. Finds ALL 'section' detections (conf > 0.8).
    2. Finds the 'loop' detection.
    3. Iterates through sections:
       a. Masks specific section.
       b. Crops to loop.
       c. Saves (adds _sec{i} suffix ONLY if multiple sections exist).
    """
    if not detections:
        return

    print(f"Received {len(detections)} detections.")

    # 1. Find the 'loop' detection (Global for the frame)
    loop_detection = next((d for d in detections if d['class_name'] == 'loop'), None)

    # 2. Find ALL 'section' detections
    sections = [d for d in detections 
                if d['class_name'] == 'section' and d.get('confidence', 0.0) > 0.8]

    if not sections:
        print("No 'section' detections found with confidence > 0.8. Frame not processed.")
        return

    # Check if we have multiple sections to determine naming convention
    is_multi_section = len(sections) > 1

    # --- Iterate over every section found ---
    for i, section_det in enumerate(sections):

        process_frame = frame.copy()
        conf = section_det.get('confidence', 0.0)

        # --- A. Apply Mask for THIS Section ---
        mask_poly = section_det.get("mask", [])

        if mask_poly and len(mask_poly) > 0:
            h, w = process_frame.shape[:2]
            polygon_mask = np.zeros((h, w), dtype=np.uint8)

            polygons = []
            if isinstance(mask_poly[0][0], (list, tuple, np.ndarray)):
                for poly in mask_poly:
                    polygons.append(np.array(poly, dtype=np.int32))
            else:
                polygons.append(np.array(mask_poly, dtype=np.int32))

            cv2.fillPoly(polygon_mask, polygons, 255)
            process_frame = cv2.bitwise_and(process_frame, process_frame, mask=polygon_mask)
        else:
            print(f"Warning: Section {i} has no mask polygon. Skipping masking.")

        # --- B. Crop to 'loop' BBox ---
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

            # --- C. Resize ---
            process_frame = cv2.resize(process_frame, (640, 640))

        # --- D. Save ---
        base_filename = filename if filename else "unknown"

        # CONDITIONAL NAMING: Only add suffix if there are multiple sections
        if is_multi_section:
            save_name = f"{base_filename}_sec{i}"
            print(f"Processing Section {i+1}/{len(sections)} (Conf: {conf:.2f}) -> {save_name}")
        else:
            save_name = base_filename
            print(f"Processing Single Section (Conf: {conf:.2f}) -> {save_name}")

        output_path = os.path.join(TEST_OUT_DIR, f"{save_name}.jpg")
        cv2.imwrite(output_path, process_frame)
        print(f"Saved: {output_path}")
