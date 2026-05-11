# autotomeqc/yolo_segmentation/post_processing.py
import logging
from typing import Optional
import cv2
import numpy as np
from autotomeqc.core.models import Detection

logger = logging.getLogger(__name__)

class YoloPostProcessor:
    def __init__(self, config):
        self.config = config # This would be config.qc.yolo_post_processing

    def get_best_section_detection(self, detections: list[Detection]) -> Optional[Detection]:
        # Find ALL 'section' detections with high confidence
        valid_sections = [
            d for d in detections
            if d.class_name == 'section'
        ]
        if not valid_sections:
            return None

        # Return early if there's only one valid section
        if len(valid_sections) == 1:
            return valid_sections[0]

        # If multiple exist (e.g., debugging/global mode), pick the highest confidence
        return max(valid_sections, key=lambda x: x.confidence)

    def get_overlap_ratio(self, section_poly: list, loop_poly: list, section_bbox: list, loop_bbox: list) -> float:
        # Quick BBox Check (Cheap)
        # If the rectangles don't even touch, the ratio is definitely 0.0
        x1_s, y1_s, x2_s, y2_s = section_bbox
        x1_l, y1_l, x2_l, y2_l = loop_bbox

        if x1_s > x2_l or x2_s < x1_l or y1_s > y2_l or y2_s < y1_l:
            return 0.0

        # Precise Mask Check (Only if BBoxes overlap)
        try:
            output_dim = self.config.out_dim if self.config and self.config.out_dim else (640, 640)
            img_dim = (output_dim[0], output_dim[1])  # (w, h)
            mask_s = np.zeros((img_dim[1], img_dim[0]), dtype=np.uint8)
            mask_l = np.zeros((img_dim[1], img_dim[0]), dtype=np.uint8)

            poly_s = np.array(section_poly, dtype=np.int32).reshape((-1, 1, 2))
            poly_l = np.array(loop_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_s, [poly_s], (255,))
            cv2.fillPoly(mask_l, [poly_l], (255,))

            intersection = cv2.bitwise_and(mask_s, mask_l)
            area_s = np.sum(mask_s > 0)
            area_int = np.sum(intersection > 0)

            return float(area_int / area_s) if area_s > 0 else 0.0
        except Exception:
            return 0.0

    def validate_detections(self, detections: list[Detection]) -> tuple[bool, str, list[Detection]]:
        """
        Validates detections against AutoTomeQC logic cases (1-5).
        Returns: (is_valid, error_reason)
        """
        loop_detection = next((d for d in detections if d.class_name == 'loop'), None)
        all_sections = [d for d in detections if d.class_name == 'section']
        allow_no_loop = self.config.allow_no_loop if self.config else False

        # Case 1: No Section detected in the whole frame
        if not all_sections:
            return False, "No section detected", []

        # Case 2: No Loop logic
        if not loop_detection:
            if not allow_no_loop:
                return False, "No loop detected", []
            return True, "N/A", detections  # Proceed in Global Mode (Section only) for debugging purposes

        # --- Identify Sections relative to the Loop ---
        sections_in_loop = []
        sections_outside_loop = []
        loop_mask = loop_detection.mask if loop_detection else []
        loop_bbox = loop_detection.bbox if loop_detection else [0,0,0,0]
        for s in all_sections:
            # Check overlap ratio
            overlap = self.get_overlap_ratio(
                section_poly=s.mask,
                loop_poly=loop_mask,
                section_bbox=s.bbox,
                loop_bbox=loop_bbox
            )
            s.overlap_ratio = overlap
            thres = self.config.overlap_threshold if self.config else 0.0
            if overlap > thres:
                sections_in_loop.append(s)
            else:
                sections_outside_loop.append(s)

        # Case 3: Loop present but section is outside
        if len(sections_in_loop) == 0 and len(sections_outside_loop) > 0:
            filtered_detections = [loop_detection] + sections_outside_loop
            ratio = [round(s.overlap_ratio, 2) for s in sections_outside_loop]
            msg = f"Section detected outside loop. IoA for section(s): {ratio} (threshold: {thres})"
            return False, msg, filtered_detections

        # Case 4: Multiple Sections in Loop
        if len(sections_in_loop) > 1:
            filtered_detections = [loop_detection] + sections_in_loop
            msg = f"Multiple sections ({len(sections_in_loop)}) detected in loop"
            return True, msg, filtered_detections

        # Case 5: Success (Exactly one section in loop)
        filtered_detections = [loop_detection, sections_in_loop[0]]
        return True, "N/A", filtered_detections

    def cropped_segmented(self, frame: np.ndarray, detections: list[Detection], filename="") -> list[Detection]:
        """
        Processes each section in detections:
        1. If a loop exists, it is used as the global cropping frame.
        2. If no loop exists (and allowed), the section's own BBox is used for cropping.
        3. Masks the section, crops, resizes, and attaches to 'section_image'.
        """
        if not detections:
            return []

        margin = self.config.loop_bbox_margin if self.config else 0
        output_dim = tuple(self.config.out_dim) if self.config else (640, 640)
        # Check for a global loop context
        loop_detection = next((d for d in detections if d.class_name == 'loop'), None)

        # Iterate through all detections to find 'sections'
        for d in detections:
            if d.class_name != 'section':
                continue

            # Work on a fresh copy of the frame for each section
            temp_frame = frame.copy()
            mask_poly = d.mask
            if mask_poly and len(mask_poly) > 0:
                poly_array = np.array(mask_poly, dtype=np.int32)
                if poly_array.ndim == 2:
                    poly_array = poly_array.reshape((-1, 1, 2))
                d.area_in_pixels = int(cv2.contourArea(poly_array))
            else:
                d.area_in_pixels = 0

            # --- STEP A: Masking (Section Specific) ---
            if mask_poly and len(mask_poly) > 0:
                h, w = temp_frame.shape[:2]
                polygon_mask = np.zeros((h, w), dtype=np.uint8)
                # Convert polygon to required numpy format
                poly_array = np.array(mask_poly, dtype=np.int32)
                if poly_array.ndim == 2:
                    poly_array = poly_array.reshape((-1, 1, 2))
                # Fill the mask and apply it
                cv2.fillPoly(polygon_mask, [poly_array], color=(255,))
                temp_frame = cv2.bitwise_and(temp_frame, temp_frame, mask=polygon_mask)
            else:
                logger.warning(f"[{filename}] Section missing mask. No segmentation applied.")

            # --- STEP B: Cropping (Handling allow_no_loop) ---
            # Priority: Loop BBox > Section BBox
            target_bbox = loop_detection.bbox if loop_detection else d.bbox
            if target_bbox and len(target_bbox) == 4:
                x1, y1, x2, y2 = map(int, target_bbox)
                # Apply margins and constrain to frame boundaries
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)
                # Slice the image
                if x2 > x1 and y2 > y1:
                    temp_frame = temp_frame[y1:y2, x1:x2]

            # --- STEP C: Standardization & Injection ---
            if temp_frame.size > 0:
                # Resize to ensure the QC algorithm receives expected dimensions
                temp_frame = cv2.resize(temp_frame, output_dim)
                d.section_image = temp_frame
            else:
                d.section_image = None
                logger.error(f"[{filename}] Resulting crop for section is empty.")

        return detections