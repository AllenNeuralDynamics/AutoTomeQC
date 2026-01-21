# autotomeqc/core/pipeline.py
from pathlib import Path
import time
import logging
from typing import Dict, Any
import cv2
import numpy as np
from autotomeqc.utils.io import save_json_results, save_debug_image
from concurrent.futures import ThreadPoolExecutor, as_completed
from autotomeqc.yolo_segmentation.yolo_client import YOLOClient
from autotomeqc.config.config_loader import CONFIG
from autotomeqc.yolo_segmentation.visualization import cropped_segmented

from autotomeqc.algorithms.coverage import SectionCoverageQC
from autotomeqc.algorithms.knife_mark import KnifeMarksQC
from autotomeqc.algorithms.thickness_consistency import ThicknessConsistencyQC
from autotomeqc.algorithms.thickness import ThicknessQC


logger = logging.getLogger(__name__)


class AutoTomePipeline:
    def __init__(self):
        self.output_path = Path(CONFIG["qc"]["output_dir"])
        self.save_segmented_img = CONFIG["qc"].get("save_segmented_images", True)

        # Reuse threads for QC criteria
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="QC_Worker")

        # Preprocessing - Segmentation via YOLO
        self.yolo = YOLOClient(
            config=CONFIG["qc"],
            detection_callback=self._handle_detection
        )

        logger.info("Initializing QC Models...")
        self.qc_modules = {
            "coverage": SectionCoverageQC(CONFIG["qc"]),
            "knife_mark": KnifeMarksQC(CONFIG["qc"]),
            "thickness_consistency": ThicknessConsistencyQC(CONFIG["qc"]),
            "thickness": ThicknessQC(CONFIG["qc"]),
        }

    def start(self):
        logger.info("Starting Pipeline...")
        self.yolo.start_client()

    def stop(self):
        logger.info("Stopping Pipeline...")
        self.yolo.stop()
        self.executor.shutdown(wait=False)  # Cleanup threads

    def process_image(self, file_path):
        """Entry point for processing a single file."""
        frame = cv2.imread(str(file_path))
        if frame is None:
            logger.error(f"Failed to load {file_path}")
            return

        logger.info(f"Processing: {str(file_path)}")
        # Pass the filename so we can use it in the JSON output later
        self.yolo.newframe_captured(frame, current=time.time(), filename=file_path.stem)

    def _handle_detection(self, frame, detections, filename):
        """
        Callback triggered when YOLO finishes.
        Args:
            frame: The original image (numpy array).
            detections: The YOLO result object.
            filename: The name of the file (string).
        """
        timestamp = time.time()

        # Get the cropped/segmented image
        qc_input_image = cropped_segmented(frame, detections)
        if qc_input_image is None:
            logger.warning(f"No segmentation found for {filename}, skipping QC.")
            return

        # Run QC Checks in Parallel
        qc_results = self._run_all_checks(qc_input_image)

        # Compile Final JSON
        final_summary = "PASS" if all(r["pass"] for r in qc_results.values()) else "FAIL"
        final_output = {
            "filename": filename,
            "timestamp": timestamp,
            "qc_summary": final_summary,
            "criteria": qc_results
        }

        # Output Logic
        json_filename = self.output_path / f"{filename}_qc.json"
        save_json_results(final_output, json_filename)
        if self.save_segmented_img:  # Save Image (Optional)
            # Construct the full path for the image file
            img_filename = self.output_path / f"{filename}_segmented.jpg"
            save_debug_image(qc_input_image, img_filename)

    def _run_all_checks(self, image):
        """Runs defined QC modules in parallel + geometry check."""
        results = {}
        futures = {}

        # Submit QC checks (Coverage, etc.)
        for name, module in self.qc_modules.items():
            futures[self.executor.submit(module.check, image)] = name

        # Collect results
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result(timeout=2.0) # 2s timeout per check
            except Exception as e:
                logger.error(f"QC Check {name} failed: {e}")
                results[name] = {"pass": False, "error": str(e)}

        return results