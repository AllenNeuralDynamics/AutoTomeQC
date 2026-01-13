import time
import logging
import json
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from autotomeqc.yolo_segmentation.yolo_client import YOLOClient
from autotomeqc.config.config_loader import CONFIG, TEST_OUT_DIR
from autotomeqc.yolo_segmentation.visualization import cropped_segmented

logger = logging.getLogger(__name__)

class AutoTomePipeline:
    def __init__(self):
        # Preprocessing - Segmentation via YOLO
        self.yolo = YOLOClient(
            config=CONFIG["qc"], 
            detection_callback=self._handle_detection_and_qc
        )

        # Run QC criteria in parallel
        self.qc_criteria = [
            self.check_shape,
            self.check_color,
            self.check_cracks,
            self.check_wrinkles,
            self.check_bubbles,
            self.check_dimensions
        ]

    def start(self):
        logger.info("Starting Pipeline...")
        self.yolo.start_client()

    def stop(self):
        logger.info("Stopping Pipeline...")
        self.yolo.stop()

    def process_image(self, file_path):
        """Entry point for processing a single file."""
        frame = cv2.imread(str(file_path))
        if frame is None:
            logger.error(f"Failed to load {file_path}")
            return

        logger.info(f"Processing: {file_path.name}")
        # Pass the filename so we can use it in the JSON output later
        self.yolo.newframe_captured(frame, current=time.time(), filename=file_path.stem)

    def _handle_detection_and_qc(self, frame, detections, filename):
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

        # Save the segmented check image (Optional)
        output_path = TEST_OUT_DIR / f"{filename}_segmented.jpg"
        cv2.imwrite(str(output_path), qc_input_image)

        # Run QC Checks in Parallel
        qc_results = self._run_parallel_qc(qc_input_image)
        
        # Compile Final JSON
        final_output = {
            "filename": filename,
            "timestamp": timestamp,
            "qc_summary": "PASS" if all(r["pass"] for r in qc_results.values()) else "FAIL",
            "criteria": qc_results
        }

        # Output Logic
        self._save_json(final_output)

    def _run_parallel_qc(self, image):
        """Runs all 6 criteria concurrently and waits for all to finish."""
        results = {}
        
        # ThreadPoolExecutor manages the threads automatically
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Map each function to a future
            future_to_criteria = {
                executor.submit(func, image): func.__name__ 
                for func in self.qc_criteria
            }

            # Wait for all to complete
            for future in as_completed(future_to_criteria):
                criteria_name = future_to_criteria[future]
                try:
                    data = future.result()
                    results[criteria_name] = data
                except Exception as e:
                    logger.error(f"{criteria_name} generated an exception: {e}")
                    results[criteria_name] = {"error": str(e), "pass": False}
        
        return results

    def _save_json(self, data):
        """Helper to save the result."""
        print(json.dumps(data, indent=2))
        # TODO : Save or return file

    # --- QC criteria Functions Examples ---, # TODO replace with actual implementations
    def check_shape(self, img):
        return {"status": "Diamond", "vertices": 4, "pass": True}

    def check_color(self, img):
        time.sleep(0.1) # Simulate work
        return {"value": "Gold", "pass": True}
        
    def check_cracks(self, img): return {"count": 0, "pass": True}
    def check_wrinkles(self, img): return {"score": 0.0, "pass": True}
    def check_bubbles(self, img): return {"count": 0, "pass": True}
    def check_dimensions(self, img): return {"area": 1500, "pass": True}