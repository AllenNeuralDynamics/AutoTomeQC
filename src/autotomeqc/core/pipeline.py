# autotomeqc/core/pipeline.py
from datetime import datetime
from pathlib import Path
import time
import logging
import cv2
import numpy as np
import uuid
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from autotomeqc.utils.io import save_json_results, save_failure_report, save_debug_image
from autotomeqc.yolo_segmentation.yolo_segmentation import YoloSegmentation
from autotomeqc.config.config_loader import CONFIG
from autotomeqc.yolo_segmentation.visualization import cropped_segmented, get_best_section_detection

from autotomeqc.algorithms.coverage import SectionCoverageQC
from autotomeqc.algorithms.knife_mark import KnifeMarksQC
from autotomeqc.algorithms.thickness_consistency import ThicknessConsistencyQC
from autotomeqc.algorithms.thickness import ThicknessQC
from autotomeqc.algorithms.shape import ShapeQC


class AutoTomePipeline:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.output_path = Path(CONFIG["qc"]["output_dir"])
        self.save_segmented_img = CONFIG["qc"].get("save_segmented_images", True)
        self.save_input_img = CONFIG["qc"].get("save_input_images", False)

        # Reuse threads for QC criteria
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="QC_Worker")

        # Registry to connect requests to results
        self.pending_results = {}

        # Preprocessing - Segmentation via YOLO
        self.segmenter = YoloSegmentation(
            config=CONFIG["qc"].get('yolo', {}),
            detection_callback=self._handle_detection
        )

        self.log.info("Initializing QC Models...")
        self.qc_modules = {
            "coverage": SectionCoverageQC(CONFIG["qc"]),
            "knife_mark": KnifeMarksQC(CONFIG["qc"]),
            "thickness_consistency": ThicknessConsistencyQC(CONFIG["qc"]),
            "thickness": ThicknessQC(CONFIG["qc"]),
            "shape": ShapeQC(CONFIG["qc"]),
        }

    def start(self):
        self.log.info("Starting Pipeline...")
        try:
            is_ready = self.segmenter.ready.wait(timeout=60.0)  # Wait
            if not is_ready:
                self.log.error("Pipeline Start Failed: YOLO Model initialization timed out.")
                return False
            # Check if the model actually loaded correctly
            if self.segmenter.model is None:
                 self.log.error("Pipeline Start Failed: YOLO Model is None (Load failed).")
                 return False
            # Start the Segmenter Thread
            if self.segmenter.start():
                self.log.info("Pipeline started successfully.")
                return True
            else:
                self.log.error("YOLO Segmentation refused to start.")
                return False

        except Exception as e:
            self.log.error(f"Critical error starting pipeline: {e}")
            return False

    def stop(self):
        self.log.info("Stopping Pipeline...")
        if self.segmenter:
            self.segmenter.stop()
        self.executor.shutdown(wait=False)  # Cleanup threads

    def process(self, img_path: Optional[str] = None, frame: Optional[np.ndarray] = None) -> Future:
        """Entry point for processing a single file."""
        future_ticket: Future[Dict[str, Any]] = Future()
        ts = time.time()

        # Validate Input (XOR logic)
        if (img_path is None) == (frame is None):
            msg = "Ambiguous input: Provide either 'img_path' OR 'frame', not both/neither."
            self.log.error(msg)
            return self._fail_report(future_ticket, "Unknown", msg, ts)

        # Setup Identifiers
        request_id = str(uuid.uuid4())
        filename = ""

        # Handle Image Loading
        if img_path is not None:
            path_obj = Path(img_path)
            filename = path_obj.stem
            frame = cv2.imread(str(img_path))
            if frame is None:
                msg = f"Failed to load image from path: {img_path}"
                self.log.error(msg)
                return self._fail_report(future_ticket, filename, msg, ts)

        elif frame is not None:
            # Create a filename if one wasn't provided
            ts_dt = datetime.fromtimestamp(ts)
            filename = f"{ts_dt:%Y%m%d_%H%M%S}_{ts_dt.microsecond // 1000:03d}"

        # Register Ticket
        self.pending_results[request_id] = future_ticket

        # Dispatch to YOLO
        # Passing 'ts' (float) and 'request_id' to be returned in callback
        self.segmenter.process_frame(frame, id=request_id, filename=filename, ts=ts)

        return future_ticket

    def _fail_report(self, ticket: Future, filename: str, reason: str, ts: float) -> Future:
        """Generates a failure report and returns a failed Future."""
        # Ensure output path exists or handle it inside save_failure_report
        save_failure_report(self.output_path, filename, reason, ts)
        ticket.set_exception(ValueError(f"{reason}: {filename}"))
        return ticket

    def _handle_detection(self, frame: np.ndarray, detections: list[Dict[str, Any]], filename: str, id: str, ts: float):
        """
        Callback triggered when YOLO finishes.
        Args:
            frame: The original image (numpy array).
            detections: The YOLO result object.
            filename: The name of the file (string).
            id: The UUID request_id.
            ts: The timestamp (float).
        """
        # Retrieve the waiting ticket
        future_ticket = self.pending_results.pop(id, None)

        # Convert float timestamp to datetime object for formatting
        ts_dt = datetime.fromtimestamp(ts)
        timestamp_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Get the cropped/segmented image
        qc_input_image = cropped_segmented(frame, detections)
        best_section = get_best_section_detection(detections)
        get_section_conf = round(best_section.get('confidence', 0.0), 2) if best_section else 0.0

        if qc_input_image is None:
            self.log.warning(f"No segmentation found for {filename}, skipping QC.")
            # Construct failure output
            if future_ticket:
                output = {
                    "filename": filename,
                    "timestamp": timestamp_str, # Use the formatted string
                    "qc_summary": "FAIL",
                    "error_reason": "Segmentation Failed: No section detected",
                    "segmentation_conf": 0.0,
                    "criteria": {}
                }
                # Create a failure record on disk as well
                save_json_results(output, self.output_path / f"{filename}_qc.json")
                if self.save_input_img:
                    input_img_filename = self.output_path / f"{filename}_input.jpg"
                    save_debug_image(frame, input_img_filename)
                future_ticket.set_result(output)
            return

        # Run QC Checks in Parallel
        qc_results = self._run_all_checks(qc_input_image)

        # Measure processing time in seconds since the frame timestamp
        processing_time = time.time() - ts

        # Compile Final JSON
        final_summary = "PASS" if all(r.get("pass", False) for r in qc_results.values()) else "FAIL"
        final_output = {
            "filename": filename,
            "timestamp": timestamp_str,
            "processing_time_sec": round(processing_time, 4),
            "qc_summary": final_summary,
            "segmentation_conf": get_section_conf,
            "criteria": qc_results,
        }


        # Output Logic
        json_filename = self.output_path / f"{filename}_qc.json"
        save_json_results(final_output, json_filename)
        if self.save_segmented_img:
            img_filename = self.output_path / f"{filename}_segmented.jpg"
            save_debug_image(qc_input_image, img_filename)
        if self.save_input_img:
            input_img_filename = self.output_path / f"{filename}_input.jpg"
            save_debug_image(frame, input_img_filename)

        # Deliver Result
        if future_ticket:
            future_ticket.set_result(final_output)

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
                self.log.error(f"QC Check {name} failed: {e}")
                results[name] = {"pass": False, "error": str(e)}

        return results