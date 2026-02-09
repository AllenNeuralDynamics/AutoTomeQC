# autotomeqc/core/pipeline.py
from datetime import datetime
from pathlib import Path
import time
import logging
import cv2
import numpy as np
import uuid
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, Future
from autotomeqc.utils.io import save_json_results, save_failure_report, save_debug_image
from autotomeqc.yolo_segmentation.yolo_segmentation import YoloSegmentation
from autotomeqc.config.config_loader import CONFIG
from autotomeqc.core.models import PipelineResult, QCCriteria
from autotomeqc.yolo_segmentation.post_processing import (cropped_segmented,
                                                          get_best_section_detection,
                                                          validate_detections )
from autotomeqc.algorithms.coverage import SectionCoverageQC
from autotomeqc.algorithms.knife_mark import KnifeMarksQC
from autotomeqc.algorithms.thickness_consistency import ThicknessConsistencyQC
from autotomeqc.algorithms.thickness import ThicknessQC
from autotomeqc.algorithms.shape import ShapeQC


class AutoTomePipeline:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.output_path = Path(CONFIG.qc.output_dir)
        self.save_segmented_img = CONFIG.qc.save_segmented_images
        self.save_input_img = CONFIG.qc.save_input_images

        # Reuse threads for QC criteria
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="QC_Worker")

        # Registry to connect requests to results
        self.pending_results = {}

        # Preprocessing - Segmentation via YOLO
        self.segmenter = YoloSegmentation(
            config=CONFIG.qc.yolo,
            detection_callback=self._handle_detection
        )

        self.log.info("Initializing QC Models...")
        self.qc_modules = {
            "coverage": SectionCoverageQC(CONFIG.qc.section_coverage),
            "knife_mark": KnifeMarksQC(CONFIG.qc.knife_mark),
            "thickness_consistency": ThicknessConsistencyQC(CONFIG.qc.thickness_consistency),
            "thickness": ThicknessQC(CONFIG.qc.thickness),
            "shape": ShapeQC(CONFIG.qc.shape, output_dir=self.output_path),
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
        """Callback triggered when YOLO finishes."""
        future_ticket = self.pending_results.pop(id, None)
        timestamp_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        # 1. Validation
        is_valid, error_reason, validated_detections = validate_detections(detections)
        if not is_valid:
            self._handle_pipeline_failure(frame, filename, timestamp_str, error_reason, future_ticket)
            return

        # 2. Pre-processing for QC (Segmentation & Cropping)
        qc_input_image = cropped_segmented(frame, validated_detections)
        if qc_input_image is None:
            self._handle_pipeline_failure(frame, filename, timestamp_str, "Segmentation Failed", future_ticket)
            return

        # 3. Execution for QC Algorithms
        self._handle_pipeline_success(frame, qc_input_image, validated_detections, filename, timestamp_str, ts, future_ticket)

    def _handle_pipeline_failure(self, frame: np.ndarray, filename: str, timestamp: str, reason: str, future_ticket):
        """Standardized reporting for any rejection or failure in the pipeline."""
        self.log.warning(f"[{filename}] Pipeline Rejected: {reason}")
        result = PipelineResult(
            filename=filename,
            timestamp=timestamp,
            qc_summary="FAIL",
            error_reason=reason
        )
        output = result.model_dump(exclude_none=True)
        save_json_results(output, self.output_path / f"{filename}_qc.json")
        if self.save_input_img:
            save_debug_image(frame, self.output_path / f"{filename}_input.jpg")
        if future_ticket:
            future_ticket.set_result(output)

    def _handle_pipeline_success(self, frame: np.ndarray, qc_image: np.ndarray, detections: list, filename: str, timestamp: str, start_ts: float, future_ticket):
        """Executes QC checks and compiles final successful results."""
        # Extract metadata
        best_section = get_best_section_detection(detections)
        section_conf = round(best_section.get('confidence', 0.0), 2) if best_section else 0.0
        section_ratio = best_section.get('overlap_ratio', 0.0) if best_section else 0.0

        # Run Algorithms
        qc_results = self._run_all_checks(qc_image)
        processing_time = round(time.time() - start_ts, 4)
        final_summary = "PASS" if all(r.pass_status for r in qc_results.values()) else "FAIL"
        result = PipelineResult(
            filename=filename,
            timestamp=timestamp,
            processing_time_sec=processing_time,
            qc_summary=final_summary,
            segmentation_conf=section_conf,
            overlap_ratio=section_ratio,
            criteria=qc_results
        )
        output = result.model_dump(exclude_none=True)

        # IO Operations
        save_json_results(output, self.output_path / f"{filename}_qc.json")
        if self.save_segmented_img:
            save_debug_image(qc_image, self.output_path / f"{filename}_segmented.jpg")
        if self.save_input_img:
            save_debug_image(frame, self.output_path / f"{filename}_input.jpg")
        if future_ticket:
            future_ticket.set_result(output)

    def _run_all_checks(self, qc_image: np.ndarray) -> Dict[str, QCCriteria]:
        """Submits QC tasks and resolves them with robust error handling."""
        futures = {
            name: self.executor.submit(module.check, qc_image)
            for name, module in self.qc_modules.items()
        }
        results = {}
        for name, future in futures.items():
            try:
                raw_res = future.result(timeout=2.0)
                results[name] = QCCriteria(**raw_res)
            except Exception as e:
                self.log.error(f"QC Check {name} failed or timed out: {e}")
                # Ensure consistent field naming in fallback
                results[name] = QCCriteria(
                    pass_status=False,
                    label="Error",
                    message=str(e)
                )
        return results