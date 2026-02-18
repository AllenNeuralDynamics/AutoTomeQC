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
from autotomeqc.utils.io import save_json_results, save_debug_image
from autotomeqc.yolo_segmentation.yolo_segmentation import YoloSegmentation
from autotomeqc.config.config_loader import CONFIG
from autotomeqc.core.models import PipelineResult, QCCriteria, SectionResult
from autotomeqc.yolo_segmentation.post_processing import cropped_segmented, validate_detections
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
        timestamp_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Validate Input (XOR logic)
            if (img_path is None) == (frame is None):
                msg = "Ambiguous input: Provide either 'img_path' OR 'frame', not both/neither."
                self.log.error(msg)
                self._handle_pipeline_failure(None, [], "Unknown", timestamp_str, msg, future_ticket)
                return future_ticket

            # Setup Identifiers
            request_id = str(uuid.uuid4())

            # Handle Image Loading
            if img_path is not None:
                path_obj = Path(img_path)
                filename = path_obj.stem
                if not path_obj.exists():
                    self._handle_pipeline_failure(None, [], filename, timestamp_str, f"File not found: {img_path}", future_ticket)
                    return future_ticket
                frame = cv2.imread(str(img_path))
                if frame is None:
                    self._handle_pipeline_failure(None, [], filename, timestamp_str, f"File load failed: {img_path}", future_ticket)
                    return future_ticket
            elif frame is not None:
                ts_dt = datetime.fromtimestamp(ts)
                filename = f"{ts_dt:%Y%m%d_%H%M%S}_{ts_dt.microsecond // 1000:03d}"

            # Register Ticket and Dispatch
            self.pending_results[request_id] = future_ticket
            self.segmenter.process_frame(frame, id=request_id, filename=filename, ts=ts)
            return future_ticket

        except Exception as e:
            # 3. Handle any unexpected errors (Corrupt files, invalid types, etc.)
            self._handle_pipeline_failure(
                frame=None, detections=[], filename="Error",
                timestamp=timestamp_str, reason=str(e),
                future_ticket=future_ticket
            )
            return future_ticket

    def _handle_detection(self, frame: np.ndarray, detections: list[Dict[str, Any]], filename: str, id: str, ts: float):
        """Callback triggered when YOLO finishes."""
        future_ticket = self.pending_results.pop(id, None)
        timestamp_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        # Validation
        is_valid, error_reason, detections = validate_detections(detections)
        if not is_valid:
            self._handle_pipeline_failure(frame, detections, filename, timestamp_str, error_reason, future_ticket)
            return

        # Pre-processing for QC (Segmentation & Cropping)
        detections = cropped_segmented(frame, detections)

        # Execution for QC Algorithms
        self._handle_pipeline_valid_input(frame, detections, filename, timestamp_str, ts, future_ticket, validation_msg=error_reason)

    def _handle_pipeline_failure(self, frame: np.ndarray, detections: list[Dict[str, Any]], filename: str, timestamp: str, reason: str, future_ticket):
        """Standardized reporting for any rejection or failure in the pipeline."""
        self.log.warning(f"[{filename}] Pipeline Rejected: {reason}")
        highest_ratio = round(max((d.get('overlap_ratio', 0.0) for d in detections), default=0.0), 2)
        result = PipelineResult(
            filename=filename,
            timestamp=timestamp,
            qc_summary="FAIL",
            fail_reason=reason,
            overlap_ratio=highest_ratio,
            sections={}
        )
        output = result.model_dump(exclude_none=True)
        save_json_results(output, self.output_path / f"{filename}_qc.json")
        if self.save_input_img and frame is not None:
            save_debug_image(frame, self.output_path / f"{filename}_input.jpg")
        if future_ticket:
            future_ticket.set_result(output)

    def _handle_pipeline_valid_input(self, frame: np.ndarray, detections: list, filename: str, timestamp: str, start_ts: float, future_ticket, validation_msg: str = "N/A"):
        """
        Executes QC checks on all sections and compiles the final result using Pydantic models.
        """
        # Filter out valid sections
        sections = [d for d in detections if d.get('class_name') == 'section' and d.get('section_image') is not None]
        sections_map = {}
        all_qc_passed = True

        # Iterate through each section and run QC
        for i, section_dict in enumerate(sections):
            section_id = str(i)
            target_img = section_dict['section_image']

            # Run the parallelized QC checks (Returns Dict[str, dict])
            qc_results = self._run_all_checks(target_img)

            # Determine if this specific section passed
            section_passed = all(obj.pass_status for obj in qc_results.values())
            if not section_passed:
                all_qc_passed = False

            # Instantiate SectionResult model
            sections_map[section_id] = SectionResult(
                qc_result="PASS" if section_passed else "FAIL",
                segmentation_conf=round(section_dict.get('confidence', 0.0), 2),
                area_in_pixels=section_dict.get("area_in_pixels", 0),
                overlap_ratio=round(section_dict.get('overlap_ratio', 0.0), 2),
                criteria=qc_results
            )

        # Final global summary report Logic
        multiple_detected = len(sections) > 1
        processing_time = round(time.time() - start_ts, 4)
        global_pass = all_qc_passed and not multiple_detected
        if multiple_detected:
            current_fail_reason = validation_msg
        elif not all_qc_passed:
            current_fail_reason = "Section failed QC criteria"
        else:
            current_fail_reason = "N/A"

        # Construct Final PipelineResult Model
        result_obj = PipelineResult(
            filename=filename,
            timestamp=timestamp,
            qc_summary="PASS" if global_pass else "FAIL",
            fail_reason=current_fail_reason,
            processing_time_sec=processing_time,
            sections=sections_map
        )
        output = result_obj.model_dump(exclude_none=True)

        # IO Operations
        save_json_results(output, self.output_path / f"{filename}_qc.json")
        if self.save_input_img:
            save_debug_image(frame, self.output_path / f"{filename}_input.jpg")
        if self.save_segmented_img:
            for sid, section_res in sections_map.items():
                img_to_save = sections[int(sid)]['section_image']
                save_debug_image(img_to_save, self.output_path / f"{filename}_section_{sid}.jpg")
        # Resolve the Future
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


