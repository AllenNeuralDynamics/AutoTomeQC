# autotomeqc/core/pipeline.py
from datetime import datetime
from pathlib import Path
import time
import logging
import cv2
import numpy as np
import queue
import threading
from typing import Dict, Optional, Any
from concurrent.futures import Future
from autotomeqc.utils.io import save_json_results, save_debug_image
from autotomeqc.config.config_loader import CONFIG
from autotomeqc.core.models import PipelineResult, QCCriteria, SectionResult
from autotomeqc.yolo_segmentation.yolo_segmentation import YoloSegmentation
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

        self.input_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False

        # Preprocessing - Segmentation via YOLO
        self.segmenter = YoloSegmentation(config=CONFIG.qc.yolo)

        self.log.info("Initializing QC Models...")
        self.qc_modules = {
            "coverage": SectionCoverageQC(CONFIG.qc.section_coverage),
            "knife_mark": KnifeMarksQC(CONFIG.qc.knife_mark),
            "thickness_consistency": ThicknessConsistencyQC(CONFIG.qc.thickness_consistency),
            "thickness": ThicknessQC(CONFIG.qc.thickness),
            "shape": ShapeQC(CONFIG.qc.shape, output_dir=self.output_path),
        }

    def start(self):
        if self.is_running:
            self.log.warning("Pipeline.start() called, but pipeline is already running.")
            return True

        self.log.info("Starting Pipeline...")
        try:
            is_ready = self.segmenter.ready.wait(timeout=60.0)  # Wait
            if not is_ready or self.segmenter.model is None:
                self.log.error("Pipeline Start Failed: YOLO Model initialization timed out.")
                return False

            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

            self.log.info("Pipeline started successfully.")
            return True
        except Exception as e:
            self.log.error(f"Critical error starting pipeline: {e}")
            return False

    def stop(self):
        self.log.info("Stopping Pipeline...")
        self.is_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None

    def process(self, img_path: Optional[str] = None, frame: Optional[np.ndarray] = None) -> Future:
        """Entry point for processing a single file."""
        filename = "Unknown"
        future_ticket: Future[Dict[str, Any]] = Future()
        ts = time.time()
        timestamp_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 1. Validate Input (XOR logic)
            if (img_path is None) == (frame is None):
                msg = "Ambiguous input: Provide either 'img_path' OR 'frame', not both/neither."
                self._handle_pipeline_failure(None, [], filename, timestamp_str, msg, future_ticket)
                return future_ticket

            # 2. Handle Image Loading/Naming
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
            else:
                ts_dt = datetime.fromtimestamp(ts)
                filename = f"{ts_dt:%Y%m%d_%H%M%S}_{ts_dt.microsecond // 1000:03d}"

            # 3. Package and Enqueue Task
            frame = self.segmenter.resize_frame(frame)
            task = {
                "frame": frame,
                "filename": filename,
                "timestamp": timestamp_str,
                "start_ts": ts,
                "future": future_ticket
            }
            self.input_queue.put(task)
            self.log.info(f"[{filename}] Task enqueued. Queue size: {self.input_queue.qsize()}")

            return future_ticket

        except Exception as e:
            self._handle_pipeline_failure(None, [], filename, timestamp_str, f"Process Error: {str(e)}", future_ticket)
            return future_ticket
    
    def _worker_loop(self):
        """The 'Heavy Lifter': Consumes tasks and runs AI + QC logic."""
        self.log.info("Worker thread active.")
        while self.is_running:
            try:
                # Retrieve task (block for 1s to allow clean shutdown check)
                try:
                    task = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                try:
                    frame = task.get("frame")
                    filename = task.get("filename", "unknown")
                    future_ticket = task.get("future")
                    timestamp = task.get("timestamp", "N/A")
                    start_ts = task.get("start_ts", time.time())
                    if frame is None or future_ticket is None:
                        raise KeyError("Task missing 'frame' or 'future' ticket.")

                    # Execute YOLO
                    detections = self.segmenter.process_frame(frame)

                    # Validate Detections
                    is_valid, error_reason, detections = validate_detections(detections)
                    if not is_valid:
                        self._handle_pipeline_failure(
                            frame, detections, filename, timestamp, error_reason, future_ticket
                        )
                    else:
                        # Pre-processing for QC (Segmentation & Cropping)
                        detections = cropped_segmented(frame, detections)
                        # Execution for QC Algorithms
                        self._handle_pipeline_valid_input(
                            frame, detections, filename, timestamp, start_ts,
                            validation_msg=error_reason,
                            future_ticket=future_ticket,
                        )
                except Exception as e:
                    self.log.error(f"Worker Error on {filename}: {e}")
                    self._handle_pipeline_failure(frame, [], filename, timestamp, str(e), future_ticket)
                finally:
                    self.input_queue.task_done()
            except Exception as e:
                self.log.error(f"Worker Loop Error: {e}")

    def _handle_pipeline_failure(
        self,
        frame: Optional[np.ndarray],
        detections: list[Dict[str, Any]],
        filename: str,
        timestamp: str,
        reason: str,
        future_ticket: Future
    ) -> None:
        """Standardized reporting for any rejection or failure in the pipeline."""
        self.log.warning(f"[{filename}] Pipeline Rejected: {reason}")
        result = PipelineResult(
            filename=filename,
            timestamp=timestamp,
            qc_summary="FAIL",
            fail_reason=reason,
            sections=[]
        )
        output = result.model_dump(exclude_none=True)
        save_json_results(output, self.output_path / f"{filename}_qc.json")
        if self.save_input_img and frame is not None:
            save_debug_image(frame, self.output_path / f"{filename}_input.jpg")
        if future_ticket:
            future_ticket.set_result(output)

    def _handle_pipeline_valid_input(
            self,
            frame: np.ndarray,
            detections: list,
            filename: str,
            timestamp: str,
            start_ts: float,
            future_ticket,
            validation_msg: str = "N/A"
        ) -> None:
        """
        Executes QC checks on all sections and compiles the final result using Pydantic models.
        """
        # Filter out valid sections
        sections = [d for d in detections if d.get('class_name') == 'section' and d.get('section_image') is not None]
        sections_list = []
        all_qc_passed = True

        # Iterate through each section and run QC
        for i, section_dict in enumerate(sections):
            target_img = section_dict['section_image']

            # Run the QC checks (Returns Dict[str, dict])
            qc_results = self._run_all_checks(target_img)

            # Determine if this specific section passed
            section_passed = all(obj.pass_status for obj in qc_results.values())
            if not section_passed:
                all_qc_passed = False

            # Instantiate SectionResult model
            sections_list.append(SectionResult(
                qc_result="PASS" if section_passed else "FAIL",
                segmentation_conf=round(section_dict.get('confidence', 0.0), 2),
                area_in_pixels=section_dict.get("area_in_pixels", 0),
                overlap_ratio=round(section_dict.get('overlap_ratio', 0.0), 2),
                criteria=qc_results
            ))

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
            sections=sections_list
        )
        output = result_obj.model_dump(exclude_none=True)

        # IO Operations
        save_json_results(output, self.output_path / f"{filename}_qc.json")
        if self.save_input_img:
            save_debug_image(frame, self.output_path / f"{filename}_input.jpg")
        if self.save_segmented_img:
            for i, section_dict in enumerate(sections):
                img_to_save = section_dict['section_image']
                save_debug_image(img_to_save, self.output_path / f"{filename}_section_{i}.jpg")
        # Resolve the Future
        if future_ticket:
            future_ticket.set_result(output)

    def _run_all_checks(self, qc_image: np.ndarray) -> Dict[str, QCCriteria]:
        results = {}
        for name, module in self.qc_modules.items():
            try:
                # Execution in the worker thread
                raw_res = module.check(qc_image)
                results[name] = QCCriteria(**raw_res)
            except Exception as e:
                self.log.error(f"QC Check {name} failed: {e}")
                results[name] = QCCriteria(pass_status=False, label="Error", message=str(e))
        return results


