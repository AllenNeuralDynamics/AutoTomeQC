# autotomeqc/core/autotomeService.py
import logging
from pathlib import Path
import numpy as np
from concurrent.futures import Future
from typing import Optional
from autotomeqc.core.pipeline import AutoTomePipeline


class AutoTomeService:
    """
    The High-Level Controller. 
    It explicitly handles the 3 main events: Start, Stop, Process.
    """
    def __init__(self):
        self.pipeline = None
        self.running = False
        self.log = logging.getLogger(self.__class__.__name__)

    def start(self) -> bool:
        """
        Returns: True if pipeline started successfully, False if already running or failed.
        """
        if self.running:
            self.log.warning("Service is already running.")
            return False
        self.log.info(">>> exampEVENT: START")
        try:
            self.pipeline = AutoTomePipeline()
            success = self.pipeline.start()
            if success:
                self.running = True
                self.log.info("Service Initialized & Ready.")
                return True
            else:
                self.log.error("Pipeline failed to start.")
                self.pipeline = None
                return False
        except Exception as e:
            self.log.error(f"Critical error starting service: {e}")
            self.running = False
            return False

    def stop(self) -> bool:
        """
        Returns: True if stopped successfully, False if it wasn't running.
        """
        if not self.running:
            self.log.warning("Cannot stop: Service is not running.")
            return False

        self.log.info(">>> EVENT: STOP")
        try:
            if self.pipeline:
                self.pipeline.stop()
            self.running = False
            self.pipeline = None
            self.log.info("Service Shutdown Complete.")
            return True
        except Exception as e:
            self.log.error(f"Error during shutdown: {e}")
            return False

    def process(self, img_path: Optional[str] = None, frame: Optional[np.ndarray] = None) -> Future:
        """
        PROCESS IMAGE - Submits either a file path OR a raw frame to the pipeline.

        Args:
            img_path (str, optional): Path to the image file.
            frame (np.ndarray, optional): Raw image data (e.g., from cv2.imread).

        Returns:
            Future: A Future object representing the pending result.
        """
        if not self.running:
            raise RuntimeError("Service is stopped")

       # Validate Input
        if img_path is not None and frame is not None:
            raise ValueError("Ambiguous input: Provide either 'img_path' OR 'frame', not both.")
        if img_path is None and frame is None:
            raise ValueError("Missing input: Must provide either 'img_path' or 'frame'.")

        try:
            # Handle File Path
            if img_path:
                path = Path(img_path.strip('"').strip("'"))
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")
                self.log.info(f">>> EVENT: PROCESS_FILE | File: {path.name}")
                return self.pipeline.process(img_path=path)

            # Handle Numpy Frame
            elif frame is not None:
                if frame.size == 0:
                    raise ValueError("Frame is empty.")
                self.log.info(f">>> EVENT: PROCESS_FRAME | Shape: {frame.shape}")
                return self.pipeline.process(frame=frame)

        except Exception as e:
            self.log.error(f"Submission Failed: {e}")
            f = Future()
            f.set_exception(e)
            return f