# autotomeqc/core/autotomeService.py
import logging
from pathlib import Path
from concurrent.futures import Future
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
        self.log.info(">>> EVENT: START")
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

    def process(self, input_image_path: str) -> Future:
        """
        PROCESS IMAGE - Submits the image to the pipeline.

        Returns:
            Future: A Future object representing the pending result.
        """
        if not self.running:
            raise RuntimeError("Service is stopped")

        input_path = Path(input_image_path.strip('"').strip("'"))
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        self.log.info(f">>> EVENT: PROCESS_IMAGE | File: {input_path.name}")        
        try:
            return self.pipeline.process_image(input_path)
        except Exception as e:
            self.log.error(f"Submission Failed: {e}")
            # If submission fails (rare), return a failed Future so the client doesn't hang
            f = Future()
            f.set_exception(e)
            return f