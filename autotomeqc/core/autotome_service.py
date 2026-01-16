# autotomeqc/core/autotomeService.py
import logging
from pathlib import Path

from autotomeqc.core.pipeline import AutoTomePipeline

logger = logging.getLogger("AutoTomeService")

class AutoTomeService:
    """
    The High-Level Controller. 
    It explicitly handles the 3 main events: Start, Stop, Process.
    """
    def __init__(self):
        self.pipeline = None
        self.running = False

    def start(self):
        """Event 1: START - Initialize the pipeline resources."""
        if self.running:
            logger.warning("Service is already running.")
            return

        logger.info(">>> EVENT: START")
        self.pipeline = AutoTomePipeline()
        self.pipeline.start()  # Start YOLO client
        self.running = True
        logger.info("Service Initialized & Ready.")

    def stop(self):
        """Event 2: STOP - Clean shutdown of resources."""
        if not self.running:
            return
            
        logger.info(">>> EVENT: STOP")
        if self.pipeline:
            self.pipeline.stop()
        self.running = False
        logger.info("Service Shutdown Complete.")

    def process(self, input_image_path: str):
        """Event 3: PROCESS IMAGE - Run logic on a specific file."""
        if not self.running:
            logger.error("Cannot process: Service is stopped. Type 'start' first.")
            return

        input_path = Path(input_image_path.strip('"').strip("'"))
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return

        logger.info(f">>> EVENT: PROCESS_IMAGE | File: {input_path.name}")        
        try:
            # Non-blocking call
            self.pipeline.process_image(input_path)

        except Exception as e:
            logger.error(f"Processing Failed: {e}")