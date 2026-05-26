# autotomeqc/core/autotomeService.py
import logging
import numpy as np
from concurrent.futures import Future
from typing import Optional
from autotomeqc.core.pipeline import AutoTomePipeline
from autotomeqc.config.schemas import AppConfig
from autotomeqc.config.config_loader import load_app_config


class AutoTomeService:
    """
    The High-Level Controller. 
    It explicitly handles the 3 main events: Start, Stop, Process.
    """
    def __init__(self, config: Optional[AppConfig] = None):
        # If the user doesn't provide a config, load the default
        self.config = config or load_app_config()
        self.pipeline: Optional[AutoTomePipeline] = None
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
            self.pipeline = AutoTomePipeline(config=self.config)
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
        if not self.running or self.pipeline is None:
            raise RuntimeError("Service is stopped")

        # The Service just hands it off to the Pipeline
        return self.pipeline.process(img_path=img_path, frame=frame)