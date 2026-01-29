# autotomeqc/yolo_segmentation/yolo_client.py
import numpy as np
import logging
from autotomeqc.yolo_segmentation.yolo_server import YoloSegmentation


class YOLOClient:
    def __init__(self, config={}, detection_callback=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.fps = config.get('fps', 5)
        self.current_time = None
        yolo_config = config.get('yolo', {})
        self.yolo_worker = YoloSegmentation(yolo_config, detection_callback=detection_callback)

    def start_client(self):
        """Start the YOLO processing worker"""
        try:
            self.yolo_worker.start()
            self.log.info("Simple YOLO client started successfully")
            return True
        except Exception as e:
            self.log.error(f"Error starting Simple YOLO client: {e}")
            return False
        
    def newframe_captured(self, frame: np.ndarray, id: str, filename: str = "", ts: float = 0.0):
        """Put new frame at the specified FPS rate"""
        # Rate limit the frames sent to the YOLO worker
        #if self.current_time is None or current - self.current_time > (1/self.fps):
        self.yolo_worker.process_frame(frame, id=id, filename=filename, ts=ts)
            
    def stop(self):
        """Stop the YOLO worker"""
        if self.yolo_worker:
            self.yolo_worker.stop()
            self.log.info("YOLO client stopped.")