# autotomeqc/yolo_segmentation/yolo_segmentation.py
from collections.abc import Callable
from typing import Deque, Optional, Tuple
import numpy as np
import time
import logging
from threading import Event
from collections import deque
import cv2
from threading import Thread # Using Event for better thread signaling
from autotomeqc.config.schemas import YoloSettings
from ultralytics import YOLO
import torch


class YoloSegmentation:
    """YOLO segmentation worker that runs in its own thread"""
    
    def __init__(self, config: YoloSettings, detection_callback: Optional[Callable] = None):
        """
        :param config: Configuration dictionary.
        :param detection_callback: A function to call with the list of detections.
        """
        self.weights_path = config.weights_path
        self.conf_thresh = config.conf_thresh
        self.img_size = config.img_size
        self.img_dim = config.img_dim
        self.max_det = config.max_det
        self.log = logging.getLogger(self.__class__.__name__)

        # State
        self.model = None
        self.frame_queue: Deque[Tuple[np.ndarray, Optional[str], str, float]] = deque(maxlen=None)
        self.running = False
        self.worker_thread = None
        self.detection_callback = detection_callback
        self.ready: Event = Event()

        self._load_model()

    def _load_model(self):
        """Encapsulate model loading logic"""
        try:
            self.model = YOLO(self.weights_path)
            self.log.info(f"YOLO model loaded from: {self.weights_path}")

            # Check device availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            self.log.info(f"Model is running on: {device}")
            self._warmup_model()
            self.ready.set()
        except Exception as e:
            self.log.error(f"Failed to load YOLO model: {e}. Running in DUMMY mode.")
            self.model = None

    def start(self):
        """Start the YOLO segmentation thread"""
        if self.running:
            self.log.warning("YOLO segmentation thread already running")
            return True
        self.running = True
        self.worker_thread = Thread(target=self._process_frames, daemon=True)
        self.worker_thread.start()
        self.log.info("YOLO segmentation thread started")
        return True
    
    def _warmup_model(self):
        """Warm up the model with dummy inference to avoid first-frame delay"""
        if self.model is None:
            return

        self.log.info("Warming up YOLO model...")
        warmup_start = time.time()

        try:
            # Create dummy frame matching your expected input
            dummy_frame = np.random.randint(0, 255, (self.img_dim[1], self.img_dim[0], 3), dtype=np.uint8)

            # Run several warmup inferences
            for i in range(3):
                # Using predict for simple warmup instead of track if tracking is not essential here
                _ = self.model.track(dummy_frame, persist=False)

            # Additional GPU warmup if using CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                
            warmup_time = time.time() - warmup_start
            self.log.info(f"Model warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            self.log.error(f"Warmup failed: {e}")
        
    def stop(self):
        """Stop the YOLO processing thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        self.log.info("YOLO segmentation worker stopped")
        
    def process_frame(self, frame: np.ndarray, id: str | None = None, filename: str = "", ts: float = 0.0):
        """Add frame to processing queue"""
        if not self.running:
            return

        try:
            frame = self._resize_frame(frame)
            self.frame_queue.append((frame, id, filename, ts))
        except Exception as e:
            self.log.error(f"Error queuing frame: {e}")
            pass

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Ensures the input frame matches the model's required dimensions.
        """
        target_h, target_w = self.img_dim[1], self.img_dim[0]
        current_h, current_w = frame.shape[:2]

        if current_h != target_h or current_w != target_w:
            self.log.debug(f"Resizing frame from ({current_w}x{current_h}) to ({target_w}x{target_h}). ")
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return frame

    def _process_frames(self):
        """Process frames from the queue"""
        while self.running:
            try:
                if len(self.frame_queue) > 0:
                    (frame, id, filename, ts) = self.frame_queue.pop()
                    detections = []
                    
                    if self.model is None:
                        # Dummy model for debugging
                        h, w = frame.shape[:2]
                        detections = [{
                            'timestamp': ts,
                            'bbox': [w*0.2, h*0.2, w*0.8, h*0.8],
                            'confidence': 0.95,
                            'class_name': 'dummy_object',
                            'class_id': 0
                        }]
                    else:
                        # Run YOLO inference
                        results = self.model.track(
                            frame, 
                            persist=False,
                            conf=self.conf_thresh,
                            imgsz=self.img_size,
                            max_det=self.max_det,
                            retina_masks=True,
                            verbose=False,
                        )

                        # Convert results to detection format
                        if results and len(results) > 0:
                            result = results[0]
                            # Initialize mask data structure
                            masks_data = {}
                            if hasattr(result, 'masks') and result.masks is not None:
                                # result.masks.xy contains the polygon coordinates for each mask
                                # It's a list of NumPy arrays, where each array is N x 2 (N points, x, y coordinates)
                                masks_data = {i: mask_poly.tolist() for i, mask_poly in enumerate(result.masks.xy)}

                            if hasattr(result, 'boxes') and result.boxes is not None:
                                boxes = result.boxes
                                for i in range(len(boxes)):
                                    bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                                    conf = float(boxes.conf[i].cpu().numpy())
                                    cls_id = int(boxes.cls[i].cpu().numpy())
                                    class_name = self.model.names[cls_id] if cls_id < len(self.model.names) else f"class_{cls_id}"
                                    # Get tracking ID if available      
                                    if hasattr(boxes, 'id') and boxes.id is not None:
                                        search_id = int(boxes.id[i].cpu().numpy())
                                    else:
                                        search_id = 0
                                    detection = {
                                        'bbox': bbox.tolist(),
                                        'class': int(cls_id),
                                        'class_name': class_name,
                                        'confidence': conf,
                                        'id': search_id,
                                        'mask': masks_data.get(i, []),
                                    }
                                    detections.append(detection)
                    
                    # Call the provided callback function with detections
                    if self.detection_callback:
                        self.detection_callback(frame, detections, filename, id, ts)
                else:
                    # No frames to process, sleep briefly
                    time.sleep(0.01)

            except Exception as e:
                self.log.error(f"Error processing frame: {e}")
                time.sleep(0.01)
                continue