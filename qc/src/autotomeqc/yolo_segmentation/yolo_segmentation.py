# autotomeqc/yolo_segmentation/yolo_segmentation.py
import numpy as np
import time
import logging
from threading import Event
import cv2
from autotomeqc.config.schemas import YoloSettings
from autotomeqc.core.models import Detection
from ultralytics import YOLO
import torch


class YoloSegmentation:
    """Synchronous YOLO segmentation component for loading the model and processing frames."""
    
    def __init__(self, config: YoloSettings):
        """
        :param config: Configuration dictionary.
        """
        self.weights_path = config.weights_path
        self.conf_thresh = config.conf_thresh
        self.img_size = config.img_size
        self.img_dim = config.img_dim
        self.max_det = config.max_det
        self.mask_open_kernel = config.mask_open_kernel
        self.log = logging.getLogger(self.__class__.__name__)

        # State
        self.model = None
        self.ready: Event = Event()
        self._load_model()

    def _load_model(self):
        """Encapsulate model loading logic"""
        try:
            self.model = YOLO(self.weights_path)
            self.log.info(f"YOLO segmentation model loaded from: {self.weights_path}")

            # Check device availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            self.log.info(f"YOLO segmentation model is running on: {device}")
            self._warmup_model()
            self.ready.set()
        except Exception as e:
            self.log.error(f"Failed to load YOLO segmentation model: {e}. Running in DUMMY mode.")
            self.model = None

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

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Ensures the input frame matches the model's required dimensions.
        """
        target_h, target_w = self.img_dim[1], self.img_dim[0]
        current_h, current_w = frame.shape[:2]

        if current_h != target_h or current_w != target_w:
            self.log.debug(f"Resizing frame from ({current_w}x{current_h}) to ({target_w}x{target_h}). ")
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def _largest_polygon(
        mask_tensor: torch.Tensor,
        orig_shape: tuple[int, int],
        fallback_xy: np.ndarray,
        open_kernel: int = 0,
    ) -> list[list[float]]:
        """
        Build a polygon from the binary mask keeping only its largest connected piece.
        masks.xy joins disjoint pieces with hairline connectors, which leaves tails and stray blobs.
        Falls back to the original polygon if the mask is empty.
        """
        h, w = orig_shape[:2]
        mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Opening removes thin bridges and 1-px spurs; shredding on the section edge is much wider than the kernel
        if open_kernel > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            if opened.any():
                mask = opened
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return fallback_xy.tolist()
        largest = max(contours, key=cv2.contourArea)
        return largest.reshape(-1, 2).astype(float).tolist()

    def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """Process frames from the queue"""
        try:
            detections: list[Detection] = []
        
            if self.model is None:
                # Dummy model for debugging
                h, w = frame.shape[:2]
                return [Detection(
                    bbox=[w*0.2, h*0.2, w*0.8, h*0.8],
                    confidence=0.95,
                    class_name='dummy_object',
                    class_id=0
                )]

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
                    masks_data = {
                        i: self._largest_polygon(result.masks.data[i], result.orig_shape, mask_poly, self.mask_open_kernel)
                        for i, mask_poly in enumerate(result.masks.xy)
                    }
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.model.names.get(cls_id, f"unknown_{cls_id}")
                        # Get tracking ID if available      
                        if hasattr(boxes, 'id') and boxes.id is not None:
                            search_id = int(boxes.id[i].cpu().numpy())
                        else:
                            search_id = 0

                        detection = Detection(
                            bbox=bbox.tolist(),
                            class_id=int(cls_id),
                            class_name=class_name,
                            confidence=conf,
                            track_id=search_id,
                            mask=masks_data.get(i, [])
                        )
                        detections.append(detection)
            return detections
        except Exception as e:
            self.log.error(f"Error processing frame: {e}")
            time.sleep(0.01)
            return []
