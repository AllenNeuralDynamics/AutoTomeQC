import cv2
import numpy as np
import time
import logging
from collections import deque
import sys # Import sys for basic signal/logging if needed

from threading import Thread, Event # Using Event for better thread signaling
from ultralytics import YOLO
import torch

# Basic logging setup (You can customize this in your main script)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class YoloSegmentation:
    """YOLO segmentation worker that runs in its own thread"""
    
    def __init__(self, config, detection_callback=None):
        """
        :param config: Configuration dictionary.
        :param detection_callback: A function to call with the list of detections.
        """
        # super().__init__() # REMOVED QObject
        self.logger = logging.getLogger(self.__class__.__name__)
        self.weights_path = config.get('weights_path', r'weights\yolo11n.pt')
        self.conf_thresh = config.get('conf_thresh', 0.25)
        self.iou_thresh = config.get('iou_thresh', 0.45)
        self.img_size = config.get('img_size', 640)
        self.img_dim = config.get('img_dim', [640, 480])  # input image dimension for YOLO (w, h)
        self.max_det = config.get('max_det', 30)
        self.model = None
        self.frame_queue = deque(maxlen=1)
        self.running = False
        self.worker_thread = None
        
        # New: Store the callback function
        self.detection_callback = detection_callback
        
        try:
            self.model = YOLO(self.weights_path)
            self.model.overrides['conf'] = self.conf_thresh
            self.model.overrides['iou'] = self.iou_thresh
            self.model.overrides['max_det'] = self.max_det
            self.model.overrides['imgsz'] = self.img_size
            self.model.overrides['verbose'] = False
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"YOLO model loaded from: {self.weights_path}")
            self.logger.info(f"Model is running on: {self.model.device}")

            # Warmup the model
            self._warmup_model()
            self.logger.info("YOLO model warmup completed")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}, running yolo in dummy mode")
            self.model = None

    def start(self):
        """Start the YOLO segmentation thread"""
        if self.running:
            return True
            
        self.running = True
        # Use a standard Thread
        self.worker_thread = Thread(target=self._process_frames, daemon=True)
        self.worker_thread.start()
        self.logger.info("YOLO segmentation thread started")
        return True
    
    def _warmup_model(self):
        """Warm up the model with dummy inference to avoid first-frame delay"""
        if self.model is None:
            return
            
        self.logger.info("Warming up YOLO model...")
        warmup_start = time.time()
        
        try:
            # Create dummy frame matching your expected input
            dummy_frame = np.random.randint(0, 255, (self.img_dim[1], self.img_dim[0], 3), dtype=np.uint8)

            # Run several warmup inferences
            for i in range(3):
                # Using predict for simple warmup instead of track if tracking is not essential here
                _ = self.model.track(dummy_frame, persist=True) 

            # Additional GPU warmup if using CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                
            warmup_time = time.time() - warmup_start
            self.logger.info(f"Model warmup completed in {warmup_time:.2f}s")
            self.warmup_done = True
            
        except Exception as e:
            self.logger.error(f"Warmup failed: {e}")
            self.warmup_done = True  # Continue anyway
    
        
    def stop(self):
        """Stop the YOLO processing thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        self.logger.info("YOLO segmentation worker stopped")
        
    def process_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        if not self.running:
            return
        
        # Use deque's nature to drop older frames if a new one arrives immediately
        try:
            # Clear old frame and append new one to ensure only the latest frame is processed
            self.frame_queue.clear() 
            self.frame_queue.append(frame)
        except:
            pass
            
    def _process_frames(self):
        """Process frames from the queue"""
        while self.running:
            try:
                if len(self.frame_queue) > 0:
                    frame = self.frame_queue.pop()
                    detections = []
                    
                    if self.model is None:
                        # Dummy model for debugging
                        h, w = frame.shape[:2]
                        detections = [{
                            'bbox': [w*0.2, h*0.2, w*0.8, h*0.8],
                            'confidence': 0.95,
                            'class_name': 'dummy_object',
                            'class_id': 0
                        }]
                    else:
                        # Run YOLO inference
                        results = self.model.track(
                            frame, 
                            persist=True
                        )

                        # Convert results to detection format
                        if results and len(results) > 0:
                            result = results[0]
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
                                        'id': search_id
                                    }
                                    detections.append(detection)
                    
                    # Call the provided callback function with detections
                    if self.detection_callback:
                        self.detection_callback(detections)
                else:
                    # No frames to process, sleep briefly
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                time.sleep(0.01)
                continue


class YOLOClient:
    
    def __init__(self, config={}, detection_callback=None):
        # super().__init__() # REMOVED QObject
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fps = config.get('fps', 5)
        self.current_time = None
        
        # Create YOLO segmentator, passing the detection callback
        yolo_config = config.get('yolo', {})
        self.yolo_worker = YoloSegmentation(yolo_config, detection_callback=detection_callback)

        # Note: All PyQT signal/slot connections have been replaced by the direct 
        # `detection_callback` function passed to YoloSegmentation's constructor.

    def start_client(self):
        """Start the YOLO processing worker"""
        try:
            self.yolo_worker.start()
            self.logger.info("Simple YOLO client started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error starting Simple YOLO client: {e}")
            return False
        
    def newframe_captured(self, frame: np.ndarray):
        """Put new frame at the specified FPS rate"""
        current = time.time()
        # Rate limit the frames sent to the YOLO worker
        if self.current_time is None or current - self.current_time > (1/self.fps):
            self.yolo_worker.process_frame(frame)
            self.current_time = current
            
    def stop(self):
        """Stop the YOLO worker"""
        if self.yolo_worker:
            self.yolo_worker.stop()


# --- Example Usage ---
# If you need an example of how to use the detection_callback:
"""
def handle_detections(detections):
    print(f"Received {len(detections)} detections.")
    # for detection in detections:
    #     print(f"  {detection['class_name']} with confidence {detection['confidence']:.2f}")

if __name__ == '__main__':
    # 1. Define configuration
    yolo_client_config = {
        'fps': 10,
        'yolo': {
            'weights_path': 'path/to/your/model.pt', # REPLACE with your actual path!
            'img_dim': [640, 480] 
        }
    }
    
    # 2. Define the callback function (replaces the Signal)
    def handle_detections(detections):
        print(f"Thread: {threading.current_thread().name} - Received {len(detections)} detections.")
        # Add your main thread processing logic here

    # 3. Initialize and start the client
    client = YOLOClient(config=yolo_client_config, detection_callback=handle_detections)
    client.start_client()
    
    # 4. Simulate a video stream (sending frames)
    # The 'newframe_captured' method is now a regular method call.
    try:
        w, h = yolo_client_config['yolo']['img_dim']
        for i in range(50):
            # Create a dummy frame
            dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            client.newframe_captured(dummy_frame)
            time.sleep(0.05) # Simulate frame capture rate

    except KeyboardInterrupt:
        print("Stopping client...")
        
    finally:
        # 5. Stop the worker thread cleanly
        client.stop()
        print("Client stopped.")
"""