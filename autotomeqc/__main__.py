import logging
import sys
import time
import cv2

from autotomeqc.yolo.yolo_client import YOLOClient
from autotomeqc.config.config_loader import CONFIG, TEST_IMG_DIR
from autotomeqc.yolo.visualization import handle_detections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    yolo = YOLOClient(config=CONFIG["qc"], detection_callback=handle_detections)
    yolo.start_client()

    image_files = [f for f in TEST_IMG_DIR.glob("*.jpg")]
    try:
        """
        w, h = CONFIG['qc']['yolo']['img_dim']
        for i in range(50):
            dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            yolo.newframe_captured(dummy_frame)
            sleep(0.05) # Simulate frame capture rate
        """
        for file_path in image_files:
            frame = cv2.imread(str(file_path))
            yolo.newframe_captured(frame, current=time.time())
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopping Yolo...")
        
    finally:
        yolo.stop()
        print("Yolo stopped.")

if __name__ == "__main__":
    main()
