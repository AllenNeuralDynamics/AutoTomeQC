import logging
import sys
import time
import numpy as np
from pathlib import Path
import cv2
import shutil

from autotomeqc.yolo.yolo_client import YOLOClient
from autotomeqc.config.config_loader import CONFIG, TEST_IMG_DIR
from autotomeqc.yolo.visualization import handle_detections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    if not CONFIG:
        logger.error("Application cannot start without configuration. Exiting.")
        sys.exit(1)

    print("Loading YOLO Client with config:", CONFIG["qc"])
    myyoloclient = YOLOClient(config=CONFIG["qc"], detection_callback=handle_detections)
    myyoloclient.start_client()

    image_files = [f for f in TEST_IMG_DIR.glob("*.jpg")]
    try:
        """
        w, h = CONFIG['qc']['yolo']['img_dim']
        for i in range(50):
            dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            myyoloclient.newframe_captured(dummy_frame)
            sleep(0.05) # Simulate frame capture rate
        """
        for file_path in image_files:
            frame = cv2.imread(str(file_path))
            myyoloclient.newframe_captured(frame, current=time.time())
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopping client...")
        
    finally:
        # 5. Stop the worker thread cleanly
        myyoloclient.stop()
        print("Client stopped.")

if __name__ == "__main__":
    main()
