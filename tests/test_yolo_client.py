import numpy as np
import time
from unittest.mock import MagicMock, patch
from autotomeqc.yolo.yolo_client import YOLOClient


def test_yolo_client_like_example_usage(monkeypatch):
    """Simulate running YOLOClient as shown in the example usage block."""

    # --- Mock YoloSegmentation so no real YOLO starts ---
    mock_seg_instance = MagicMock()
    mock_seg_class = MagicMock(return_value=mock_seg_instance)
    monkeypatch.setattr("autotomeqc.yolo.yolo_client.YoloSegmentation", mock_seg_class)

    # --- Dummy config (similar to example) ---
    yolo_client_config = {
        "fps": 10,
        "yolo": {
            "weights_path": "fake/path/to/model.pt",
            "img_dim": [64, 48]
        }
    }

    # --- Dummy callback ---
    def dummy_callback(detections):
        print(f"[TEST CALLBACK] Got {len(detections)} detections (mocked)")

    # --- Create client ---
    client = YOLOClient(config=yolo_client_config, detection_callback=dummy_callback)

    # Verify that the YOLO segmentation worker was created correctly
    mock_seg_class.assert_called_once_with(
        yolo_client_config["yolo"],
        detection_callback=dummy_callback
    )

    # --- Start client ---
    assert client.start_client() is True
    mock_seg_instance.start.assert_called_once()

    # --- Simulate sending several dummy frames ---
    w, h = yolo_client_config["yolo"]["img_dim"]
    for i in range(5):
        dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        client.newframe_captured(dummy_frame, current=time.time())
        time.sleep(0.05)

    # Verify that frames were processed
    assert mock_seg_instance.process_frame.called, "process_frame() was never called"

    # --- Stop client ---
    client.stop()
    mock_seg_instance.stop.assert_called_once()

    print("[TEST] YOLOClient example-like run completed successfully.")
