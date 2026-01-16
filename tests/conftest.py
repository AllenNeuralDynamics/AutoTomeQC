import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_config():
    """
    Provides a standard configuration dictionary for testing.
    Mirrors the 'qc' structure from yolo-config.yaml.
    """
    return {
        "qc": {
            "fps": 1,
            "yolo": {
                "weights_path": "weights/seg_fast.pt",
                "conf_thresh": 0.90,
                "iou_thresh": 0.45,
                "img_size": 960,
                "img_dim": [960, 960],
                "max_det": 30,
                "loop_bbox_margin": 30
            },
            # For testing, you often want this to be a temp folder rather than
            # creating real folders in your project root.
            "output_dir": "example/output", 
            "save_segmented_images": True
        }
    }

@pytest.fixture
def sample_image_data():
    """
    Returns fake bytes representing an image.
    Useful for testing 'utils/io.py' or 'yolo_segmentation'.
    """
    return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...'