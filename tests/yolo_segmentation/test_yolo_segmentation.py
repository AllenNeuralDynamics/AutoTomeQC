# tests/yolo_segmentation/test_yolo_segmentation.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from autotomeqc.yolo_segmentation.yolo_segmentation import YoloSegmentation

# --- FIXTURES ---

@pytest.fixture
def mock_config():
    """Provides a mocked config object for YoloSettings."""
    config = MagicMock()
    config.weights_path = "weights/test_yolo.pt"
    config.conf_thresh = 0.6
    config.img_size = 640
    config.img_dim = [640, 640]
    config.max_det = 30
    return config

@pytest.fixture
def mock_yolo_class():
    """
    Patches the ultralytics.YOLO class to avoid loading real weights.
    """
    with patch("autotomeqc.yolo_segmentation.yolo_segmentation.YOLO") as MockClass:
        mock_instance = MockClass.return_value
        mock_instance.names = {0: "section", 1: "lasso"}
        mock_instance.track.return_value = []
        yield MockClass

@pytest.fixture
def server(mock_config, mock_yolo_class):
    """
    Returns a synchronous YoloSegmentation instance.
    """
    # Force CPU to avoid CUDA dependency in CI/test environments
    with patch("torch.cuda.is_available", return_value=False):
        server = YoloSegmentation(config=mock_config)
        yield server
        server.stop()

# --- TESTS ---

def test_initialization(server, mock_config, mock_yolo_class):
    """Verify config is loaded and model settings are applied."""
    assert server.conf_thresh == 0.6
    assert server.weights_path == "weights/test_yolo.pt"
    mock_yolo_class.assert_called_once_with("weights/test_yolo.pt")

def test_warmup_logic(server):
    """Test that warmup runs inference 3 times."""
    # Reset mock because __init__ already ran warmup
    server.model.track.reset_mock()
    server._warmup_model()
    assert server.model.track.call_count == 3

def test_resize_frame(server):
    """Verify image resizing logic to target dimensions."""
    # Create a frame of wrong size (100x100)
    input_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = server.resize_frame(input_frame)
    
    # Target is 640x640 from mock_config
    assert resized.shape == (640, 640, 3)

def test_process_frame_logic(server):
    """Test data extraction from YOLO Results object."""
    # Setup Mock YOLO Result
    mock_result = MagicMock()
    
    # Setup Boxes
    mock_boxes = MagicMock()
    mock_boxes.__len__.return_value = 1
    
    def mock_tensor(value):
        t = MagicMock()
        t.cpu.return_value.numpy.return_value = value
        return t

    mock_boxes.xyxy = [mock_tensor(np.array([10, 10, 50, 50]))]
    mock_boxes.conf = [mock_tensor(np.array(0.95))]
    mock_boxes.cls  = [mock_tensor(np.array(0))]
    mock_boxes.id   = None  # Test case where tracking ID might be missing
    mock_result.boxes = mock_boxes
    
    # Setup Masks
    mock_result.masks.xy = [np.array([[10, 10], [20, 20], [10, 20]])]
    server.model.track.return_value = [mock_result]

    # Execute
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = server.process_frame(frame)

    # Assertions
    assert len(detections) == 1
    assert detections[0]['class_name'] == "section"
    assert detections[0]['confidence'] == 0.95
    assert detections[0]['id'] == 0  # Default if id is None
    assert isinstance(detections[0]['mask'], list)