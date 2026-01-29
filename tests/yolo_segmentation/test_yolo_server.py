import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch

# Adjust import path to match your project structure
from autotomeqc.yolo_segmentation.yolo_server import YoloSegmentation

# --- FIXTURES ---

@pytest.fixture
def mock_config():
    return {
        "weights_path": "dummy_weights.pt",
        "conf_thresh": 0.6,
        "iou_thresh": 0.4,
        "img_size": 640,
        "max_det": 10,
        "loop_bbox_margin": 15
    }

@pytest.fixture
def mock_yolo_class():
    """
    Patches the ultralytics.YOLO class so we don't download/load real models.
    """
    with patch("autotomeqc.yolo_segmentation.yolo_server.YOLO") as MockClass:
        mock_instance = MockClass.return_value
        # Setup 'overrides' dict to prevent KeyError in __init__
        mock_instance.overrides = {}
        # Setup 'names' for class label lookup
        mock_instance.names = {0: "section", 1: "lasso"}
        # Setup 'device' property
        mock_instance.device = "cpu"
        # Default behavior for track/predict: return empty list
        mock_instance.track.return_value = []
        mock_instance.predict.return_value = []
        yield MockClass

@pytest.fixture
def server(mock_config, mock_yolo_class):
    """
    Returns an initialized YoloSegmentation instance with mocked dependencies.
    Force CPU mode to avoid CUDA checks in tests.
    """
    with patch("torch.cuda.is_available", return_value=False):
        server = YoloSegmentation(config=mock_config)
        yield server
        # Cleanup: Ensure thread is stopped after every test
        server.stop()

# --- TESTS ---
def test_initialization(server, mock_config, mock_yolo_class):
    """Verify config is loaded and model settings are applied."""
    assert server.conf_thresh == 0.6
    assert server.weights_path == "dummy_weights.pt"
    # Check if YOLO was loaded
    mock_yolo_class.assert_called_with("dummy_weights.pt")

def test_initialization_failure(mock_config, caplog):
    """Test graceful fallback if YOLO fails to load."""
    with patch("autotomeqc.yolo_segmentation.yolo_server.YOLO", side_effect=Exception("Corrupt Weights")):
        server = YoloSegmentation(config=mock_config)
        assert server.model is None
        assert "Failed to load YOLO model" in caplog.text
        # Ensure it switches to dummy mode behavior (implied by model=None)

def test_queue_holds_multiple_frames(server):
    """
    Verify queue behavior with multiple inputs:
    Adding new frames should append them, not drop old ones.
    """
    # Simulate adding frames directly
    server.running = True # Needs to be running to accept frames
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.ones((100, 100, 3), dtype=np.uint8)

    # Add Frame 1
    server.process_frame(frame1, ts=1.0, filename="frame1")
    assert len(server.frame_queue) == 1

    # Add Frame 2 (Should now append, resulting in 2 frames)
    server.process_frame(frame2, ts=2.0, filename="frame2")
    assert len(server.frame_queue) == 2

    # Check that Frame 1 is still first (FIFO)
    # Tuple unpacking: (frame, id, filename, ts)
    (_, _, name1, ts1) = server.frame_queue[0]
    assert name1 == "frame1"
    assert ts1 == 1.0

    # Check that Frame 2 is second
    (_, _, name2, ts2) = server.frame_queue[1]
    assert name2 == "frame2"
    assert ts2 == 2.0

def test_warmup_logic(server, mock_yolo_class, caplog):
    """Test that warmup runs the model 3 times."""
    # Reset mock because __init__ already ran warmup once
    server.model.track.reset_mock()
    server._warmup_model()
    
    # Assert track was called 3 times
    assert server.model.track.call_count == 3
    assert "Warmup failed" not in caplog.text

def test_inference_callback(server):
    """
    Test that the detection callback is invoked with results.
    """
    # Setup Callback
    mock_callback = MagicMock()
    server.detection_callback = mock_callback

    # Setup YOLO Result Mock
    mock_result = MagicMock()
    # Create a container mock for 'boxes', not a list
    mock_boxes_container = MagicMock()
    # Set length so the loop range(len(boxes)) works
    mock_boxes_container.__len__.return_value = 1 
    # Helper to create a fake tensor that supports .cpu().numpy()
    def mock_tensor(value):
        t = MagicMock()
        t.cpu.return_value.numpy.return_value = value
        return t

    # Mock the box attributes
    mock_boxes_container.xyxy = [mock_tensor(np.array([10, 10, 50, 50]))]
    mock_boxes_container.conf = [mock_tensor(np.array(0.95))]
    mock_boxes_container.cls  = [mock_tensor(np.array(0))]
    mock_boxes_container.id   = [mock_tensor(np.array(1))]
    mock_result.boxes = mock_boxes_container
    mock_result.masks.xy = [np.array([[10,10], [20,20], [10,20]])]
    server.model.track.return_value = [mock_result]

    # Start and Add a frame to queue
    server.start()
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    server.process_frame(fake_frame, ts=123.4, filename="test_img")

    # Wait briefly for the thread to process
    time.sleep(0.1)

    # Stop and Assert
    server.stop()
    mock_callback.assert_called()
    
    # Debug check to see what was passed
    if mock_callback.called:
        args = mock_callback.call_args[0]
        print("Callback received:", args[1]) # Print detections list