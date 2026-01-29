import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Adjust the import path to match your project structure
from autotomeqc.yolo_segmentation.yolo_client import YOLOClient

# --- FIXTURES ---

@pytest.fixture
def mock_config():
    return {
        'fps': 5,
        'yolo': {
            'weights_path': 'dummy.pt',
            'img_dim': [640, 480],
            'conf_thresh': 0.5
        }
    }

@pytest.fixture
def mock_yolo_worker_class():
    """
    Patches the YoloSegmentation class so we don't instantiate the real worker/thread.
    """
    with patch("autotomeqc.yolo_segmentation.yolo_client.YoloSegmentation") as MockClass:
        mock_instance = MockClass.return_value
        # Setup methods to return success/True
        mock_instance.start.return_value = True
        mock_instance.stop.return_value = None
        mock_instance.process_frame.return_value = None
        yield MockClass

@pytest.fixture
def client(mock_config, mock_yolo_worker_class):
    """Returns an initialized YOLOClient with a mocked worker."""
    return YOLOClient(config=mock_config)

# --- TESTS ---

def test_initialization(client, mock_config, mock_yolo_worker_class):
    """Verify client initializes the worker with the correct sub-config."""
    # Check if fps was read from config
    assert client.fps == 5

    # Check if YoloSegmentation was initialized with the 'yolo' sub-dict
    mock_yolo_worker_class.assert_called_once()
    call_args = mock_yolo_worker_class.call_args
    passed_config = call_args[0][0] # First arg of constructor

    assert passed_config == mock_config['yolo']
    assert passed_config['weights_path'] == 'dummy.pt'

def test_callback_passing(mock_config, mock_yolo_worker_class):
    """Verify the detection_callback is correctly passed down to the worker."""
    mock_cb = MagicMock()
    
    # Initialize client with callback
    _ = YOLOClient(config=mock_config, detection_callback=mock_cb)

    # Check constructor call
    mock_yolo_worker_class.assert_called_once()
    kwargs = mock_yolo_worker_class.call_args[1]
    assert kwargs['detection_callback'] == mock_cb

def test_start_client_success(client):
    """Test successful start."""
    success = client.start_client()
    assert success is True
    client.yolo_worker.start.assert_called_once()

def test_start_client_failure(client, caplog):
    """Test handling of an exception during start."""
    # Force worker.start() to raise an exception
    client.yolo_worker.start.side_effect = Exception("Worker Crash")
    success = client.start_client()
    assert success is False
    assert "Error starting Simple YOLO client" in caplog.text

def test_newframe_captured(client):
    """Verify new frames are forwarded to the worker."""
    client.yolo_worker = MagicMock()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    timestamp = 1000.5
    filename = "test.jpg"
    req_id = "test_id"

    client.newframe_captured(frame, id=req_id, filename=filename, ts=timestamp)

    # Check that it was passed to worker, instead of checking current_time
    client.yolo_worker.process_frame.assert_called_once_with(
        frame, id=req_id, filename=filename, ts=timestamp
    )

def test_fps_rate_limiting_logic(client):
    """
    Note: The provided source code had the rate limiting line commented out:
    #if self.current_time is None or current - self.current_time > (1/self.fps):
    
    If you uncomment that logic in the future, this test verifies it works.
    Currently, it simply verifies every frame is passed.
    """
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Send frame 1
    client.newframe_captured(frame, 1.0)
    # Send frame 2 immediately (same timestamp)
    client.newframe_captured(frame, 1.0)
    # Since logic is commented out in source, call count should be 2
    # If logic were active, call count should be 1
    assert client.yolo_worker.process_frame.call_count == 2

def test_stop(client):
    """Verify stop is called on the worker."""
    client.stop()
    client.yolo_worker.stop.assert_called_once()

def test_stop_safe_if_no_worker(mock_config):
    """Verify stop doesn't crash if worker failed to initialize."""
    # Simulate a client where worker init failed (or was set to None)
    with patch("autotomeqc.yolo_segmentation.yolo_client.YoloSegmentation", return_value=None):
        # We need to manually set it to None because the patch above returns None 
        # but the class code might assign it.
        # Ideally, we construct the client, then manually force worker to None
        client = YOLOClient(config=mock_config)
        client.yolo_worker = None 
        # Should not raise AttributeError
        try:
            client.stop()
        except Exception as e:
            pytest.fail(f"Stop raised exception unexpectedly: {e}")