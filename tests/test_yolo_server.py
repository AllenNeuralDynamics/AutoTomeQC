import pytest
import numpy as np
import time
import threading
from unittest.mock import MagicMock, patch

# Assuming YoloSegmentation is available for import
from autotomeqc.yolo.yolo_server import YoloSegmentation 

# ---------------------------------------------
# --- Mocking Setup ---
# ---------------------------------------------

class MockYOLO:
    """Mock for ultralytics.YOLO, ensuring track returns structured results."""
    def __init__(self, weights_path):
        self.overrides = {}
        self.names = {0: 'test_class'}
        self.device = 'cpu'
        self.to = MagicMock()
        
        # FIX: Define track as an instance MagicMock that reliably returns the result structure
        self.track = MagicMock(return_value=[self._create_mock_result()])
        
    def _create_mock_result(self):
        """Creates a result with one object (minimal data)."""
        
        # Mocking the box data arrays
        mock_boxes = MagicMock()
        mock_boxes.xyxy = np.array([[10, 10, 50, 50]]) 
        mock_boxes.conf = np.array([0.9])
        mock_boxes.cls = np.array([0])
        mock_boxes.id = np.array([1])
        
        # Mocking mask polygon data
        mock_masks = MagicMock()
        mock_masks.xy = [np.array([[10, 10], [50, 10], [50, 50]])]

        # Assemble the final result object
        mock_result = MagicMock()
        # Ensure 'boxes' and 'masks' attributes are explicitly set and not None
        mock_result.boxes = mock_boxes
        mock_result.masks = mock_masks
        return mock_result

@pytest.fixture(autouse=True)
def setup_core_patches():
    """Patch external libraries for fast, isolated testing."""
    # FIX: Use the fully qualified path to patch the YOLO object in the module where it is used.
    YOLO_PATCH_PATH = 'autotomeqc.yolo.yolo_server.YOLO'
    
    with patch(YOLO_PATCH_PATH, new=MockYOLO), \
         patch('time.sleep', MagicMock()), \
         patch('torch.cuda.is_available', return_value=False):
        yield

# ---------------------------------------------
# --- Fixtures ---
# ---------------------------------------------

@pytest.fixture
def mock_frame():
    """Simple dummy input frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8) 

@pytest.fixture
def mock_callback():
    """Returns a mock object that can be asserted against."""
    return MagicMock()

@pytest.fixture
def callback_event():
    """Threading event for synchronization."""
    return threading.Event()

@pytest.fixture
def yolo_worker(mock_callback, callback_event):
    """Instance with a synchronized mock callback."""
    config = {'weights_path': 'dummy.pt', 'img_dim': [640, 480]} 

    # Wrapper function for synchronization
    # NOTE: This wrapper must match the signature expected by the worker: (frame, detections)
    def synchronized_callback(frame, detections):
        mock_callback(frame, detections) # 1. Call the mock object to record call history
        callback_event.set()            # 2. Signal the test thread

    # Pass the synchronization wrapper to the worker
    return YoloSegmentation(config, detection_callback=synchronized_callback)

# ---------------------------------------------
# --- Core Tests ---
# ---------------------------------------------

def test_01_initialization_and_warmup(yolo_worker):
    """Test model loads (mocked) and warmup is called 3 times."""
    assert yolo_worker.model is not None
    assert yolo_worker.model.track.call_count == 3 

def test_02_start_stop_thread_lifecycle(yolo_worker):
    """Test the thread starts and stops correctly."""
    
    yolo_worker.start()
    assert yolo_worker.running is True
    assert yolo_worker.worker_thread.is_alive()
    
    yolo_worker.stop()
    assert yolo_worker.running is False
    yolo_worker.worker_thread.join(timeout=0.1)
    assert not yolo_worker.worker_thread.is_alive()

def test_03_frame_queue_and_callback(yolo_worker, mock_frame, mock_callback, callback_event):
    """Test frame is processed and callback is triggered with detection data (SYNCHRONIZED)."""
    
    mock_ts = time.time()
    initial_call_count = yolo_worker.model.track.call_count 

    # 1. Start worker and enqueue frame
    yolo_worker.start()
    yolo_worker.process_frame(mock_frame, mock_ts)
    
    # 2. CRITICAL: Wait for the background thread to signal completion
    was_called = callback_event.wait(timeout=1.0) 

    # 3. Stop the thread
    yolo_worker.running = False
    yolo_worker.worker_thread.join(timeout=0.1)
    
    if not was_called:
        pytest.fail("Callback was not executed by the worker thread within the timeout (1.0s).")

    # The inference loop should have run exactly once after warmup
    assert yolo_worker.model.track.call_count == initial_call_count + 1 