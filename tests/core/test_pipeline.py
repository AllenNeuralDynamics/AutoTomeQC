import pytest
import cv2
import numpy as np
import logging
from unittest.mock import MagicMock, patch, ANY
from concurrent.futures import TimeoutError, Future

# Import the class
from autotomeqc.core.pipeline import AutoTomePipeline

# --- FIXTURES ---

@pytest.fixture
def mock_yolo_client_class():
    """Mock the YOLO Client so we don't load AI models."""
    with patch("autotomeqc.core.pipeline.YOLOClient") as MockClass:
        yield MockClass

@pytest.fixture
def mock_external_deps():
    """Mock File IO, OpenCV, and Visualization."""
    with patch("autotomeqc.core.pipeline.cv2") as mock_cv2, \
         patch("autotomeqc.core.pipeline.save_json_results") as mock_save_json, \
         patch("autotomeqc.core.pipeline.save_debug_image") as mock_save_img, \
         patch("autotomeqc.core.pipeline.cropped_segmented") as mock_crop:

        # Setup defaults to prevent crashes
        mock_crop.return_value = np.zeros((50, 50, 3), dtype=np.uint8)

        yield {
            "cv2": mock_cv2,
            "save_json": mock_save_json,
            "save_img": mock_save_img,
            "crop": mock_crop
        }

@pytest.fixture
def pipeline(mock_yolo_client_class, mock_config):
    """Creates the pipeline instance with mocked config."""
    with patch.dict("autotomeqc.core.pipeline.CONFIG", mock_config):
        pipe = AutoTomePipeline()
        yield pipe
        # Cleanup if test didn't call stop
        pipe.stop()

# --- TESTS ---

def test_initialization(pipeline):
    """Verify correct setup of executor and criteria."""
    # Check that we have 5 workers as defined
    assert pipeline.executor._max_workers == 5
    # Check that we have 5 criteria functions listed
    assert len(pipeline.qc_criteria) == 5
    # Check function mapping
    assert pipeline.check_color_quality in pipeline.qc_criteria

def test_stop_cleans_up_resources(pipeline):
    """Test that stop() shuts down the thread pool."""
    # Create a spy on the existing executor's shutdown method
    with patch.object(pipeline.executor, 'shutdown') as mock_shutdown:
        pipeline.stop()

        # Ensure YOLO stopped
        pipeline.yolo.stop.assert_called_once()
        # Ensure Threads stopped
        mock_shutdown.assert_called_once_with(wait=False)

def test_process_image_flow(pipeline, mock_external_deps, tmp_path):
    """Test the standard happy path from process_image -> YOLO."""
    # ARRANGE
    image_path = tmp_path / "sample.jpg"
    mock_external_deps["cv2"].imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # ACT
    pipeline.process_image(image_path)

    # ASSERT
    mock_external_deps["cv2"].imread.assert_called_once()
    pipeline.yolo.newframe_captured.assert_called_once()

def test_callback_runs_parallel_qc(pipeline, mock_external_deps):
    """
    Test that the callback triggers the QC checks and saves results.
    We use the real ThreadPool here (Integration style) to ensure no syntax errors in threads.
    """
    # ACT
    pipeline._handle_detection_and_qc(np.zeros((10,10)), {}, "test_parallel")

    # ASSERT
    mock_external_deps["save_json"].assert_called_once()
    saved_data = mock_external_deps["save_json"].call_args[0][0]

    # Check that results contains keys from your specific functions
    criteria = saved_data["criteria"]
    assert "check_color_quality" in criteria
    assert "check_thickness" in criteria

def test_qc_timeout_handling(pipeline, mock_external_deps, caplog):
    """
    Simulate a QC check hanging and timing out.
    We use real Future objects (pre-set with an exception) so as_completed doesn't hang.
    """
    # ARRANGE
    # Create 5 futures that immediately raise TimeoutError when accessed
    futures_list = []
    for _ in range(5):
        f = Future()
        f.set_exception(TimeoutError("Too slow!"))
        futures_list.append(f)
    
    # Patch submit to return these futures one by one
    with patch.object(pipeline.executor, 'submit', side_effect=futures_list):

        # ACT
        pipeline._handle_detection_and_qc(np.zeros((10,10)), {}, "test_timeout")

        # ASSERT
        # 1. Verify we logged the error
        assert "timed out!" in caplog.text

        # Verify JSON output recorded the error
        saved_data = mock_external_deps["save_json"].call_args[0][0]
        criteria = saved_data["criteria"]

        # Pick any check and ensure it failed gracefully
        check_result = list(criteria.values())[0]
        assert check_result["pass"] is False
        assert check_result["error"] == "Timeout"

def test_qc_exception_handling(pipeline, mock_external_deps):
    """Test that a generic crash in a worker thread is caught."""
    # ARRANGE
    futures_list = []
    for _ in range(5):
        f = Future()
        f.set_exception(ValueError("Math Error"))
        futures_list.append(f)

    with patch.object(pipeline.executor, 'submit', side_effect=futures_list):
        # ACT
        pipeline._handle_detection_and_qc(np.zeros((10,10)), {}, "test_crash")

        # ASSERT
        saved_data = mock_external_deps["save_json"].call_args[0][0]
        check_result = list(saved_data["criteria"].values())[0]
        assert check_result["pass"] is False
        assert saved_data["qc_summary"] == "FAIL"