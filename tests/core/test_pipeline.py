import pytest
import numpy as np
from unittest.mock import patch
from concurrent.futures import TimeoutError, Future

# Import the class
from autotomeqc.core.pipeline import AutoTomePipeline

# --- FIXTURES ---

@pytest.fixture
def mock_config():
    """Mock config with ALL required sections to avoid KeyErrors."""
    return {
        "qc": {
            "fps": 1, 
            "output_dir": "example/output",
            "save_segmented_images": True,
            "yolo": {
                "conf_thresh": 0.9, "img_dim": [960, 960], 
                "img_size": 960, "iou_thresh": 0.45, "weights_path": "dummy.pt"
            },
            "section_coverage": {
                "weights_path": "dummy.pt", "img_size": 224,
                "pass_labels": ["full"], "min_confidence": 0.5
            },
            "knife_mark": {
                "weights_path": "dummy.pt", "img_size": 640,
                "pass_labels": ["none"], "min_confidence": 0.5
            },
            "thickness_consistency": {
                "weights_path": "dummy.pt", "img_size": 224,
                "pass_labels": ["consistent"], "min_confidence": 0.5
            },
            "thickness": {
                "weights_path": "dummy.pt", "img_size": 224,
                "pass_labels": "ANY", "min_confidence": 0.5
            }
        }
    }

@pytest.fixture
def mock_yolo_client_class():
    """Mock the YOLO Client so we don't load AI models."""
    with patch("autotomeqc.core.pipeline.YOLOClient") as MockClass:
        yield MockClass

@pytest.fixture
def mock_algorithms():
    """
    CRITICAL: Mock the QC Algorithm classes.
    This prevents the pipeline from trying to load real YOLO weights
    or running real inference during unit tests.
    """
    with patch("autotomeqc.core.pipeline.SectionCoverageQC") as cov, \
         patch("autotomeqc.core.pipeline.KnifeMarksQC") as knife, \
         patch("autotomeqc.core.pipeline.ThicknessConsistencyQC") as thick_c, \
         patch("autotomeqc.core.pipeline.ThicknessQC") as thick:

        # Setup default "Pass" behavior
        for mock in [cov, knife, thick_c, thick]:
            # Each mock needs a .check() method that returns a dict
            mock.return_value.check.return_value = {
                "pass": True, "label": "Mocked", "conf": 0.99
            }

        yield {
            "coverage": cov,
            "knife": knife,
            "thick_c": thick_c,
            "thick": thick
        }

@pytest.fixture
def mock_external_deps():
    """Mock File IO, OpenCV, and Visualization helpers."""
    with patch("autotomeqc.core.pipeline.cv2") as mock_cv2, \
         patch("autotomeqc.core.pipeline.save_json_results") as mock_save_json, \
         patch("autotomeqc.core.pipeline.save_debug_image") as mock_save_img, \
         patch("autotomeqc.core.pipeline.save_failure_report") as mock_save_fail, \
         patch("autotomeqc.core.pipeline.cropped_segmented") as mock_crop, \
         patch("autotomeqc.core.pipeline.get_best_section_detection") as mock_best:

        # Setup defaults to prevent crashes
        mock_crop.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        # Mock the confidence extraction
        mock_best.return_value = {"confidence": 0.95}

        yield {
            "cv2": mock_cv2,
            "save_json": mock_save_json,
            "save_img": mock_save_img,
            "save_fail": mock_save_fail,
            "crop": mock_crop,
            "best_det": mock_best
        }

@pytest.fixture
def pipeline(mock_yolo_client_class, mock_config, mock_algorithms):
    """Creates the pipeline instance with mocked config and algorithms."""
    with patch.dict("autotomeqc.core.pipeline.CONFIG", mock_config):
        pipe = AutoTomePipeline()
        yield pipe
        # Cleanup
        pipe.stop()

# --- TESTS ---

def test_initialization(pipeline):
    """Verify correct setup of executor and criteria."""
    # Check that we have 5 workers as defined
    assert pipeline.executor._max_workers == 5
    # UPDATED: Check for 'qc_modules' instead of 'qc_criteria'
    # We now have 4 modules (coverage, knife, thickness, consistency)
    assert len(pipeline.qc_modules) == 4
    assert "coverage" in pipeline.qc_modules
    assert "knife_mark" in pipeline.qc_modules

def test_stop_cleans_up_resources(pipeline):
    """Test that stop() shuts down the thread pool."""
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
    # ACT
    pipeline.process_image(str(image_path))
    # ASSERT
    mock_external_deps["cv2"].imread.assert_called_once()
    pipeline.yolo.newframe_captured.assert_called_once()

def test_handle_detection_runs_qc(pipeline, mock_external_deps):
    """
    Test that the callback triggers the QC checks and saves results.
    UPDATED: Renamed method to _handle_detection and using list of dicts.
    """
    # ACT - Simulate YOLO finishing
    detections = [{'class_name': 'section', 'confidence': 0.9}]
    pipeline._handle_detection(np.zeros((10,10)), detections, "test_file")
    # ASSERT
    mock_external_deps["save_json"].assert_called_once()
    saved_data = mock_external_deps["save_json"].call_args[0][0]
    # Check that results contains keys from your specific functions
    criteria = saved_data["criteria"]
    assert "coverage" in criteria
    assert "knife_mark" in criteria
    assert saved_data["qc_summary"] == "PASS"

def test_qc_timeout_handling(pipeline, mock_external_deps, caplog):
    """
    Simulate a QC check hanging and timing out.
    """
    # ARRANGE
    # Create 4 futures that immediately raise TimeoutError
    futures_list = []
    for _ in range(4):
        f = Future()
        f.set_exception(TimeoutError("Too slow!"))
        futures_list.append(f)

    # Patch submit to return these futures
    with patch.object(pipeline.executor, 'submit', side_effect=futures_list):
        # ACT
        detections = [{'class_name': 'section', 'confidence': 0.9}]
        pipeline._handle_detection(np.zeros((10,10)), detections, "test_timeout")

        # ASSERT
        # Verify JSON output recorded the error
        saved_data = mock_external_deps["save_json"].call_args[0][0]
        criteria = saved_data["criteria"]

        # Pick any check and ensure it failed gracefully
        check_result = list(criteria.values())[0]
        assert check_result["pass"] is False
        assert saved_data["qc_summary"] == "FAIL"

def test_qc_exception_handling(pipeline, mock_external_deps):
    """Test that a generic crash in a worker thread is caught."""
    # ARRANGE
    futures_list = []
    for _ in range(4):
        f = Future()
        f.set_exception(ValueError("Math Error"))
        futures_list.append(f)

    with patch.object(pipeline.executor, 'submit', side_effect=futures_list):
        # ACT
        detections = [{'class_name': 'section', 'confidence': 0.9}]
        pipeline._handle_detection(np.zeros((10,10)), detections, "test_crash")

        # ASSERT
        saved_data = mock_external_deps["save_json"].call_args[0][0]
        check_result = list(saved_data["criteria"].values())[0]
        assert check_result["pass"] is False
        assert saved_data["qc_summary"] == "FAIL"