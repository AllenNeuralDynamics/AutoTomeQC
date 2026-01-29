import pytest
import numpy as np
import time
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
            "save_input_images": False,
            "yolo": {
                "conf_thresh": 0.9, "img_dim": [960, 960], 
                "img_size": 960, "iou_thresh": 0.45, "weights_path": "dummy.pt"
            },
            # Mock config for all 5 algorithms
            "section_coverage": {"weights_path": "d", "img_size": 224, "pass_labels": ["full"], "min_confidence": 0.5},
            "knife_mark": {"weights_path": "d", "img_size": 640, "pass_labels": ["none"], "min_confidence": 0.5},
            "thickness_consistency": {"weights_path": "d", "img_size": 224, "pass_labels": ["consistent"], "min_confidence": 0.5},
            "thickness": {"weights_path": "d", "img_size": 224, "pass_labels": "ANY", "min_confidence": 0.5},
            "shape": {"weights_path": "d", "img_size": 224, "pass_labels": ["good"], "min_confidence": 0.5}
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
    """
    with patch("autotomeqc.core.pipeline.SectionCoverageQC") as cov, \
         patch("autotomeqc.core.pipeline.KnifeMarksQC") as knife, \
         patch("autotomeqc.core.pipeline.ThicknessConsistencyQC") as thick_c, \
         patch("autotomeqc.core.pipeline.ThicknessQC") as thick, \
         patch("autotomeqc.core.pipeline.ShapeQC") as shape:

        # Setup default "Pass" behavior
        for mock in [cov, knife, thick_c, thick, shape]:
            mock.return_value.check.return_value = {
                "pass": True, "label": "Mocked", "conf": 0.99
            }

        yield {
            "coverage": cov, "knife": knife, "thick_c": thick_c, "thick": thick, "shape": shape
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

        # Setup defaults
        mock_crop.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_best.return_value = {"confidence": 0.95}

        yield {
            "cv2": mock_cv2, "save_json": mock_save_json, "save_img": mock_save_img, 
            "save_fail": mock_save_fail, "crop": mock_crop, "best_det": mock_best
        }

@pytest.fixture
def pipeline(mock_yolo_client_class, mock_config, mock_algorithms):
    """Creates the pipeline instance with mocked config."""
    with patch.dict("autotomeqc.core.pipeline.CONFIG", mock_config):
        pipe = AutoTomePipeline()
        yield pipe
        pipe.stop()

# --- TESTS ---

def test_initialization(pipeline):
    """Verify correct setup of executor and criteria."""
    assert pipeline.executor._max_workers == 5
    # Updated: Now checking for 5 modules including ShapeQC
    assert len(pipeline.qc_modules) == 5
    assert "shape" in pipeline.qc_modules

def test_stop_cleans_up_resources(pipeline):
    """Test that stop() shuts down the thread pool."""
    with patch.object(pipeline.executor, 'shutdown') as mock_shutdown:
        pipeline.stop()
        pipeline.yolo.stop.assert_called_once()
        mock_shutdown.assert_called_once_with(wait=False)

def test_process_flow(pipeline, mock_external_deps, tmp_path):
    """Test the standard happy path from process() -> YOLO."""
    # ARRANGE
    image_path = tmp_path / "sample.jpg"
    # ACT (Corrected: using .process instead of .process_image)
    pipeline.process(img_path=str(image_path))
    # ASSERT
    mock_external_deps["cv2"].imread.assert_called_once()
    pipeline.yolo.newframe_captured.assert_called_once()

def test_handle_detection_runs_qc(pipeline, mock_external_deps):
    """
    Test that the callback triggers the QC checks and saves results.
    """
    # ACT - Simulate YOLO finishing
    detections = [{'class_name': 'section', 'confidence': 0.9}]
    ts = time.time()
    
    # Corrected Args: frame, detections, filename, id, ts
    pipeline._handle_detection(
        np.zeros((10,10)), 
        detections, 
        "test_file", 
        "dummy_uuid", 
        ts
    )

    # ASSERT
    mock_external_deps["save_json"].assert_called_once()
    saved_data = mock_external_deps["save_json"].call_args[0][0]
    
    criteria = saved_data["criteria"]
    assert "coverage" in criteria
    assert "shape" in criteria
    assert saved_data["qc_summary"] == "PASS"

def test_qc_timeout_handling(pipeline, mock_external_deps):
    """Simulate a QC check hanging and timing out."""
    # ARRANGE: 5 futures (for 5 modules) that raise TimeoutError
    futures_list = []
    for _ in range(5):
        f = Future()
        f.set_exception(TimeoutError("Too slow!"))
        futures_list.append(f)

    with patch.object(pipeline.executor, 'submit', side_effect=futures_list):
        # ACT
        detections = [{'class_name': 'section', 'confidence': 0.9}]
        pipeline._handle_detection(
            np.zeros((10,10)), detections, "test_timeout", "dummy_uuid", time.time()
        )

        # ASSERT
        saved_data = mock_external_deps["save_json"].call_args[0][0]
        # Verify it failed
        assert saved_data["qc_summary"] == "FAIL"
        # Verify specific error recorded
        check_result = list(saved_data["criteria"].values())[0]
        assert check_result["pass"] is False

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
        detections = [{'class_name': 'section', 'confidence': 0.9}]
        pipeline._handle_detection(
            np.zeros((10,10)), detections, "test_crash", "dummy_uuid", time.time()
        )

        # ASSERT
        saved_data = mock_external_deps["save_json"].call_args[0][0]
        assert saved_data["qc_summary"] == "FAIL"