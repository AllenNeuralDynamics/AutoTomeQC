import pytest
import numpy as np
import time
from unittest.mock import patch
from concurrent.futures import TimeoutError, Future

# Import the class and config
from autotomeqc.core.pipeline import AutoTomePipeline
from autotomeqc.config.schemas import AppConfig

# --- FIXTURES ---

@pytest.fixture
def mock_config():
    return {
        "qc": {
            "fps": 1, 
            "output_dir": "example/output",
            "save_segmented_images": True,
            "save_input_images": False,
            "yolo": {
                "conf_thresh": 0.9, "img_dim": [960, 960], 
                "img_size": 960, "iou_thresh": 0.45, "weights_path": "dummy.pt",
                "max_det": 30
            },
            "yolo_post_processing": {
                "out_dim": [640, 640],
                "loop_bbox_margin": 30,
                "allow_no_loop": True
            },
            "section_coverage": {"weights_path": "d", "img_size": 224, "pass_labels": ["full"], "min_confidence": 0.5},
            "knife_mark": {"weights_path": "d", "img_size": 640, "pass_labels": ["none"], "min_confidence": 0.5},
            "thickness_consistency": {"weights_path": "d", "img_size": 224, "pass_labels": ["consistent"], "min_confidence": 0.5},
            "thickness": {"weights_path": "d", "img_size": 224, "pass_labels": ["ANY"], "min_confidence": 0.5},
            "shape": {"save_debug_img": True}
        }
    }

@pytest.fixture
def mock_external_deps():
    with patch("autotomeqc.core.pipeline.cv2") as mock_cv2, \
         patch("autotomeqc.core.pipeline.save_json_results") as mock_save_json, \
         patch("autotomeqc.core.pipeline.save_debug_image") as mock_save_img, \
         patch("autotomeqc.core.pipeline.cropped_segmented") as mock_crop:
        
        # Ensure cv2.imread returns something so the pipeline doesn't bail
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock cropped_segmented to return the detections passed to it
        mock_crop.side_effect = lambda frame, dets: dets
        
        yield {
            "cv2": mock_cv2,
            "save_json": mock_save_json,
            "save_img": mock_save_img,
            "crop": mock_crop,
        }

@pytest.fixture
def pipeline(mock_config):
    with patch("autotomeqc.core.pipeline.YoloSegmentation"), \
         patch("autotomeqc.core.pipeline.SectionCoverageQC"), \
         patch("autotomeqc.core.pipeline.KnifeMarksQC"), \
         patch("autotomeqc.core.pipeline.ThicknessConsistencyQC"), \
         patch("autotomeqc.core.pipeline.ThicknessQC"), \
         patch("autotomeqc.core.pipeline.ShapeQC"):
        
        test_config_obj = AppConfig(**mock_config)
        with patch("autotomeqc.core.pipeline.CONFIG", test_config_obj):
            pipe = AutoTomePipeline()
            yield pipe
            pipe.stop()

# --- TESTS ---

def test_process_flow(pipeline, mock_external_deps, tmp_path):
    """Test the happy path from process() -> YOLO."""
    # ARRANGE: Create a dummy file so the path exists check passes
    pipeline.start()  # Start the pipeline so the worker thread runs
    image_path = tmp_path / "sample.jpg"
    image_path.write_text("dummy data") 
    
    # ACT
    future = pipeline.process(img_path=str(image_path))
    result = future.result(timeout=2.0) # Wait for the worker thread to finish the task
    
    # ASSERT
    assert result["qc_summary"] in ["PASS", "FAIL"]
    pipeline.segmenter.process_frame.assert_called_once()

    # ACT
    pipeline.stop()

    # ASSERT
    assert pipeline.is_running is False
    assert pipeline.worker_thread is None  # Verify worker thread has been cleaned up

def test_handle_pipeline_valid_input_runs_qc(pipeline, mock_external_deps):
    """Test that valid detection triggers QC and JSON saving."""
    # ARRANGE
    future_ticket = Future()
    timestamp = "2026-04-07 12:00:00"
    start_ts = time.time()
    
    # Simulate the QC modules detecting a problem (Blank/No Signal)
    for mod in pipeline.qc_modules.values():
        mod.check.return_value = {
            "pass_status": False, 
            "label": "FAIL", 
            "message": "Low signal/Blank image detected"
        }

    # Create a dummy detection from yolo segmentation
    detections = [{
        'class_name': 'section',
        'confidence': 0.9,
        'section_image': np.zeros((100, 100, 3), dtype=np.uint8),
        'area_in_pixels': 5000,
        'overlap_ratio': 0.0
    }]
    
    # ACT
    pipeline._handle_pipeline_valid_input(
        frame=np.zeros((640, 640, 3), dtype=np.uint8),
        detections=detections,
        filename="test_file",
        timestamp=timestamp,
        start_ts=start_ts,
        future_ticket=future_ticket
    )

    # ASSERT
    # 1. Verify IO was called
    mock_external_deps["save_json"].assert_called_once()
    
    # 2. Verify the Future was resolved
    assert future_ticket.done()
    result = future_ticket.result()
    
    # 3. Verify the logic
    assert result["qc_summary"] == "FAIL"
    assert len(result["sections"]) == 1
    assert "Section failed QC criteria" in result["fail_reason"]

def test_qc_timeout_handling(pipeline, mock_external_deps):
    """Simulate a QC check timing out inside the sequential loop."""
    # ARRANGE
    future_ticket = Future()
    # This simulates a module that internally timed out or crashed
    pipeline.qc_modules["coverage"].check.side_effect = TimeoutError("QC check exceeded 2.0s limit")

    detections = [{
        'class_name': 'section',
        'confidence': 0.9,
        'section_image': np.zeros((100, 100, 3), dtype=np.uint8),
        'area_in_pixels': 5000,
        'overlap_ratio': 0.0
    }]

    # ACT
    pipeline._handle_pipeline_valid_input(
        frame=np.zeros((640, 640, 3), dtype=np.uint8),
        detections=detections,
        filename="timeout_test",
        timestamp="2026-04-07 12:00:00",
        start_ts=time.time(),
        future_ticket=future_ticket
    )

    # ASSERT
    result = future_ticket.result()
    assert result["qc_summary"] == "FAIL"
    
    # Verify the error message was caught
    coverage_res = result["sections"][0]["criteria"]["coverage"]
    assert coverage_res["pass_status"] is False
    assert "exceeded 2.0s limit" in coverage_res["message"]


def test_qc_exception_handling(pipeline, mock_external_deps):
    """Test that individual module crashes are caught by the loop and report FAIL."""
    # ARRANGE
    future_ticket = Future()
    
    # Force a crash in one of the specific modules (e.g., 'coverage')
    pipeline.qc_modules["coverage"].check.side_effect = ValueError("Math Error")

    detections = [{
        'class_name': 'section',
        'confidence': 0.9,
        'section_image': np.zeros((100, 100, 3), dtype=np.uint8),
        'area_in_pixels': 5000,
        'overlap_ratio': 0.0
    }]

    # ACT
    # We call the handler that orchestrates the QC modules
    pipeline._handle_pipeline_valid_input(
        frame=np.zeros((640, 640, 3), dtype=np.uint8),
        detections=detections,
        filename="crash_test",
        timestamp="2026-04-07 12:00:00",
        start_ts=time.time(),
        future_ticket=future_ticket
    )

    # ASSERT
    # Verify the pipeline caught the error and saved the JSON
    assert mock_external_deps["save_json"].called

    # Verify the result was resolved via the future
    result = future_ticket.result()
    assert result["qc_summary"] == "FAIL"

    # Verify the specific module caught the error as an "Error" label
    coverage_criteria = result["sections"][0]["criteria"]["coverage"]
    assert coverage_criteria["pass_status"] is False
    assert "Math Error" in coverage_criteria["message"]

def test_process_invalid_input_both_provided(pipeline, mock_external_deps):
    """Test that providing both img_path and frame raises/returns an error."""
    # ACT
    future = pipeline.process(img_path="some_path.jpg", frame=np.zeros((10, 10)))
    result = future.result()

    # ASSERT
    assert result["qc_summary"] == "FAIL"
    assert "Ambiguous input" in result["fail_reason"]
    # Ensure no processing was actually attempted
    pipeline.segmenter.process_frame.assert_not_called()

def test_process_invalid_input_none_provided(pipeline):
    """Test that providing neither img_path nor frame fails gracefully."""
    # ACT
    future = pipeline.process(img_path=None, frame=None)
    result = future.result()

    # ASSERT
    assert result["qc_summary"] == "FAIL"
    assert "Ambiguous input" in result["fail_reason"]

def test_process_nonexistent_file(pipeline, tmp_path):
    """Test handling of a file path that does not exist on disk."""
    # ARRANGE
    bad_path = tmp_path / "ghost_image.jpg" # Not created

    # ACT
    future = pipeline.process(img_path=str(bad_path))
    result = future.result()

    # ASSERT
    assert result["qc_summary"] == "FAIL"
    assert "File not found" in result["fail_reason"]

def test_process_corrupt_image_file(pipeline, tmp_path, mock_external_deps):
    """Test behavior when cv2.imread fails to decode the file."""
    # ARRANGE
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_text("not an image")
    
    # Force cv2.imread to return None (simulating a corrupt/invalid image)
    mock_external_deps["cv2"].imread.return_value = None

    # ACT
    future = pipeline.process(img_path=str(corrupt_path))
    result = future.result()

    # ASSERT
    assert result["qc_summary"] == "FAIL"
    assert "File load failed" in result["fail_reason"]

def test_process_raw_frame_success(pipeline, mock_external_deps):
    """Test that passing a numpy frame directly works correctly."""
    # ARRANGE
    pipeline.start() 
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # ACT
    future = pipeline.process(frame=frame)
    future.result(timeout=2.0)

    # ASSERT
    # Accessing the segmenter instance from the mock created in the fixture
    pipeline.segmenter.process_frame.assert_called_once()
    pipeline.stop()
