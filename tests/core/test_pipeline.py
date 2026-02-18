import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock
from concurrent.futures import TimeoutError, Future
from pathlib import Path

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
            # Default mock behavior for QC modules
            for mod in pipe.qc_modules.values():
                mod.check.return_value = {"pass_status": True, "label": "Pass", "conf": 0.9}
            yield pipe
            pipe.stop()

# --- TESTS ---

def test_process_flow(pipeline, mock_external_deps, tmp_path):
    """Test the happy path from process() -> YOLO."""
    # ARRANGE: Create a dummy file so the path exists check passes
    image_path = tmp_path / "sample.jpg"
    image_path.write_text("dummy data") 
    
    # ACT
    pipeline.process(img_path=str(image_path))
    
    # ASSERT
    mock_external_deps["cv2"].imread.assert_called_once()
    pipeline.segmenter.process_frame.assert_called_once()

def test_handle_detection_runs_qc(pipeline, mock_external_deps):
    """Test that valid detection triggers QC and JSON saving."""
    # ARRANGE: Pre-register the ID so the pipeline accepts it
    dummy_id = "dummy_uuid"
    pipeline.pending_results[dummy_id] = Future()
    
    # ACT
    detections = [{'class_name': 'section', 'confidence': 0.9, 'section_image': np.zeros((5,5))}]
    pipeline._handle_detection(np.zeros((10,10)), detections, "test_file", dummy_id, time.time())

    # ASSERT
    mock_external_deps["save_json"].assert_called_once()
    saved_data = mock_external_deps["save_json"].call_args[0][0]
    assert saved_data["qc_summary"] == "PASS"

def test_qc_timeout_handling(pipeline, mock_external_deps):
    """Simulate a QC check timing out."""
    # ARRANGE
    dummy_id = "timeout_uuid"
    pipeline.pending_results[dummy_id] = Future()
    
    futures_list = []
    for _ in range(5):
        f = Future()
        f.set_exception(TimeoutError("QC check exceeded 2.0s limit"))
        futures_list.append(f)

    with patch.object(pipeline.executor, 'submit', side_effect=futures_list):
        # ACT
        detections = [{'class_name': 'section', 'confidence': 0.9, 'section_image': np.zeros((5,5))}]
        pipeline._handle_detection(np.zeros((10,10)), detections, "test_file", dummy_id, time.time())

        # ASSERT
        mock_external_deps["save_json"].assert_called_once()
        saved_data = mock_external_deps["save_json"].call_args[0][0]
        assert saved_data["qc_summary"] == "FAIL"
        
        # Check if the error message is captured in any of the criteria
        criteria_vals = saved_data["sections"][0]["criteria"].values()
        assert any("exceeded 2.0s limit" in str(c.get("message")) for c in criteria_vals)

def test_qc_exception_handling(pipeline, mock_external_deps):
    """Test worker thread crashes are caught."""
    # ARRANGE
    dummy_id = "crash_uuid"
    pipeline.pending_results[dummy_id] = Future()
    
    futures_list = [Future() for _ in range(5)]
    for f in futures_list:
        f.set_exception(ValueError("Math Error"))

    with patch.object(pipeline.executor, 'submit', side_effect=futures_list):
        # ACT
        detections = [{'class_name': 'section', 'confidence': 0.9, 'section_image': np.zeros((5,5))}]
        pipeline._handle_detection(np.zeros((10,10)), detections, "test_file", dummy_id, time.time())

        # ASSERT
        assert mock_external_deps["save_json"].called
        saved_data = mock_external_deps["save_json"].call_args[0][0]
        assert saved_data["qc_summary"] == "FAIL"

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
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # ACT
    future = pipeline.process(frame=frame)
    
    # ASSERT
    # Check that it generated a timestamped filename (since none was provided)
    pipeline.segmenter.process_frame.assert_called_once()
    args, kwargs = pipeline.segmenter.process_frame.call_args
    assert "frame_" in kwargs["filename"] or len(kwargs["filename"]) > 0