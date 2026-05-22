import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from autotomeqc.yolo_segmentation.post_processing import YoloPostProcessor 
from autotomeqc.core.models import Detection
from autotomeqc.config.schemas import AppConfig

# --- GLOBAL FIXTURES ---

@pytest.fixture
def mock_config():
    """Provides a minimal mock config that can be parsed by AppConfig."""
    return {
        "qc": {
            "output_dir": "test_output",
            "save_qc_json": False,
            "save_segmented_images": False,
            "save_input_images": False,
            "return_mask_data": False,
            "yolo": {"weights_path": "d", "img_dim": [1,1], "img_size": 1, "conf_thresh": 0.5, "max_det": 1},
            "yolo_post_processing": {"out_dim": [640, 640], "loop_bbox_margin": 0, "allow_no_loop": True, "overlap_threshold": 0.5},
            "section_coverage": {"weights_path": "d", "img_size": 1, "img_dim": [1,1], "pass_labels": [], "min_confidence": 0.5},
            "knife_mark": {"weights_path": "d", "img_size": 1, "img_dim": [1,1], "pass_labels": [], "min_confidence": 0.5},
            "thickness_consistency": {"weights_path": "d", "img_size": 1, "img_dim": [1,1], "pass_labels": [], "min_confidence": 0.5},
            "thickness": {"weights_path": "d", "img_size": 1, "img_dim": [1,1], "pass_labels": [], "min_confidence": 0.5},
            "shape": {"save_debug_img": False}
        }
    }

@pytest.fixture
def processor(mock_config):
    config_obj = AppConfig(**mock_config)
    return YoloPostProcessor(config=config_obj.qc.yolo_post_processing)

class TestYoloPostProcessing:

    @pytest.fixture
    def sample_detections(self):
        """Standard detections for image processing tests."""
        return [
            Detection(
                bbox=[100, 100, 400, 400], 
                class_name='loop', 
                class_id=1,
                confidence=0.9,
                mask=[[100, 100], [400, 100], [400, 400], [100, 400]]
            ),
            Detection(
                bbox=[150, 150, 250, 250], 
                class_name='section', 
                class_id=0,
                confidence=0.9,
                mask=[[150, 150], [250, 150], [250, 250], [150, 250]],
                area_in_pixels=10000
            )
        ]

    # --- SECTION 1: GEOMETRIC & OVERLAP TESTS ---

    def test_get_overlap_ratio_perfect_alignment(self, processor):
        """Verify 100% overlap detection."""
        loop_poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        section_poly = [[25, 25], [75, 25], [75, 75], [25, 75]]
        # Call the method on the instance
        ratio = processor.get_overlap_ratio(section_poly, loop_poly, [25, 25, 75, 75], [0, 0, 100, 100])
        assert ratio == 1.0

    def test_get_overlap_ratio_partial(self, processor):
        """Verify 50% overlap detection."""
        loop_poly = [[0, 0], [50, 0], [50, 100], [0, 100]]
        section_poly = [[25, 0], [75, 0], [75, 100], [25, 100]]
        ratio = processor.get_overlap_ratio(section_poly, loop_poly, [25, 0, 75, 100], [0, 0, 50, 100])
        assert pytest.approx(ratio, 0.05) == 0.5

    def test_get_overlap_ratio_no_bbox_overlap(self, processor):
        """Verify cheap BBox check returns 0 immediately if boxes don't touch."""
        loop_bbox = [0, 0, 10, 10]
        section_bbox = [20, 20, 30, 30]
        ratio = processor.get_overlap_ratio([], [], section_bbox, loop_bbox)
        assert ratio == 0.0

    # --- SECTION 2: BUSINESS LOGIC TESTS (CASES 1-5 in validate_detections) ---

    def test_validate_detections_case_1_no_section(self, processor):
        """Case 1: No section in frame (FAIL)."""
        detections = [Detection(
            class_name='loop', class_id=1, confidence=0.9, mask=[[0.0,0.0]], bbox=[0.0,0.0,10.0,10.0]
        )]
        is_valid, msg, _ = processor.validate_detections(detections)
        assert not is_valid
        assert "No section detected" in msg

    def test_validate_detections_case_2_no_loop(self, processor):
        """Case 2: No loop found. Logic depends on 'allow_no_loop' config."""
        detections = [Detection(
            class_name='section', class_id=0, confidence=0.9, mask=[[0.0,0.0]], bbox=[0.0,0.0,10.0,10.0]
        )]
        # Branch A: Allowed (PASS)
        processor.config.allow_no_loop = True
        is_valid, _, _ = processor.validate_detections(detections)
        assert is_valid
        # Branch B: Not allowed (FAIL)
        processor.config.allow_no_loop = False
        is_valid, msg, _ = processor.validate_detections(detections)
        assert not is_valid
        assert "No loop detected" in msg

    def test_validate_detections_case_3_outside_loop(self, processor):
        """Case 3: Section exists but does not overlap with loop (FAIL)."""
        detections = [
            Detection(class_name='loop', class_id=1, confidence=0.9, 
                      mask=[[0.0,0.0], [10.0,0.0], [10.0,10.0], [0.0,10.0]], 
                      bbox=[0.0,0.0,10.0,10.0]),
            Detection(class_name='section', class_id=0, confidence=0.9, 
                      mask=[[20.0,20.0], [30.0,20.0], [30.0,30.0], [20.0,30.0]], 
                      bbox=[20.0,20.0,30.0,30.0])
        ]
        is_valid, msg, _ = processor.validate_detections(detections)
        assert not is_valid
        assert "outside loop" in msg

    def test_validate_detections_case_4_multiple_sections(self, processor):
        """Case 4: Multiple sections in loop (WARNING/PASS)."""
        detections = [
            Detection(class_name='loop', class_id=1, confidence=0.9, 
                      mask=[[0.0,0.0], [100.0,0.0], [100.0,100.0], [0.0,100.0]], 
                      bbox=[0.0,0.0,100.0,100.0]),
            Detection(class_name='section', class_id=0, confidence=0.9, mask=[[10.0,10.0]], bbox=[10.0,10.0,20.0,20.0]),
            Detection(class_name='section', class_id=0, confidence=0.9, mask=[[40.0,40.0]], bbox=[40.0,40.0,50.0,50.0])
        ]
        is_valid, msg, _ = processor.validate_detections(detections)
        assert is_valid  # Code currently allows this but warns
        assert "Multiple sections (2)" in msg

    def test_validate_detections_case_5_success(self, processor):
        """Case 5: Exactly one section inside the loop (HAPPY PATH)."""
        detections = [
            Detection(class_name='loop', class_id=1, confidence=0.9, 
                      mask=[[0.0,0.0], [100.0,0.0], [100.0,100.0], [0.0,100.0]], 
                      bbox=[0.0,0.0,100.0,100.0]),
            Detection(class_name='section', class_id=0, confidence=0.9, mask=[[25.0,25.0]], bbox=[25.0,25.0,75.0,75.0])
        ]
        is_valid, msg, filtered = processor.validate_detections(detections)
        assert is_valid
        assert msg == "N/A"
        assert len(filtered) == 2

    # --- SECTION 3: IMAGE PROCESSING & CROPPING ---

    def test_cropped_segmented_output_shape(self, processor, sample_detections):
        """Ensure the resulting section_image is standardized to out_dim."""
        dummy_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        result_list = processor.cropped_segmented(dummy_frame, sample_detections)
        section_det = next((d for d in result_list if d.class_name == 'section'), None)
        assert section_det.section_image.shape == (640, 640, 3)

    def test_cropped_segmented_masking_applied(self, processor):
        """Ensure pixels outside the section mask are blacked out."""
        # Create a white frame
        dummy_frame = (np.ones((200, 200, 3), dtype=np.uint8) * 255)
        detections = [Detection(
            class_name='section',
            class_id=0,
            confidence=0.9,
            bbox=[50.0, 50.0, 150.0, 150.0],
            mask=[[80.0, 80.0], [120.0, 80.0], [120.0, 120.0], [80.0, 120.0]] # Small square in center
        )]
        result = processor.cropped_segmented(dummy_frame, detections)
        img = result[0].section_image
        # Corner of standardized image should be black
        assert np.all(img[0, 0] == 0)
        # Center of standardized image should be white
        assert np.all(img[320, 320] == 255)