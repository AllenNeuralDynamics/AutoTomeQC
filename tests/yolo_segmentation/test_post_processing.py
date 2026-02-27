import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from autotomeqc.yolo_segmentation.post_processing import (
    cropped_segmented, 
    get_overlap_ratio, 
    validate_detections
)

# --- GLOBAL FIXTURES ---

@pytest.fixture
def mock_config():
    """Provides a MagicMock that supports dot-notation access for CONFIG."""
    config = MagicMock()
    post_proc = config.qc.yolo_post_processing
    post_proc.out_dim = [640, 640]
    post_proc.loop_bbox_margin = 30
    post_proc.allow_no_loop = True
    post_proc.overlap_threshold = 0.5
    return config

class TestYoloPostProcessing:

    @pytest.fixture
    def sample_detections(self):
        """Standard detections for image processing tests."""
        return [
            {
                'bbox': [100, 100, 400, 400], 
                'class_name': 'loop', 
                'mask': [[100, 100], [400, 100], [400, 400], [100, 400]]
            },
            {
                'bbox': [150, 150, 250, 250], 
                'class_name': 'section', 
                'mask': [[150, 150], [250, 150], [250, 250], [150, 250]],
                'area_in_pixels': 10000
            }
        ]

    # --- SECTION 1: GEOMETRIC & OVERLAP TESTS ---

    def test_get_overlap_ratio_perfect_alignment(self, mock_config):
        """Verify 100% overlap detection."""
        loop_poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        section_poly = [[25, 25], [75, 25], [75, 75], [25, 75]]
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            ratio = get_overlap_ratio(section_poly, loop_poly, [25, 25, 75, 75], [0, 0, 100, 100])
            assert ratio == 1.0

    def test_get_overlap_ratio_partial(self, mock_config):
        """Verify 50% overlap detection."""
        loop_poly = [[0, 0], [50, 0], [50, 100], [0, 100]]
        section_poly = [[25, 0], [75, 0], [75, 100], [25, 100]]
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            ratio = get_overlap_ratio(section_poly, loop_poly, [25, 0, 75, 100], [0, 0, 50, 100])
            assert pytest.approx(ratio, 0.05) == 0.5

    def test_get_overlap_ratio_no_bbox_overlap(self):
        """Verify cheap BBox check returns 0 immediately if boxes don't touch."""
        loop_bbox = [0, 0, 10, 10]
        section_bbox = [20, 20, 30, 30]
        ratio = get_overlap_ratio([], [], section_bbox, loop_bbox)
        assert ratio == 0.0

    # --- SECTION 2: BUSINESS LOGIC TESTS (CASES 1-5 in validate_detections) ---

    def test_validate_detections_case_1_no_section(self, mock_config):
        """Case 1: No section in frame (FAIL)."""
        detections = [{'class_name': 'loop', 'mask': [[0,0]], 'bbox': [0,0,10,10]}]
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            is_valid, msg, _ = validate_detections(detections)
            assert not is_valid
            assert "No section detected" in msg

    def test_validate_detections_case_2_no_loop(self, mock_config):
        """Case 2: No loop found. Logic depends on 'allow_no_loop' config."""
        detections = [{'class_name': 'section', 'mask': [[0,0]], 'bbox': [0,0,10,10]}]
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            # Branch A: Allowed (PASS)
            mock_config.qc.yolo_post_processing.allow_no_loop = True
            is_valid, _, _ = validate_detections(detections)
            assert is_valid
            # Branch B: Not allowed (FAIL)
            mock_config.qc.yolo_post_processing.allow_no_loop = False
            is_valid, msg, _ = validate_detections(detections)
            assert not is_valid
            assert "No loop detected" in msg

    def test_validate_detections_case_3_outside_loop(self, mock_config):
        """Case 3: Section exists but does not overlap with loop (FAIL)."""
        detections = [
            {'class_name': 'loop', 'mask': [[0,0], [10,0], [10,10], [0,10]], 'bbox': [0,0,10,10]},
            {'class_name': 'section', 'mask': [[20,20], [30,20], [30,30], [20,30]], 'bbox': [20,20,30,30]}
        ]
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            is_valid, msg, _ = validate_detections(detections)
            assert not is_valid
            assert "outside loop" in msg

    def test_validate_detections_case_4_multiple_sections(self, mock_config):
        """Case 4: Multiple sections in loop (WARNING/PASS)."""
        detections = [
            {'class_name': 'loop', 'mask': [[0,0], [100,0], [100,100], [0,100]], 'bbox': [0,0,100,100]},
            {'class_name': 'section', 'mask': [[10,10]], 'bbox': [10,10,20,20]},
            {'class_name': 'section', 'mask': [[40,40]], 'bbox': [40,40,50,50]}
        ]
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            is_valid, msg, _ = validate_detections(detections)
            assert is_valid  # Code currently allows this but warns
            assert "Multiple sections (2)" in msg

    def test_validate_detections_case_5_success(self, mock_config):
        """Case 5: Exactly one section inside the loop (HAPPY PATH)."""
        detections = [
            {'class_name': 'loop', 'mask': [[0,0], [100,0], [100,100], [0,100]], 'bbox': [0,0,100,100]},
            {'class_name': 'section', 'mask': [[25,25]], 'bbox': [25,25,75,75]}
        ]
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            is_valid, msg, filtered = validate_detections(detections)
            assert is_valid
            assert msg == "N/A"
            assert len(filtered) == 2

    # --- SECTION 3: IMAGE PROCESSING & CROPPING ---

    def test_cropped_segmented_output_shape(self, sample_detections, mock_config):
        """Ensure the resulting section_image is standardized to out_dim."""
        dummy_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            result_list = cropped_segmented(dummy_frame, sample_detections)
            section_det = next((d for d in result_list if d['class_name'] == 'section'), None)
            assert section_det['section_image'].shape == (640, 640, 3)

    def test_cropped_segmented_masking_applied(self, mock_config):
        """Ensure pixels outside the section mask are blacked out."""
        # Create a white frame
        dummy_frame = (np.ones((200, 200, 3), dtype=np.uint8) * 255)
        detections = [{
            'class_name': 'section',
            'bbox': [50, 50, 150, 150],
            'mask': [[80, 80], [120, 80], [120, 120], [80, 120]] # Small square in center
        }]
        with patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", mock_config):
            result = cropped_segmented(dummy_frame, detections)
            img = result[0]['section_image']
            # Corner of standardized image should be black
            assert np.all(img[0, 0] == 0)
            # Center of standardized image should be white
            assert np.all(img[320, 320] == 255)