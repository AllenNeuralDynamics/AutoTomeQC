import pytest
import numpy as np
from unittest.mock import patch
from autotomeqc.yolo_segmentation.post_processing import (
    get_overlap_ratio, 
    validate_detections, 
    cropped_segmented
)

# Mock CONFIG to avoid file I/O during tests
MOCK_CONFIG = {
    "qc": {
        "yolo_post_processing": {
            "out_dim": [640, 640],
            "allow_no_loop": True,
            "loop_bbox_margin": 10
        }
    }
}

@pytest.fixture
def sample_detections():
    """Generates a standard set of detections: one loop, one section inside."""
    return [
        {
            'class_name': 'loop',
            'confidence': 0.9,
            'bbox': [100, 100, 400, 400],
            'mask': [[100, 100], [400, 100], [400, 400], [100, 400]]
        },
        {
            'class_name': 'section',
            'confidence': 0.85,
            'bbox': [150, 150, 250, 250],
            'mask': [[150, 150], [250, 150], [250, 250], [150, 250]]
        }
    ]

@patch("autotomeqc.yolo_segmentation.post_processing.CONFIG", MOCK_CONFIG)
class TestYoloPostProcessing:

    def test_get_overlap_ratio_full_overlap(self):
        # Two identical squares
        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        bbox = [0, 0, 10, 10]
        ratio = get_overlap_ratio(poly, poly, bbox, bbox)
        assert ratio == 1.0

    def test_get_overlap_ratio_no_overlap(self):
        poly_s = [[0, 0], [2, 0], [2, 2], [0, 2]]
        bbox_s = [0, 0, 2, 2]
        poly_l = [[10, 10], [12, 10], [12, 12], [10, 12]]
        bbox_l = [10, 10, 12, 12]
        ratio = get_overlap_ratio(poly_s, poly_l, bbox_s, bbox_l)
        assert ratio == 0.0

    def test_get_overlap_ratio_partial_overlap(self):
        """
        Tests 50% overlap using pixel-aware coordinates.
        Section: 0 to 9 (10 pixels wide) -> Area 100
        Loop: 5 to 14 (10 pixels wide)
        Overlap: 5 to 9 (5 pixels wide) -> Area 50
        Result: 50/100 = 0.5
        """
        # Section Square (10x10 pixels)
        poly_s = [[0, 0], [9, 0], [9, 9], [0, 9]]
        bbox_s = [0, 0, 9, 9]
        # Loop Square overlapping exactly half (5 pixels)
        poly_l = [[5, 0], [14, 0], [14, 9], [5, 9]]
        bbox_l = [5, 0, 14, 9]
        ratio = get_overlap_ratio(poly_s, poly_l, bbox_s, bbox_l)

        # This should now result in exactly 0.5
        assert ratio == pytest.approx(0.5, abs=1e-2)

    def test_validate_detections_success(self, sample_detections):
        is_valid, reason, filtered = validate_detections(sample_detections)
        assert is_valid is True
        assert len(filtered) == 2
        assert reason is None

    def test_validate_detections_no_section(self):
        detections = [{'class_name': 'loop', 'bbox': [0,0,10,10], 'mask': []}]
        is_valid, reason, filtered = validate_detections(detections)
        assert is_valid is False
        assert "No section" in reason

    def test_validate_detections_outside_loop(self):
        detections = [
            {'class_name': 'loop', 'bbox': [0, 0, 10, 10], 'mask': [[0,0], [10,0], [10,10], [0,10]]},
            {'class_name': 'section', 'bbox': [50, 50, 60, 60], 'mask': [[50,50], [60,50], [60,60], [50,60]]}
        ]
        is_valid, reason, filtered = validate_detections(detections)
        assert is_valid is False
        assert "outside loop" in reason

    def test_cropped_segmented_output_shape(self, sample_detections):
        # Create a dummy 1000x1000 image
        dummy_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
        result = cropped_segmented(dummy_frame, sample_detections)

        assert result is not None
        # Check if it resized to CONFIG dimensions (640, 640)
        assert result.shape == (640, 640, 3)
        assert isinstance(result, np.ndarray)

    def test_cropped_segmented_masking_applied(self):
        # Ensure that areas outside the section mask are blacked out
        dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections = [
            {
                'class_name': 'section',
                'confidence': 0.9,
                'mask': [[10, 10], [20, 10], [20, 20], [10, 20]],
                'bbox': [10, 10, 20, 20]
            }
        ]
        # We expect the resulting image to have some black pixels now
        result = cropped_segmented(dummy_frame, detections)
        # Check corner (should be blacked out)
        assert np.all(result[0, 0] == 0)