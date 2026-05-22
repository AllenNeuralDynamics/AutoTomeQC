import pytest
from unittest.mock import patch, MagicMock
from autotomeqc.core.autotome_service import AutoTomeService
from autotomeqc.config.schemas import AppConfig

# --- FIXTURES ---

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
            "yolo_post_processing": {"out_dim": [1,1], "loop_bbox_margin": 0, "allow_no_loop": True, "overlap_threshold": 0.5},
            "section_coverage": {"weights_path": "d", "img_size": 1, "img_dim": [1,1], "pass_labels": [], "min_confidence": 0.5},
            "knife_mark": {"weights_path": "d", "img_size": 1, "img_dim": [1,1], "pass_labels": [], "min_confidence": 0.5},
            "thickness_consistency": {"weights_path": "d", "img_size": 1, "img_dim": [1,1], "pass_labels": [], "min_confidence": 0.5},
            "thickness": {"weights_path": "d", "img_size": 1, "img_dim": [1,1], "pass_labels": [], "min_confidence": 0.5},
            "shape": {"save_debug_img": False}
        }
    }

@pytest.fixture
def mock_pipeline_class():
    """
    Patches the AutoTomePipeline class so we don't start the real YOLO engine.
    """
    with patch("autotomeqc.core.autotome_service.AutoTomePipeline") as MockClass:
        mock_instance = MockClass.return_value
        mock_instance.start.return_value = True
        yield MockClass

@pytest.fixture
def service(mock_pipeline_class, mock_config):
    """Returns a fresh instance of AutoTomeService for each test."""
    config = AppConfig(**mock_config)
    return AutoTomeService(config=config)

# --- TESTS ---

def test_initial_state(service):
    """Ensure service starts in a clean, stopped state."""
    assert service.running is False
    assert service.pipeline is None
    assert service.config is not None

def test_start_service(service, mock_pipeline_class):
    """Test that start() initializes the pipeline and sets running to True."""
    service.start()
    assert service.running is True
    # Verify AutoTomePipeline() was instantiated with config
    mock_pipeline_class.assert_called_once_with(config=service.config)
    # Verify pipeline.start() was called on the instance
    service.pipeline.start.assert_called_once()

def test_start_twice(service, caplog):
    """Test that calling start() twice logs a warning and doesn't restart."""
    service.start()
    service.pipeline.start.reset_mock()

    # Act
    service.start()

    # Assert
    assert "Service is already running" in caplog.text
    service.pipeline.start.assert_not_called()

def test_stop_service(service, mock_pipeline_class):
    """Test that stop() shuts down the pipeline."""
    service.start()
    pipeline_mock = service.pipeline
    service.stop()

    # Assert
    assert service.running is False
    pipeline_mock.stop.assert_called_once()

def test_process_image_success(service, tmp_path):
    """
    Happy path: Service is running, file exists.
    """
    service.start()
    dummy_image = tmp_path / "test_image.jpg"
    dummy_image.touch()

    # Act
    service.process(str(dummy_image))

    # Assert
    # Note: Matches the call in your Service code: self.pipeline.process()
    # If your Pipeline actually uses .process_image(), update the Service code and this assertion.
    service.pipeline.process.assert_called_once()

def test_process_fails_if_stopped(service):
    """Test that processing raises RuntimeError if service isn't running."""
    # Act & Assert
    # The service RAISES RuntimeError, it does not log it.
    with pytest.raises(RuntimeError, match="Service is stopped"):
        service.process("some/path/image.jpg")
