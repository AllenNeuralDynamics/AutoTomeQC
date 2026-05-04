import pytest
from unittest.mock import patch
from autotomeqc.core.autotome_service import AutoTomeService

# --- FIXTURES ---

@pytest.fixture
def mock_pipeline_class():
    """
    Patches the AutoTomePipeline class so we don't start the real YOLO engine.
    """
    with patch("autotomeqc.core.autotome_service.AutoTomePipeline") as MockClass:
        yield MockClass

@pytest.fixture
def service(mock_pipeline_class):
    """Returns a fresh instance of AutoTomeService for each test."""
    return AutoTomeService()

# --- TESTS ---

def test_initial_state(service):
    """Ensure service starts in a clean, stopped state."""
    assert service.running is False
    assert service.pipeline is None

def test_start_service(service, mock_pipeline_class):
    """Test that start() initializes the pipeline and sets running to True."""
    service.start()
    assert service.running is True
    # Verify AutoTomePipeline() was instantiated
    mock_pipeline_class.assert_called_once()
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
