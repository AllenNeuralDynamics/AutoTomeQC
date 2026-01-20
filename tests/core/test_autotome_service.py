import pytest
from unittest.mock import patch
from autotomeqc.core.autotome_service import AutoTomeService 

# --- FIXTURES ---

@pytest.fixture
def mock_pipeline_class():
    """
    Patches the AutoTomePipeline class so we don't start the real YOLO engine.
    This mock will replace the 'AutoTomePipeline' used INSIDE autotome_service.py.
    """
    with patch("autotomeqc.core.autotome_service.AutoTomePipeline") as MockClass:
        # The MockClass is the class itself.
        # MockClass.return_value is the instance created when calling AutoTomePipeline()
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
    """Test that calling start() twice doesn't crash or restart the pipeline."""
    service.start()
    # Reset mocks to verify they aren't called again
    service.pipeline.start.reset_mock()
    # Call start again
    service.start()

    # ASSERT
    assert "Service is already running" in caplog.text
    service.pipeline.start.assert_not_called()

def test_stop_service(service, mock_pipeline_class):
    """Test that stop() shuts down the pipeline."""
    service.start()
    service.stop()

    # ASSERT
    assert service.running is False
    service.pipeline.stop.assert_called_once()

def test_process_image_success(service, tmp_path):
    """
    Happy path: Service is running, file exists.
    Uses 'tmp_path' fixture to create a real dummy file.
    """
    service.start()
    dummy_image = tmp_path / "test_image.jpg"
    dummy_image.touch()  # Create an empty file
    service.process(str(dummy_image))

    # ASSERT
    # Check if pipeline.process_image was called.
    service.pipeline.process_image.assert_called_once()

def test_process_fails_if_stopped(service, caplog):
    """Test that processing is blocked if service isn't running."""
    # ACT (No start called and process directly)
    service.process("some/path/image.jpg")

    # ASSERT
    assert "Service is stopped" in caplog.text
    # Since pipeline is None, accessing it would crash if the guard clause failed
    assert service.pipeline is None

def test_process_fails_file_not_found(service, caplog):
    """Test that non-existent files are caught."""
    service.start()
    service.process("non_existent_ghost_file.jpg")

    # ASSERT
    service.pipeline.process_image.assert_not_called()
    assert "File not found" in caplog.text

def test_process_handles_exception(service, tmp_path, caplog):
    """Test that exceptions during processing don't crash the service."""
    # ARRANGE
    service.start()
    dummy_image = tmp_path / "crash_me.jpg"
    dummy_image.touch()
    # Make the pipeline raise an error when processed
    service.pipeline.process_image.side_effect = Exception("YOLO Crashed")
    # ACT
    service.process(str(dummy_image))

    # ASSERT
    assert "Processing Failed: YOLO Crashed" in caplog.text
    # Service should remain running despite the error
    assert service.running is True