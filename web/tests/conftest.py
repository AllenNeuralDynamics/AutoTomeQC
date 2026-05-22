import pytest
from nicegui import app
from unittest.mock import MagicMock, patch, AsyncMock

from autotome_ui.controllers.uploader_controller import UploaderController


@pytest.fixture
def mock_callbacks():
    """Provides a dictionary of mock callbacks for the UploaderController."""
    return {
        "add_ui_callback": MagicMock(),
        "remove_ui_callback": MagicMock(),
        "set_active_ui_callback": MagicMock(),
        "refresh_workspace": MagicMock(),
        "refresh_inspector": MagicMock(),
    }

@pytest.fixture
def uploader_controller(mock_callbacks):
    """Provides an UploaderController instance with mocked dependencies."""
    return UploaderController(**mock_callbacks)

@pytest.fixture(autouse=True)
def mock_backend_api_calls():
    """
    Automatically mock functions that try to reach the backend API,
    so that UI tests can run in isolation without a live backend.
    """
    # These mocks prevent the app from hanging on startup while waiting for the backend.
    with patch('autotome_ui.controllers.state_controller.wait_backend_ready', new_callable=AsyncMock), \
         patch('autotome_ui.controllers.state_controller.on_fetch_config', new_callable=AsyncMock):
        yield

@pytest.fixture(autouse=True)
def reset_nicegui():
    """Reset NiceGUI state after each test to prevent teardown errors."""
    yield
    app.storage.clear()
