import pytest
from unittest.mock import MagicMock

from web.controllers.uploader_controller import UploaderController


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