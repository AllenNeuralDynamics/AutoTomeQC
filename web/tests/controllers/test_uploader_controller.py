from unittest.mock import patch, MagicMock


class MockQueuedFile:
    """A simple mock for QueuedFile to avoid dependency on the real model."""
    def __init__(self, name):
        self.name = name
        self.path = MagicMock()
        self.json_path = None


def test_load_next_with_items_in_queue(uploader_controller):
    """Test that load_next() cycles through the queue correctly."""
    # 1. Setup: Mock the global app_state and queued files
    mock_app_state = MagicMock()
    mock_app_state.queued_files = {
        "file1": MockQueuedFile("file1.jpg"),
        "file2": MockQueuedFile("file2.jpg"),
        "file3": MockQueuedFile("file3.jpg"),
    }
    
    with patch('autotome_ui.controllers.uploader_controller.app_state', mock_app_state):
        with patch.object(uploader_controller, 'load_result') as mock_load_result:
            # 2. Action & Assertion: From no active file, should go to the first
            mock_app_state.active_file_id = None
            uploader_controller.load_next()
            mock_load_result.assert_called_once_with("file1")
            mock_load_result.reset_mock()

            # 3. Action & Assertion: From the first file, should go to the second
            mock_app_state.active_file_id = "file1"
            uploader_controller.load_next()
            mock_load_result.assert_called_once_with("file2")
            mock_load_result.reset_mock()

            # 4. Action & Assertion: From the last file, should wrap around to the first
            mock_app_state.active_file_id = "file3"
            uploader_controller.load_next()
            mock_load_result.assert_called_once_with("file1")

def test_load_prev_with_items_in_queue(uploader_controller):
    """Test that load_prev() cycles through the queue correctly in reverse."""
    # 1. Setup: Mock the global app_state and queued files
    mock_app_state = MagicMock()
    mock_app_state.queued_files = {
        "file1": MockQueuedFile("file1.jpg"),
        "file2": MockQueuedFile("file2.jpg"),
        "file3": MockQueuedFile("file3.jpg"),
    }
    
    with patch('autotome_ui.controllers.uploader_controller.app_state', mock_app_state):
        with patch.object(uploader_controller, 'load_result') as mock_load_result:
            # 2. Action & Assertion: From no active file, should go to the last
            mock_app_state.active_file_id = None
            uploader_controller.load_prev()
            mock_load_result.assert_called_once_with("file3")
            mock_load_result.reset_mock()

            # 3. Action & Assertion: From the second file, should go to the first
            mock_app_state.active_file_id = "file2"
            uploader_controller.load_prev()
            mock_load_result.assert_called_once_with("file1")
            mock_load_result.reset_mock()

            # 4. Action & Assertion: From the first file, should wrap around to the last
            mock_app_state.active_file_id = "file1"
            uploader_controller.load_prev()
            mock_load_result.assert_called_once_with("file3")

def test_remove_active_file_and_shift(uploader_controller):
    """Test removing the currently active file shifts focus to the next one."""
    # 1. Setup
    mock_app_state = MagicMock()
    mock_file1 = MockQueuedFile("file1.jpg")
    mock_file2 = MockQueuedFile("file2.jpg")
    mock_app_state.queued_files = { "file1": mock_file1, "file2": mock_file2 }
    
    with patch('autotome_ui.controllers.uploader_controller.app_state', mock_app_state):
        with patch.object(uploader_controller, 'load_next') as mock_load_next:
            # 2. Action: Remove the active file
            mock_app_state.active_file_id = "file1"
            uploader_controller.remove_file(["file1"])

            # 3. Assertions
            # It should try to load the next file before deleting
            mock_load_next.assert_called_once()
            assert "file1" not in mock_app_state.queued_files
            mock_file1.path.unlink.assert_called_once_with(missing_ok=True)
            uploader_controller.remove_ui.assert_called_once_with(["file1"])

def test_remove_last_file_goes_idle(uploader_controller):
    """Test removing the last file from the queue sets the view to idle."""
    # 1. Setup
    mock_app_state = MagicMock()
    mock_file1 = MockQueuedFile("file1.jpg")
    mock_app_state.queued_files = {"file1": mock_file1}
    
    with patch('autotome_ui.controllers.uploader_controller.app_state', mock_app_state):
        with patch.object(uploader_controller, '_set_view_state') as mock_set_view_state:
            # 2. Action: Remove the last file
            mock_app_state.active_file_id = "file1"
            uploader_controller.remove_file(["file1"])

            # 3. Assertions
            assert not mock_app_state.queued_files
            # View state should be set to 'idle'
            mock_set_view_state.assert_called_with('idle')
            assert mock_app_state.active_file_id is None