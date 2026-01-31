import pytest
import logging  # <--- Added import
from unittest.mock import MagicMock, patch

# Import the function to test
from autotomeqc.interface.cli import run_interactive_cli

# --- FIXTURES ---

@pytest.fixture
def mock_service_class():
    # Patch the class where it is IMPORTED, not where it is defined
    with patch("autotomeqc.interface.cli.AutoTomeService") as mock:
        yield mock

# --- TESTS ---

def test_interactive_mode_happy_path(mock_service_class, caplog):
    """
    Scenario: Human user runs app, types a filename, then types 'exit'.
    Checks: Banner logs, start(), process(), and stop().
    """
    caplog.set_level(logging.INFO)
    # ARRANGE
    with patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", side_effect=["test_image.jpg", "exit"]):

        # ACT
        run_interactive_cli()

        # ASSERT
        # Check that the welcome banner was logged
        assert "AutoTomeQC Interactive Service" in caplog.text

        # Check Service Lifecycle
        service = mock_service_class.return_value
        service.start.assert_called_once()
        service.process.assert_called_once_with("test_image.jpg")
        service.stop.assert_called_once()
        assert "Bye!" in caplog.text

def test_machine_mode_happy_path(mock_service_class, caplog):
        caplog.set_level(logging.INFO)
        
        # Mock the service instance
        service = mock_service_class.return_value
        # Configure 'process' to return a Mock Future
        mock_future = MagicMock()
        # Configure the Future to return a valid dict when .result() is called
        mock_future.result.return_value = {"qc_summary": "PASS", "criteria": {}}
        # Attach this Future to the process method
        service.process.return_value = mock_future

        # ARRANGE
        inputs = ["img1.png\n", "img2.png\n", ""]

        with patch("sys.stdin.isatty", return_value=False), \
             patch("sys.stdin.readline", side_effect=inputs):

            # ACT
            run_interactive_cli()

            # ASSERT
            assert "AutoTomeQC Interactive Service" not in caplog.text
            service.start.assert_called_once()

            # Verify process was called twice with stripped strings
            assert service.process.call_count == 2
            service.process.assert_any_call("img1.png")
            service.process.assert_any_call("img2.png")

def test_exit_commands(mock_service_class):
    """Test that various exit keywords break the loop immediately."""
    exit_cmds = ["q", "QUIT", "stop", "exit"]

    for cmd in exit_cmds:
        with patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", side_effect=[cmd]):

            # Reset mock for each iteration
            mock_service_class.return_value.reset_mock()

            run_interactive_cli()

            service = mock_service_class.return_value
            # Should start and stop, but NEVER process
            service.start.assert_called()
            service.process.assert_not_called()
            service.stop.assert_called()

def test_quoted_path_handling(mock_service_class):
    """Test that quotes are stripped from file paths (common with Copy as Path)."""
    # User inputs path with double quotes
    user_input = '"C:\\Users\\Data\\image.tif"'

    with patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", side_effect=[user_input, "exit"]):
        
        run_interactive_cli()

        service = mock_service_class.return_value
        # Verify quotes were removed before passing to process()
        service.process.assert_called_once_with("C:\\Users\\Data\\image.tif")

def test_keyboard_interrupt(mock_service_class, caplog):
    """Test that Ctrl+C is handled gracefully."""
    caplog.set_level(logging.INFO)

    with patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", side_effect=KeyboardInterrupt):
    
        run_interactive_cli()

        assert "Interrupted by User" in caplog.text
        mock_service_class.return_value.stop.assert_called_once()

def test_process_exception_handling(mock_service_class, caplog):
    """Test that if processing one file fails, the app stays alive for the next one."""
    caplog.set_level(logging.INFO)

    # 1. Get the mock instance that the CLI will use
    service_instance = mock_service_class.return_value
    
    # 2. Setup a "Good" future object for the successful file
    # This mimics the concurrent.futures.Future object
    good_future = MagicMock()
    good_future.result.return_value = {"qc_summary": "PASS", "criteria": {}}

    # 3. Configure the side effects for service.process()
    # Call 1: Raises an Exception ("Corrupt File")
    # Call 2: Returns the good_future object
    service_instance.process.side_effect = [Exception("Corrupt File"), good_future]

    # 4. Simulate User Input: 
    #   1. "bad.jpg" (triggers exception)
    #   2. "good.jpg" (triggers success)
    #   3. "exit" (quits loop)
    with patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", side_effect=["bad.jpg", "good.jpg", "exit"]):

        run_interactive_cli()

    # 5. Assertions
    # Verify the first error was caught and logged gracefully
    assert "Error: Corrupt File" in caplog.text
    
    # Verify the app stayed alive and processed the second file successfully
    # (If the app crashed on the first error, this line would be missing)
    assert "PASS" in caplog.text