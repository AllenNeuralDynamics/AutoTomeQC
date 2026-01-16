import pytest
import logging  # <--- Added import
from unittest.mock import patch, MagicMock, call
import sys

# Import the function to test
from autotomeqc.interface.cli import run_interactive_cli

# --- FIXTURES ---

@pytest.fixture
def mock_service_class():
    """
    Patches the AutoTomeService class.
    We intercept the creation of the service so we can check if .start(), .process(), and .stop() are called.
    """
    with patch("autotomeqc.interface.cli.AutoTomeService") as MockClass:
        # Create a mock instance that the CLI will use
        mock_instance = MockClass.return_value
        # Ensure the loop condition (while service.running) defaults to True
        mock_instance.running = True
        yield MockClass

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
    """
    Scenario: Machine (Robot) pipes input.
    Checks: No banner logged, reads from sys.stdin.readline, handles process loop.
    """
    # --- FIX: Capture INFO level logs ---
    caplog.set_level(logging.INFO)

    # ARRANGE
    inputs = ["img1.png\n", "img2.png\n", ""]

    with patch("sys.stdin.isatty", return_value=False), \
         patch("sys.stdin.readline", side_effect=inputs):

        # ACT
        run_interactive_cli()

        # ASSERT
        # Ensure NO human banner was printed
        assert "AutoTomeQC Interactive Service" not in caplog.text

        service = mock_service_class.return_value
        service.start.assert_called_once()

        # Verify process was called twice with stripped strings
        expected_calls = [call("img1.png"), call("img2.png")]
        service.process.assert_has_calls(expected_calls)

        service.stop.assert_called_once()

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
    # --- FIX: Capture INFO level logs ---
    caplog.set_level(logging.INFO)

    with patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", side_effect=KeyboardInterrupt):
    
        run_interactive_cli()

        assert "Interrupted by User" in caplog.text
        mock_service_class.return_value.stop.assert_called_once()

def test_process_exception_handling(mock_service_class, caplog):
    """Test that if processing one file fails, the app stays alive for the next one."""
    # Note: ERROR logs are captured by default, but adding this doesn't hurt.
    caplog.set_level(logging.INFO)

    service = mock_service_class.return_value
    # First file crashes, second file works
    service.process.side_effect = [Exception("Corrupt File"), None]
    
    with patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", side_effect=["bad.jpg", "good.jpg", "exit"]):

        run_interactive_cli()

        # Verify error was logged
        assert "Invalid input or path error" in caplog.text

        # Verify we tried to process both
        assert service.process.call_count == 2