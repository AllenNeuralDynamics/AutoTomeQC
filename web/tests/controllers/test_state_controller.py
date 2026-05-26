import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from autotome_ui.controllers.state_controller import wait_backend_ready, on_fetch_config, on_toggle_masks
from autotome_ui.models.backend_schemas import AppConfig

@pytest.mark.asyncio
async def test_wait_backend_ready():
    """
    Test that wait_backend_ready loops until the backend is running.
    """
    # 1. Setup: Mock the is_running_async function to simulate backend state
    # It will return False twice, then True.
    mock_is_running = AsyncMock(side_effect=[False, False, True])

    # 2. Patch the dependencies
    with patch('autotome_ui.controllers.state_controller.is_running_async', mock_is_running), \
         patch('autotome_ui.controllers.state_controller.asyncio.sleep', new_callable=AsyncMock) as mock_sleep, \
         patch('autotome_ui.controllers.state_controller.app_state') as mock_app_state:
        
        # 3. Action: Run the function
        await wait_backend_ready()

        # 4. Assertions
        # It should have called the check 3 times
        assert mock_is_running.call_count == 3
        mock_is_running.assert_called_with(mock_app_state.is_ready_url)
        # It should have slept twice
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)

@pytest.mark.asyncio
async def test_on_fetch_config():
    """
    Test that on_fetch_config correctly fetches config and updates app_state.
    """
    # 1. Setup
    mock_config_obj = MagicMock(spec=AppConfig)

    # 2. Patch dependencies
    with patch('autotome_ui.controllers.state_controller.fetch_config_async', AsyncMock(return_value=mock_config_obj)) as mock_fetch, \
         patch('autotome_ui.controllers.state_controller.app_state') as mock_app_state:
        
        # 3. Action
        await on_fetch_config()

        # 4. Assertions
        mock_fetch.assert_called_once_with(mock_app_state.config_url)
        assert mock_app_state.config == mock_config_obj
        assert mock_app_state.is_backend_ready is True

def test_on_toggle_masks():
    """Test that on_toggle_masks correctly flips the boolean state."""
    with patch('autotome_ui.controllers.state_controller.app_state') as mock_app_state:
        mock_app_state.view.show_masks = True
        on_toggle_masks()
        assert mock_app_state.view.show_masks is False
        on_toggle_masks()
        assert mock_app_state.view.show_masks is True
