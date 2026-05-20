import pytest
from nicegui.testing import User
from pathlib import Path

@pytest.mark.asyncio
async def test_initial_workspace_state(user: User):
    """Test that the main workspace initially shows the idle state."""
    await user.open('/')
    
    # The main workspace should show an idle message
    await user.should_see('VIEWPORT IDLE')
