import pytest
from nicegui import ui
from unittest.mock import patch, AsyncMock
from nicegui.testing import User
from pathlib import Path
from web.models.backend_schemas import PipelineResult, SectionResult, QCCriteria

@pytest.mark.asyncio
async def test_initial_inspector_state(user: User):
    """Test that the inspector sidebar initially shows an info message."""
    await user.open('/')
    
    await user.should_see('Inspector')
    await user.should_see('Select an image or run batch to view informatics')
