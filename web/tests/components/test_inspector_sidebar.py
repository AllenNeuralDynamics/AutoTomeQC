from nicegui.testing import User

async def test_initial_inspector_state(user: User):
    """Test that the inspector sidebar initially shows an info message."""
    await user.open('/')
    
    await user.should_see('Inspector')
    await user.should_see('Select an image or run batch to view informatics')