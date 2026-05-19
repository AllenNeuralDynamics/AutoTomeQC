import pytest
from nicegui.testing import User
from pathlib import Path

@pytest.mark.asyncio
async def test_initial_workspace_state(user: User):
    """Test that the main workspace initially shows the idle state."""
    await user.open('/')
    
    # The main workspace should show an idle message
    await user.should_see('VIEWPORT IDLE')
    
    # Navigation should not be visible as there are no items
    user.find('chevron_left').should_not_be_visible()
    user.find('chevron_right').should_not_be_visible()

@pytest.mark.asyncio
async def test_navigation_appears_with_multiple_files(user: User, tmp_path: Path):
    """
    Test that navigation controls appear when more than one file is in the queue
    and that they function correctly.
    """
    # 1. Create dummy files for upload
    (tmp_path / "test1.jpg").write_bytes(b"fake1")
    (tmp_path / "test2.jpg").write_bytes(b"fake2")

    # 2. Open page and upload the files
    await user.open('/')
    await user.upload(tmp_path / "test1.jpg")
    await user.upload(tmp_path / "test2.jpg")

    # 3. Wait for uploads to register in the UI
    await user.should_see("test1.jpg")
    await user.should_see("test2.jpg")

    # 4. Click the first item to make it active and check navigation
    await user.click("test1.jpg")
    await user.should_see('1 / 2')
    user.find('chevron_left').should_be_visible()
    user.find('chevron_right').should_be_visible()

    # 5. Test navigation by clicking the 'next' button
    await user.click(user.find('chevron_right'))
    await user.should_see('2 / 2')

    # 6. Verify the active item has updated in the sidebar
    active_item = user.find(text='test2.jpg').parent_by_tag('div', 'queue-item')
    active_item.should_have_class('active')