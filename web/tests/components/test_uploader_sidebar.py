import pytest
from nicegui.testing import User
from pathlib import Path

# The pytest-nicegui plugin and the user fixture are enabled by the pytest.ini file.

@pytest.mark.asyncio
async def test_initial_sidebar_state(user: User):
    """Test that the uploader sidebar initially shows the empty state."""
    await user.open('/')
    
    # Check for the title and the empty state message
    await user.should_see('AutoTome-QC')
    await user.should_see('NO DATA LOADED')

@pytest.mark.asyncio
async def test_upload_and_item_appears(user: User, tmp_path: Path):
    """
    Test that uploading a file adds it to the queue in the sidebar.
    """
    # 1. Create a dummy file to upload
    fake_image_path = tmp_path / "test.jpg"
    fake_image_path.write_bytes(b"fake image data")

    # 2. Open the page
    await user.open('/')
    await user.should_see('NO DATA LOADED')

    # 3. Trigger the upload
    await user.upload(fake_image_path)

    # 4. Verify the UI updates: the empty message is gone, and the new file appears.
    await user.should_not_see('NO DATA LOADED')
    await user.should_see('test.jpg')
    await user.should_see('PENDING')

@pytest.mark.asyncio
async def test_delete_all_dialog(user: User):
    """Test that the 'delete all' button shows a confirmation dialog."""
    await user.open('/')
    await user.click(user.find_by_tag('button', icon='delete_sweep'))
    await user.should_see('Are you sure you want to delete all items in the queue?')
    await user.click('Cancel')
    await user.should_not_see('Are you sure you want to delete all items in the queue?')