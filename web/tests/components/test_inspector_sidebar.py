import pytest
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

@pytest.mark.asyncio
async def test_inspector_shows_results_after_processing(user: User, tmp_path: Path):
    """
    Test that the inspector updates with results after a successful analysis.
    """
    # 1. Create a dummy file to upload
    fake_image_path = tmp_path / "test.jpg"
    fake_image_path.write_bytes(b"fake image data")

    # 2. Create a mock result to be returned by the mocked API
    mock_criteria = {"coverage": QCCriteria(pass_status=True, label="full_section")}
    mock_section = SectionResult(qc_result="PASS", segmentation_conf=0.9, area_in_pixels=100, overlap_ratio=1.0, criteria=mock_criteria)
    mock_result = PipelineResult(filename='test.jpg', timestamp='now', qc_summary='PASS', fail_reason='n/a', processing_time_sec=0.5, sections=[mock_section])
    mock_raw_json = mock_result.model_dump()

    # 3. Mock the backend analysis to return our fake result
    with patch('web.controllers.uploader_controller.analyze_image', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = (mock_result, mock_raw_json)

        # 4. Open page, upload the file, and click the process button
        await user.open('/')
        await user.upload(fake_image_path)
        await user.should_see("test.jpg")
        await user.click("PROCESS BATCH")

        # 5. Verify the inspector UI updates with the mocked results
        await user.should_see('Status')
        await user.should_see('PASS')
        await user.should_see('0.5s')
        await user.should_see('SECTION 1')
        await user.should_see('full_section')
        
        # 6. Test that the mask toggle button is now visible
        user.find_by_tag('button', icon='visibility').should_be_visible()