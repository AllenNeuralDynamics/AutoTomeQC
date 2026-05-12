from nicegui import ui
from pathlib import Path
from web.controllers.uploader_controller import UploaderController

def render_uploader_sidebar(BACKEND_URL: str, temp_upload_dir: Path, temp_upload_url_prefix: str, image_container, inspector_container):
    """Renders the left sidebar and contains the upload logic."""
    with ui.left_drawer(fixed=True).classes('sidebar') as left_drawer:
        
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')
            
            # Small icon button replacing the large upload dropzone
            upload_btn = ui.button(icon='upload', color=None).classes('btn-upload')
        
        queue_container = ui.column().classes('queue-container custom-scrollbar gap-0')
        with queue_container:
            empty_state = ui.column().classes('queue-empty-state')
            with empty_state:
                with ui.element('div').classes('queue-empty-icon-wrapper'):
                    ui.icon('image', size='sm').classes('text-white')
                ui.label('NO DATA LOADED').classes('queue-empty-text')
                
        # Initialize the state and logic controller
        controller = UploaderController(BACKEND_URL, temp_upload_dir, temp_upload_url_prefix, image_container, inspector_container, queue_container, empty_state)
        
        # Hidden native uploader triggered by the header button
        uploader = ui.upload(on_upload=controller.handle_upload, multiple=True, auto_upload=True).props('accept="image/*"').classes('hidden-uploader')
        upload_btn.on('click', lambda: uploader.run_method('pickFiles'))
        
        with ui.row().classes('sidebar-footer'):
            ui.button('PROCESS BATCH', icon='play_arrow', color=None, on_click=controller.process_batch).classes('btn-process')
            
    return left_drawer