from nicegui import ui
from pathlib import Path

def render_uploader_sidebar(on_upload, on_process):
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
                
        # Hidden native uploader triggered by the header button
        uploader = ui.upload(on_upload=on_upload, multiple=True, auto_upload=True).props('accept="image/*"').classes('hidden-uploader')
        upload_btn.on('click', lambda: uploader.run_method('pickFiles'))
        
        with ui.row().classes('sidebar-footer'):
            ui.button('PROCESS BATCH', icon='play_arrow', color=None, on_click=on_process).classes('btn-process')
            
    return left_drawer, queue_container, empty_state