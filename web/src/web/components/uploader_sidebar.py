from nicegui import ui

from web.components.uploader_controller import UploaderController

def render_uploader_sidebar(BACKEND_URL, image_container, inspector_container):
    """Renders the left sidebar and contains the upload logic."""
    with ui.left_drawer(fixed=True).classes('sidebar w-80 p-0') as left_drawer:
        
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC v2.4').classes('sidebar-title-text text-white')
            
            # Small icon button replacing the large upload dropzone
            upload_btn = ui.button(icon='upload', color=None) \
                .props('flat dense') \
                .classes('btn-icon border border-[var(--border-light)] w-8 h-8')
        
        queue_container = ui.column().classes('sidebar-content custom-scrollbar gap-1 flex-nowrap overflow-y-auto')
        with queue_container:
            empty_state = ui.column().classes('h-full w-full flex flex-col items-center justify-center p-8 text-center gap-4 opacity-40')
            with empty_state:
                with ui.element('div').classes('w-12 h-12 rounded-full border border-dashed border-[#555555] flex items-center justify-center'):
                    ui.icon('image', size='sm').classes('text-white')
                ui.label('NO DATA LOADED').classes('text-xs font-mono text-white')
                
        # Initialize the state and logic controller
        controller = UploaderController(BACKEND_URL, image_container, inspector_container, queue_container, empty_state)
        
        # Hidden native uploader triggered by the header button
        uploader = ui.upload(on_upload=controller.handle_upload, multiple=True, auto_upload=True).props('accept="image/*"').classes('hidden')
        upload_btn.on('click', lambda: uploader.run_method('pickFiles'))
        
        with ui.row().classes('sidebar-footer'):
            ui.button('PROCESS BATCH', icon='play_arrow', color=None, on_click=controller.process_batch).classes('btn-process flex items-center justify-center gap-2')
            
    return left_drawer