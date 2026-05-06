from nicegui import ui

from web.components.uploader_controller import UploaderController

def render_uploader_sidebar(BACKEND_URL, image_container, inspector_container):
    """Renders the left sidebar and contains the upload logic."""
    with ui.left_drawer(fixed=True).classes('sidebar bg-[#0A0A0A] flex flex-col border-r border-[#222222] p-0 w-80') as left_drawer:
        
        with ui.row().classes('p-4 border-b border-[#222222] flex items-center justify-between w-full m-0'):
            with ui.row().classes('items-center gap-2 m-0'):
                ui.icon('monitor_heart').classes('text-[#F27D26] text-xl')
                ui.label('AutoTome-QC v2.4').classes('font-mono font-bold tracking-tight text-sm uppercase text-white')
            
            # Small icon button replacing the large upload dropzone
            upload_btn = ui.button(icon='upload', color=None) \
                .props('flat dense') \
                .classes('hover:bg-[#222222] rounded border border-[#333333] transition-colors text-white w-8 h-8')
        
        queue_container = ui.column().classes('flex-1 overflow-y-auto w-full p-2 gap-1 custom-scrollbar flex-nowrap')
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
        
        with ui.row().classes('p-4 border-t border-[#222222] w-full m-0'):
            ui.button('PROCESS BATCH', icon='play_arrow', color=None, on_click=controller.process_batch).classes('w-full py-2 bg-[#F27D26] hover:bg-[#ff8c3a] text-white rounded font-mono text-xs font-bold transition-all flex items-center justify-center gap-2')
            
    return left_drawer