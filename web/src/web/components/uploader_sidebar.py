import asyncio
import base64
import httpx
from nicegui import ui

from web.services.api import analyze_image
from web.components.results_card import display_qc_result

def render_uploader_sidebar(BACKEND_URL, image_container, inspector_container):
    """Renders the left sidebar and contains the upload logic."""
    
    async def handle_upload(e):
        image_container.clear()
        inspector_container.clear()
        with image_container:
            ui.spinner('dots', size='lg')
        
        # --- NiceGUI Version Compatibility ---
        if hasattr(e, 'content'):
            file_obj = e.content
        elif hasattr(e, 'file'):
            file_obj = e.file
        elif hasattr(e, 'stream'):
            file_obj = e.stream
        else:
            ui.notify(f"Unknown upload format. Attributes available: {dir(e)}", type='negative')
            return
            
        if hasattr(file_obj, 'read'):
            read_result = file_obj.read()
            if asyncio.iscoroutine(read_result):
                file_bytes = await read_result
            else:
                file_bytes = read_result
        else:
            file_bytes = file_obj

        file_name = getattr(e, 'name', 'uploaded_image.jpg')
        
        base64_img = base64.b64encode(file_bytes).decode('utf-8')
        img_src = f"data:image/jpeg;base64,{base64_img}"
        
        try:
            result, raw_json = await analyze_image(BACKEND_URL, file_name, file_bytes)
            display_qc_result(result, raw_json, img_src, image_container, inspector_container)
                    
        except httpx.HTTPStatusError as exc:
            inspector_container.clear()
            with inspector_container:
                ui.label(f"Backend Error: {exc.response.text}").classes('text-red-600 font-bold')
        except httpx.RequestError:
            inspector_container.clear()
            with inspector_container:
                ui.label(f"Failed to connect to the backend at {BACKEND_URL}").classes('text-red-600 font-bold')

    with ui.left_drawer(fixed=True).classes('sidebar') as left_drawer:
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC v2.4').classes('sidebar-title-text')
        
        with ui.column().classes('sidebar-content'):
            ui.upload(on_upload=handle_upload, multiple=True, label="UPLOAD DATA", auto_upload=True).props('accept=".jpg,.jpeg,.png,.tif,.tiff" flat bordered').classes('w-full')
        
        with ui.row().classes('sidebar-footer'):
            ui.button('PROCESS BATCH', icon='play_arrow', color=None).classes('btn-process')
            
    return left_drawer