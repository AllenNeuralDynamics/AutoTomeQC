import asyncio
import base64
import httpx
import tempfile
import uuid
from pathlib import Path
from nicegui import ui

from web.services.api import analyze_image
from web.components.results_card import display_qc_result

def render_uploader_sidebar(BACKEND_URL, image_container, inspector_container):
    """Renders the left sidebar and contains the upload logic."""
    
    # Create a temporary directory to store files on disk instead of in RAM
    temp_dir = Path(tempfile.mkdtemp(prefix="autotome_"))
    
    uploaded_files = []

    async def handle_upload(e):
        file_name = getattr(e, 'name', 'uploaded_image.jpg')
        
        # 1. Enforce image-only files at the logic level
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            ui.notify(f"Skipped {file_name}: Only image files are allowed.", type='warning')
            return
            
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

        # 2. Save the file to the temporary directory on disk
        unique_filename = f"{uuid.uuid4().hex}_{file_name}"
        file_path = temp_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
        # 3. Store both the original name and the unique file path in our queue
        uploaded_files.append((file_name, file_path))

    async def process_batch(e):
        if not uploaded_files:
            ui.notify("Please upload images first.", type='warning')
            return
            
        e.sender.disable()
        ui.notify(f"Processing {len(uploaded_files)} images...")
        
        for file_name, file_path in uploaded_files:
            image_container.clear()
            inspector_container.clear()
            with image_container:
                ui.spinner('dots', size='lg')
            
            # Load only ONE image into RAM at a time
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            # Generate display source on the fly
            ext = file_name.lower().split('.')[-1]
            mime_type = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"
            base64_img = base64.b64encode(file_bytes).decode('utf-8')
            img_src = f"data:{mime_type};base64,{base64_img}"
            
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
                    
            # Brief pause to allow the UI to render the result before the next iteration
            await asyncio.sleep(1.0)
            
            # Clean up the temp file to save disk space
            file_path.unlink(missing_ok=True)
            
        # Empty the queue
        uploaded_files.clear()
            
        e.sender.enable()
        ui.notify("Batch processing complete!", type='positive')

    with ui.left_drawer(fixed=True).classes('sidebar') as left_drawer:
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC v2.4').classes('sidebar-title-text')
        
        with ui.column().classes('sidebar-content'):
            # 3. Enforce image-only files at the browser picker level
            ui.upload(on_upload=handle_upload, multiple=True, label="UPLOAD DATA", auto_upload=True).props('accept="image/*" flat bordered').classes('w-full')
        
        with ui.row().classes('sidebar-footer'):
            ui.button('PROCESS BATCH', icon='play_arrow', color=None, on_click=process_batch).classes('btn-process')
            
    return left_drawer