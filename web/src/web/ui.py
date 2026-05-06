#he "glue" that connects the Services to the Components when a user visits a URL or clicks a button.
import asyncio
from nicegui import ui, app
import httpx
import os
import argparse
import base64
from pathlib import Path

from web.services.api import analyze_image
from web.components.results_card import display_qc_result

# Parse arguments for port mapping
parser = argparse.ArgumentParser()

# Mount the static directory so the browser can access files inside it
static_dir = Path(__file__).resolve().parent / "static"
app.add_static_files("/static", str(static_dir))

@ui.page('/')
def index():
    ui.dark_mode().enable()
    
    # Globally replace Quasar's default "Sky Blue" with our theme's Orange
    ui.colors(primary='#F27D26', secondary='#151515', accent='#F27D26')
    
    # Connect to the external stylesheet
    ui.add_head_html('<link href="/static/tailwind.css" rel="stylesheet">')

    # Read from environment variable, fallback to localhost for local development
    BACKEND_URL = os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000/api/v1/process")

    # --- 1. TOOLBAR (Header) ---
    with ui.header().classes('app-header').classes(remove='bg-primary'):
        with ui.row().classes('header-left'):
            ui.button(icon='chevron_left', color=None, on_click=lambda: left_drawer.toggle()).props('flat dense').classes('btn-icon')
            ui.element('div').classes('header-divider')
            with ui.row().classes('project-title-container'):
                ui.label('PROJECT').classes('text-accent')
                ui.label('/')
                ui.label('UNTITLED_PROJECT').classes('text-title')
        
        with ui.row().classes('header-right'):
            ui.button('CONFIG', icon='settings', color=None).classes('btn-config')
            ui.button('EXPORT', icon='download', color=None).classes('btn-export')

    # --- 2. MAIN WORKSPACE (Image Viewer) ---
    ui.query('.q-page').classes('main-workspace bg-grid')
    image_container = ui.column().classes('image-container')
    with image_container:
        with ui.column().classes('viewport-idle'):
            ui.icon('aspect_ratio', size='6rem')
            ui.label('VIEWPORT_IDLE')

    # --- 3. UPLOAD CALLBACK (Logic) ---
    async def handle_upload(e):
        image_container.clear()
        inspector_container.clear()
        with image_container:
            ui.spinner('dots', size='lg')
        
        # --- NiceGUI Version Compatibility ---
        # NiceGUI recently changed their upload API and renamed the file object.
        # This checks for the correct attribute dynamically.
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
            # If the read method is async (returns a coroutine), we must await it!
            if asyncio.iscoroutine(read_result):
                file_bytes = await read_result
            else:
                file_bytes = read_result
        else:
            file_bytes = file_obj

        file_name = getattr(e, 'name', 'uploaded_image.jpg')
        
        # Convert bytes to base64 so NiceGUI can render it natively
        base64_img = base64.b64encode(file_bytes).decode('utf-8')
        img_src = f"data:image/jpeg;base64,{base64_img}"
        
        try:
            # 1. Fetch Data (Service)
            result, raw_json = await analyze_image(BACKEND_URL, file_name, file_bytes)
            
                # 2. Draw Data (Component)
            display_qc_result(result, raw_json, img_src, image_container, inspector_container)
                    
        except httpx.HTTPStatusError as exc:
            inspector_container.clear()
            with inspector_container:
                ui.label(f"Backend Error: {exc.response.text}").classes('text-red-600 font-bold')
        except httpx.RequestError:
            inspector_container.clear()
            with inspector_container:
                ui.label(f"Failed to connect to the backend at {BACKEND_URL}").classes('text-red-600 font-bold')

    # --- 4. LEFT SIDEBAR (Uploader) ---
    with ui.left_drawer(fixed=True).classes('sidebar') as left_drawer:
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC v2.4').classes('sidebar-title-text')
        
        # Inject the uploader here so it can use the callback
        with ui.column().classes('sidebar-content'):
            ui.upload(on_upload=handle_upload, label="UPLOAD DATA", auto_upload=True).props('accept=".jpg,.jpeg,.png,.tif,.tiff" flat bordered').classes('w-full')
        
        with ui.row().classes('sidebar-footer'):
            ui.button('PROCESS BATCH', icon='play_arrow', color=None).classes('btn-process')

    # --- 5. RIGHT SIDEBAR (Inspector) ---
    with ui.right_drawer(fixed=True).classes('sidebar') as right_drawer:
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('terminal').classes('text-accent text-lg')
                ui.label('Inspector').classes('sidebar-title-text')
        
        # Store reference so the callback can push data here
        inspector_container = ui.column().classes('inspector-content')
        with inspector_container:
            with ui.column().classes('viewport-idle'):
                ui.icon('info', size='2rem')
                ui.label('Select an image or run batch to view informatics')
    
if __name__ in {"__main__", "__mp_main__"}:
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to run the NiceGUI server on")
    args = parser.parse_args()

    # Start the NiceGUI engine outside the page route
    ui.run(port=args.port, title="AutoTomeQC", show=False, reload=False)
