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
    # Inject the custom Tailwind CSS from the static folder
    ui.add_head_html('<link href="/static/tailwind.css" rel="stylesheet">')

    # Read from environment variable, fallback to localhost for local development
    BACKEND_URL = os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000/api/v1/process")

    ui.label("AutoTomeQC Dashboard").classes("text-3xl font-bold mb-6")

    results_container = ui.column()

    async def handle_upload(e):
        results_container.clear()
        with results_container:
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
            
            results_container.clear()
            with results_container:
                # 2. Draw Data (Component)
                display_qc_result(result, raw_json, img_src)
                    
        except httpx.HTTPStatusError as exc:
            results_container.clear()
            with results_container:
                ui.label(f"Backend Error: {exc.response.text}").classes('text-red-600 font-bold')
        except httpx.RequestError:
            results_container.clear()
            with results_container:
                ui.label(f"Failed to connect to the backend at {BACKEND_URL}").classes('text-red-600 font-bold')

    ui.upload(on_upload=handle_upload, label="Upload a section image", auto_upload=True).props('accept=".jpg,.jpeg,.png,.tif,.tiff"')
    
if __name__ in {"__main__", "__mp_main__"}:
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to run the NiceGUI server on")
    args = parser.parse_args()

    # Start the NiceGUI engine outside the page route
    ui.run(port=args.port, title="AutoTomeQC", show=False, reload=False)
