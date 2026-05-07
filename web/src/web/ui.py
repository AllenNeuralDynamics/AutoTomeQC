#he "glue" that connects the Services to the Components when a user visits a URL or clicks a button.
import os
import argparse
import tempfile
from pathlib import Path
from nicegui import ui, app

from web.components.app_header import render_header
from web.components.main_workspace import render_main_workspace
from web.components.inspector_sidebar import render_inspector_sidebar
from web.components.uploader_sidebar import render_uploader_sidebar

# Parse arguments for port mapping
parser = argparse.ArgumentParser()

# Mount the static directory so the browser can access files inside it
static_dir = Path(__file__).resolve().parent / "static"
app.add_static_files("/static", str(static_dir))

# Create a persistent temporary directory for uploaded files for the session
# This ensures the static route is consistent and files persist across interactions
temp_upload_dir = Path(tempfile.mkdtemp(prefix="autotome_"))
temp_upload_url_prefix = f"/temp_files/{temp_upload_dir.name}" # Use 'temp_files' to match user's example
app.add_static_files(temp_upload_url_prefix, str(temp_upload_dir))

@ui.page('/')
def index():
    ui.dark_mode().enable()
    
    # Globally replace Quasar's default "Sky Blue" with our theme's Orange
    ui.colors(primary='#F27D26', secondary='#151515', accent='#F27D26')
    
    # Connect to the custom application stylesheet
    ui.add_head_html('<link href="/static/theme.css" rel="stylesheet">')

    # Read from environment variable, fallback to localhost for local development
    BACKEND_URL = os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000/api/v1/process")

    # --- 1. RIGHT SIDEBAR (Inspector) ---
    # We initialize it first so we can pass its container to the uploader logic
    right_drawer, inspector_container = render_inspector_sidebar()

    # --- 2. MAIN WORKSPACE (Image Viewer) ---
    image_container = render_main_workspace()

    # --- 3. LEFT SIDEBAR (Uploader) ---
    # Pass the persistent temporary upload directory and its URL prefix
    left_drawer = render_uploader_sidebar(BACKEND_URL, temp_upload_dir, temp_upload_url_prefix, image_container, inspector_container)

    # --- 4. TOOLBAR (Header) ---
    render_header(left_drawer)
    
if __name__ in {"__main__", "__mp_main__"}:
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to run the NiceGUI server on")
    args = parser.parse_args()

    # Start the NiceGUI engine outside the page route
    # WEB UI
    ui.run(port=args.port, title="AutoTomeQC", favicon="🔬", show=True, reload=False)
    
    # Desktop option
    #ui.run(native=True, reload=False, title="AutoTomeQC", port=args.port)
