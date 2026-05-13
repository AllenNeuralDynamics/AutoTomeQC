#he "glue" that connects the Services to the Components when a user visits a URL or clicks a button.
import os
import argparse
import tempfile
from pathlib import Path
from nicegui import ui, app

from web.models.status import app_state
from web.components.app_header import render_header
from web.components.main_workspace import render_main_workspace, update_main_workspace, set_workspace_idle, set_workspace_pending, set_workspace_error
from web.components.inspector_sidebar import render_inspector_sidebar, update_inspector_sidebar, set_inspector_idle, set_inspector_pending, set_inspector_error
from web.components.uploader_sidebar import render_uploader_sidebar
from web.components.loading_overlay import render_loading_overlay
import web.controllers.state_controller
from web.controllers.uploader_controller import UploaderController
from web.protocol.events import image_selected, image_pending, image_error, clear_views

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
    ui.colors(primary='#F27D26', secondary='#151515', accent='#F27D26')
    ui.add_head_html('<link href="/static/theme.css" rel="stylesheet">')  # connect to theme

    # --- LOADING OVERLAY ---
    render_loading_overlay(app_state.is_ready_url, app_state.config_url)

    # --- 1. RIGHT SIDEBAR (Inspector) ---
    _, inspector_container = render_inspector_sidebar()

    # --- 2. MAIN WORKSPACE (Image Viewer) ---
    image_container = render_main_workspace()

    # --- 3. CONTROLLERS ---
    uploader_controller = UploaderController(app_state.process_url, temp_upload_dir, temp_upload_url_prefix)

    # --- 4. LEFT SIDEBAR (Uploader) ---
    left_drawer, q_container, e_state = render_uploader_sidebar(
        on_upload=uploader_controller.handle_upload,
        on_process=uploader_controller.process_batch
    )
    uploader_controller.queue_container = q_container
    uploader_controller.empty_state = e_state

    # --- 5. WIRE UP EVENT SUBSCRIBERS ---
    @image_selected.subscribe
    def handle_image_selected(data):
        path, result, raw_json = data
        update_main_workspace(image_container, path, result)
        update_inspector_sidebar(inspector_container, result, raw_json)
        
    @image_pending.subscribe
    def handle_image_pending(img_src):
        set_workspace_pending(image_container, img_src)
        set_inspector_pending(inspector_container)
        
    @image_error.subscribe
    def handle_image_error(msg):
        set_workspace_error(image_container, msg)
        set_inspector_error(inspector_container, msg)
        
    @clear_views.subscribe
    def handle_clear_views(_=None):
        set_workspace_idle(image_container)
        set_inspector_idle(inspector_container)

    # --- 6. TOOLBAR (Header) ---
    render_header(left_drawer)
    
if __name__ in {"__main__", "__mp_main__"}:
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to run the NiceGUI server on")
    args = parser.parse_args()

    # Start the NiceGUI engine outside the page route
    # WEB UI
    ui.run(port=args.port, title="AutoTomeQC", favicon="🔬", show=False, reload=False)
    
    # Desktop option
    #ui.run(native=True, reload=False, title="AutoTomeQC", port=args.port)
