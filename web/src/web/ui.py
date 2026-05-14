# main.py
import argparse
from pathlib import Path
from nicegui import ui, app

from web.models.status import app_state
from web.controllers.state_controller import wait_backend_ready, on_fetch_config
from web.components.app_header import render_header
from web.components.main_workspace import render_main_workspace
from web.components.inspector_sidebar import render_inspector_sidebar, inspector_content
from web.components.loading_overlay import render_loading_overlay
from web.components.uploader_sidebar import render_uploader_sidebar, render_queue_list
from web.controllers.uploader_controller import UploaderController
# from web.protocol.events import ... (Deleted!)

parser = argparse.ArgumentParser()
static_dir = Path(__file__).resolve().parent / "static"
app.add_static_files("/static", str(static_dir))
app.add_static_files(app_state.temp_upload_url_prefix, str(app_state.temp_upload_dir))

@ui.page('/')
def index():
    ui.dark_mode().enable()
    ui.colors(primary='#F27D26', secondary='#151515', accent='#F27D26')
    ui.add_head_html('<link href="/static/theme.css" rel="stylesheet">')

    # --- LOADING OVERLAY ---
    render_loading_overlay(
        wait_backend_ready_callback=wait_backend_ready,
        fetch_config_callback=on_fetch_config
    )

    # --- 1. RIGHT SIDEBAR (Inspector) ---
    render_inspector_sidebar()

    # --- 2. MAIN WORKSPACE (Image Viewer) ---
    render_main_workspace()

    # --- 3. CONTROLLERS ---
    # Pass all component .refresh methods into the controller
    uploader_controller = UploaderController(
        refresh_ui_callback=render_queue_list.refresh,
        refresh_workspace=render_main_workspace.refresh,
        refresh_inspector=inspector_content.refresh
    )

    # --- 4. LEFT SIDEBAR (Uploader) ---
    left_drawer = render_uploader_sidebar(
        on_upload_callback=uploader_controller.handle_upload,
        on_process_callback=uploader_controller.process_batch,
        on_item_click=uploader_controller.load_result,
        on_item_delete=uploader_controller.remove_file
    )

    # --- 5. TOOLBAR (Header) ---
    render_header(left_drawer)

if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    ui.run(port=args.port, title="AutoTomeQC", favicon="🔬", show=False, reload=False)