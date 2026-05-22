import argparse
from pathlib import Path
import shutil
from nicegui import ui, app
import logging

# Controller/Component Imports...
from web.models.status import app_state
from web.controllers.state_controller import wait_backend_ready, on_fetch_config, on_toggle_masks
from web.controllers.uploader_controller import UploaderController
from web.components.app_header import render_header
from web.components.main_workspace import MainWorkspace
from web.components.inspector_sidebar import render_inspector_sidebar, inspector_content
from web.components.loading_overlay import render_loading_overlay
from web.components.uploader_sidebar import render_uploader_sidebar
from web.components.queue_renderer import QueueRenderer
from web.utils.launcher_utils import configure_logging


logger = logging.getLogger(__name__)

# --- Configuration & Paths ---
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = Path(app_state.temp_upload_dir)

def setup_app_resources():
    """Register static paths and prepare workspace."""
    app.add_static_files("/static", str(STATIC_DIR))
    
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
    app.add_static_files('/temp_uploads', str(TEMP_DIR))

# Initialize Resources
setup_app_resources()

@ui.page('/')
def index():
    logger.debug(f"Starting AutoTomeQC in {'Web' if args.web else 'Native'} mode.")
    ui.add_css((STATIC_DIR / 'theme.css').read_text())
    ui.dark_mode().enable()
    ui.colors(primary='#F27D26', secondary='#151515', accent='#F27D26')

    # --- UI Layout ---
    render_loading_overlay(wait_backend_ready, on_fetch_config)

    workspace = MainWorkspace(
        on_prev_callback=lambda: uploader_controller.load_prev(),
        on_next_callback=lambda: uploader_controller.load_next()
    )

    queue_renderer = QueueRenderer()
    uploader_controller = UploaderController(
        add_ui_callback=queue_renderer.add_item,
        remove_ui_callback=queue_renderer.remove_items,
        set_active_ui_callback=queue_renderer.set_active,
        refresh_workspace=workspace.render.refresh,
        refresh_inspector=inspector_content.refresh
    )

    render_inspector_sidebar(toggle_masks_callback=on_toggle_masks)
    workspace.render()

    left_drawer = render_uploader_sidebar(
        renderer=queue_renderer,
        on_upload_callback=uploader_controller.handle_upload,
        on_process_callback=uploader_controller.process_batch,
        on_item_click=uploader_controller.load_result,
        on_item_delete=uploader_controller.remove_file
    )

    render_header(left_drawer=left_drawer)

@app.on_shutdown
def cleanup_temp_directory():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_known_args()[0]

    configure_logging(level=args.log_level)
    app_state.is_native = not args.web

    ui.run(
        port=args.port, 
        title="AutoTomeQC", 
        favicon=str(STATIC_DIR / "favicon.ico"), 
        show=False, 
        reload=False, 
        reconnect_timeout=60.0,
        native=app_state.is_native,
        window_size=(1600, 1200) if app_state.is_native else None
    )