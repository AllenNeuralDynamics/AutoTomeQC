# main.py
import argparse
from pathlib import Path
from nicegui import ui, app
import shutil

from web.models.status import app_state
from web.controllers.state_controller import wait_backend_ready, on_fetch_config, on_toggle_masks
from web.controllers.uploader_controller import UploaderController
from web.components.app_header import render_header
from web.components.main_workspace import MainWorkspace
from web.components.inspector_sidebar import render_inspector_sidebar, inspector_content
from web.components.loading_overlay import render_loading_overlay
from web.components.uploader_sidebar import QueueRenderer, render_uploader_sidebar

static_dir = Path(__file__).resolve().parent / "static"
app.add_static_files("/static", str(static_dir))
temp_dir = Path(app_state.temp_upload_dir)
if not temp_dir.exists():
    temp_dir.mkdir(parents=True, exist_ok=True)
app.add_static_files('/temp_uploads', str(temp_dir))

@ui.page('/')
def index():
    ui.add_css((static_dir / 'theme.css').read_text())
    ui.dark_mode().enable()
    ui.colors(primary='#F27D26', secondary='#151515', accent='#F27D26')

    # --- LOADING OVERLAY ---
    render_loading_overlay(
        wait_backend_ready_callback=wait_backend_ready,
        fetch_config_callback=on_fetch_config
    )

    # --- INSTANTIATE WORKSPACE ---
    # Use lambdas here so it can reference uploader_controller before it's created
    workspace = MainWorkspace(
        on_prev_callback=lambda: uploader_controller.load_prev(),
        on_next_callback=lambda: uploader_controller.load_next()
    )

    # --- CONTROLLERS ---
    queue_renderer = QueueRenderer() # instantiage queue renderer
    uploader_controller = UploaderController(
        add_ui_callback=queue_renderer.add_item,
        remove_ui_callback=queue_renderer.remove_items,
        set_active_ui_callback=queue_renderer.set_active,
        refresh_workspace=workspace.render.refresh,
        refresh_inspector=inspector_content.refresh
    )

    # --- RIGHT SIDEBAR (Inspector) ---
    render_inspector_sidebar(toggle_masks_callback=on_toggle_masks)

    # --- MAIN WORKSPACE (Image Viewer) ---
    # Render the class component into the UI layout
    workspace.render()

    # --- LEFT SIDEBAR (Uploader) ---
    left_drawer = render_uploader_sidebar(
        renderer = queue_renderer,
        on_upload_callback=uploader_controller.handle_upload,
        on_process_callback=uploader_controller.process_batch,
        on_item_click=uploader_controller.load_result,
        on_item_delete=uploader_controller.remove_file
    )

    # --- TOOLBAR (Header) ---
    render_header(left_drawer=left_drawer)

# --- CLEANUP ON SHUTDOWN ---
@app.on_shutdown
def cleanup_temp_directory():
    """Removes the temporary directory and all its contents when the app closes."""
    temp_dir = app_state.temp_upload_dir
    if temp_dir and temp_dir.exists():
        # shutil.rmtree deletes the folder and all files inside it
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args, unknown = parser.parse_known_args()
    # uvicorn_reload_includes='*.py,*.css'
    ui.run(
        port=args.port, 
        title="AutoTomeQC", 
        favicon=str(static_dir / "favicon.ico"), 
        show=False, 
        reload=False, 
        native=False,
        reconnect_timeout=10.0,
        uvicorn_reload_includes='*.css'
    )