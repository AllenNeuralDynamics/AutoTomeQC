# web/components/loading_overlay.py
from nicegui import ui
from web.events import is_running, fetch_config

def render_loading_overlay():
    """Renders a full-screen loading dialog that waits for the backend to become ready."""
    with ui.dialog().props('persistent maximized') as loading_dialog:
        with ui.column().classes('w-full h-full items-center justify-center bg-[#151515]'):
            ui.spinner('dots', size='5em', color='primary')
            ui.label('Initializing Models...').classes('text-2xl text-white mt-4 font-bold')
            ui.label('This may take up to 30 seconds').classes('text-lg text-gray-400 mt-2')
    
    loading_dialog.open()

    async def _check_backend_ready():
        try:
            await is_running.call()
            # Backend is ready! Fetch the configuration before closing the overlay
            try:
                await fetch_config.call()
                loading_dialog.close()
                health_timer.deactivate()
            except Exception as e:
                ui.notify(f"Failed to fetch config: {str(e)}", type='negative')
        except Exception as e:
            ui.notify(f"Health check failed: {str(e)}", type='negative')

    health_timer = ui.timer(1.0, _check_backend_ready)