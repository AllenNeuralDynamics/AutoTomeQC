from nicegui import ui
from web.services.api import check_health_async, fetch_config_async
from web.models.status import app_state

def render_loading_overlay(health_url: str, config_url: str):
    """Renders a full-screen loading dialog that waits for the backend to become ready."""
    with ui.dialog().props('persistent maximized') as loading_dialog:
        with ui.column().classes('w-full h-full items-center justify-center bg-[#151515]'):
            ui.spinner('dots', size='5em', color='primary')
            ui.label('Initializing Models...').classes('text-2xl text-white mt-4 font-bold')
            ui.label('This may take up to 30 seconds').classes('text-lg text-gray-400 mt-2')
    
    loading_dialog.open()

    async def check_backend_ready():
        if await check_health_async(health_url):
            # Backend is ready! Fetch the configuration before closing the overlay
            try:
                app_state.config = await fetch_config_async(config_url)
                app_state.is_backend_ready = True
                loading_dialog.close()
                health_timer.deactivate()
            except Exception as e:
                ui.notify(f"Failed to fetch config: {str(e)}", type='negative')

    health_timer = ui.timer(2.0, check_backend_ready)