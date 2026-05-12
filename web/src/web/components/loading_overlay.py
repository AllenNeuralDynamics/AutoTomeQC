from nicegui import ui
from web.services.api import check_health_async

def render_loading_overlay(health_url: str):
    """Renders a full-screen loading dialog that waits for the backend to become ready."""
    with ui.dialog().props('persistent maximized') as loading_dialog:
        with ui.column().classes('w-full h-full items-center justify-center bg-[#151515]'):
            ui.spinner('dots', size='5em', color='primary')
            ui.label('Initializing Models...').classes('text-2xl text-white mt-4 font-bold')
            ui.label('This may take up to 30 seconds').classes('text-lg text-gray-400 mt-2')
    
    loading_dialog.open()

    async def check_backend_ready():
        if await check_health_async(health_url):
            loading_dialog.close()
            health_timer.deactivate()

    health_timer = ui.timer(2.0, check_backend_ready)