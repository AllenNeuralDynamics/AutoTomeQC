# web/controllers/state_controller.py
import asyncio
import logging
from autotome_ui.services.api import fetch_config_async, is_running_async
from autotome_ui.models.status import app_state

logger = logging.getLogger(__name__)

async def wait_backend_ready():
    while True:
        is_running = await is_running_async(app_state.is_ready_url)
        if is_running:
            break
        else:
            await asyncio.sleep(1)

async def on_fetch_config():
    app_state.config = await fetch_config_async(app_state.config_url)
    app_state.is_backend_ready = True

def on_toggle_masks():
    app_state.view.show_masks = not app_state.view.show_masks