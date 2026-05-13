# web/controllers/state_controller.py
import asyncio

from web.services.api import fetch_config_async, is_running_async
from web.models.status import app_state
from web.events import fetch_config, is_running

@fetch_config.subscribe
async def _on_fetch_config(config_url: str):
    app_state.config = await fetch_config_async(config_url)
    app_state.is_backend_ready = True


@is_running.subscribe
async def _on_is_running(is_running_url: str):
    while True:
        is_running = await is_running_async(is_running_url)
        if is_running:
            break
        else:
            await asyncio.sleep(1)