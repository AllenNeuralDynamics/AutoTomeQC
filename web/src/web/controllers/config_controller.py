from nicegui import ui
from web.protocol.events import config_requested
from web.models.status import app_state

def show_config():
    @config_requested.subscribe
    def _show_config(_=None):
        if app_state.is_backend_ready and app_state.config:
            ui.notify(str(app_state.config)) #TODO no ui
        else:
            ui.notify("Configuration not loaded yet.", type='warning')


