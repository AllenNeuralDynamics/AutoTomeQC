from nicegui import ui
from web.protocol.events import config_requested, export_requested

def render_header(left_drawer):
    """Renders the top application header."""
    with ui.header().classes('app-header').classes(remove='bg-primary'):
        with ui.row().classes('header-left'):
            ui.button(icon='chevron_left', color=None, on_click=lambda: left_drawer.toggle()).props('flat dense').classes('btn-icon')

        with ui.row().classes('header-right'):
            btn_config = ui.button('CONFIG',icon='settings', color=None,
                                   on_click=lambda: config_requested.emit(None)).classes('btn-config')
            btn_export = ui.button('EXPORT', icon='download', color=None,
                                   on_click=lambda: export_requested.emit(None)).classes('btn-export')
