from nicegui import ui

def render_header(left_drawer):
    """Renders the top application header."""
    with ui.header().classes('app-header').classes(remove='bg-primary'):
        with ui.row().classes('header-left'):
            ui.button(icon='chevron_left', color=None, on_click=lambda: left_drawer.toggle()).props('flat dense').classes('btn-icon')

        with ui.row().classes('header-right'):
            btn_config = ui.button('CONFIG',icon='settings', color=None,
                                   on_click=lambda: ui.notify('Config clicked')).classes('btn-config')
            btn_export = ui.button('EXPORT', icon='download', color=None,
                                   on_click=lambda: ui.notify('Export clicked')).classes('btn-export')
