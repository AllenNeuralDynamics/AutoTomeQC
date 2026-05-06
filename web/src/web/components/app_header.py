from nicegui import ui

def render_header(left_drawer):
    """Renders the top application header."""
    with ui.header().classes('app-header').classes(remove='bg-primary'):
        with ui.row().classes('header-left'):
            ui.button(icon='chevron_left', color=None, on_click=lambda: left_drawer.toggle()).props('flat dense').classes('btn-icon')
            ui.element('div').classes('header-divider')
            with ui.row().classes('project-title-container'):
                ui.label('PROJECT').classes('text-accent')
                ui.label('/')
                ui.label('UNTITLED_PROJECT').classes('text-title')
        
        with ui.row().classes('header-right'):
            ui.button('CONFIG', icon='settings', color=None).classes('btn-config')
            ui.button('EXPORT', icon='download', color=None).classes('btn-export')