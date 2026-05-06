from nicegui import ui

def render_inspector_sidebar():
    """Renders the right sidebar and returns the inspector container."""
    with ui.right_drawer(fixed=True).classes('sidebar') as right_drawer:
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('terminal').classes('text-accent text-lg')
                ui.label('Inspector').classes('sidebar-title-text')
        
        # Store reference so the callback can push data here
        inspector_container = ui.column().classes('inspector-content')
        with inspector_container:
            with ui.column().classes('viewport-idle'):
                ui.icon('info', size='2rem')
                ui.label('Select an image or run batch to view informatics')
    
    return right_drawer, inspector_container