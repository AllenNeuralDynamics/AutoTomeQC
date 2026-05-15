import json
from nicegui import ui
from web.models.status import app_state

def render_header(left_drawer):
    """Renders the top application header."""
    
    with ui.dialog() as config_dialog, ui.card().classes('w-[700px] max-w-4xl p-6 bg-[#1a1a1a] border border-[#333333]'):
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('System Configuration').classes('text-xl font-bold text-white')
            ui.button(icon='close', on_click=config_dialog.close).props('flat round dense').classes('text-gray-400')
        
        # Fixed max-h-[500px] and added overflow-auto via custom-scrollbar to keep it compact and scannable
        config_display = ui.code('', language='json').classes('w-full text-sm custom-scrollbar max-h-[500px] overflow-auto')

        with ui.row().classes('w-full justify-end mt-4'):
            ui.button('CLOSE', on_click=config_dialog.close).classes('bg-neutral-700 text-white')

    def handle_config_click():
        try:
            if app_state.config is not None:
                # If it's your Pydantic AppConfig model
                if hasattr(app_state.config, 'model_dump_json'):
                    config_str = app_state.config.model_dump_json(indent=2)
                elif isinstance(app_state.config, dict):
                    config_str = json.dumps(app_state.config, indent=2)
                else:
                    config_str = str(app_state.config)
            else:
                config_str = "No system configuration loaded yet from backend."
        except Exception as e:
            config_str = f"Error reading system config layout: {e}"

        # Push the fresh configuration string to the browser DOM element
        config_display.set_content(config_str)
        config_dialog.open()

    # Render the top header layout
    with ui.header().classes('app-header').classes(remove='bg-primary'):
        with ui.row().classes('header-left'):
            ui.button(icon='chevron_left', color=None, on_click=lambda: left_drawer.toggle()).props('flat dense').classes('btn-icon')

        with ui.row().classes('header-right'):
            btn_config = ui.button('CONFIG', icon='settings', color=None,
                                   on_click=handle_config_click).classes('btn-config')
            
            btn_export = ui.button('EXPORT', icon='download', color=None,
                                   on_click=lambda: ui.notify("Export button clicked")).classes('btn-export')