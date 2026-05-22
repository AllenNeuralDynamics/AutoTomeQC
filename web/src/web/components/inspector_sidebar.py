# web/components/inspector_sidebar.py
from nicegui import ui
import json
import logging
from web.models.backend_schemas import PipelineResult
from web.models.status import app_state

logger = logging.getLogger(__name__)

def render_inspector_sidebar(toggle_masks_callback=None):
    """Renders the right sidebar structure."""
    with ui.right_drawer(fixed=True).classes('sidebar'):
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('terminal').classes('text-accent text-lg')
                ui.label('Inspector').classes('sidebar-title-text')

            # mask visiblity button
            visibility_btn = ui.button(on_click=toggle_masks_callback).props('flat dense round size=sm')
            visibility_btn.bind_icon_from(app_state.view, 'show_masks', 
                               backward=lambda val: 'visibility' if val else 'visibility_off')
            visibility_btn.bind_visibility_from(app_state.view, 'status', backward=lambda s: s == 'result')
        
        # Render the reactive content inside the drawer
        inspector_content()

@ui.refreshable
def inspector_content():
    """Reactively renders the inspector content based on app_state."""
    inspector_container = ui.column().classes('inspector-content w-full h-full')
    status = getattr(app_state.view, 'status', 'idle')
    
    with inspector_container:
        if status == 'idle':
            with ui.column().classes('viewport-idle items-center justify-center p-8 text-center text-gray-500'):
                ui.icon('info', size='2rem')
                ui.label('Select an image or run batch to view informatics')
                
        elif status == 'pending':
            with ui.column().classes('viewport-idle items-center justify-center p-8 text-center text-gray-500'):
                ui.icon('info', size='2rem')
                ui.label('Pending image processing...')

        elif status == 'processing':
            with ui.column().classes('viewport-idle items-center justify-center p-8 text-center text-gray-500'):
                ui.icon('info', size='2rem')
                ui.label('Processing image...')

        elif status == 'error':
            msg = getattr(app_state.view, 'error', 'An unknown error occurred')
            ui.label(msg).classes('text-red-600 font-bold')
            
        elif status == 'result':
            result = getattr(app_state.view, 'result', None)
            raw_json = getattr(app_state.view, 'raw_json', {})
            if result:
                _display_qc_result(result, raw_json)

def _display_qc_result(result: PipelineResult, raw_json: dict):
    """Renders the QC results breakdown on the screen."""
    colors = app_state.view.section_colors

    with ui.column().classes('inspector-list'):
        
        # Header Summary
        with ui.column().classes('summary-card'):
            with ui.row().classes('summary-row'):
                ui.label('Status').classes('metric-label')
                badge_class = 'badge-pass' if result.qc_summary == 'PASS' else 'badge-fail'
                ui.label(result.qc_summary).classes(f'status-badge {badge_class}')
            
            with ui.row().classes('summary-row'):
                ui.label('Time').classes('metric-label')
                ui.label(f"{result.processing_time_sec}s").classes('metric-value')

        # Sections Details
        if result.sections:
            for i, sec in enumerate(result.sections):
                stroke_color, _ = colors[i % len(colors)]

                with ui.column().classes('section-container'):
                    with ui.row().classes('section-header'):
                        ui.label(f'SECTION {i + 1}') \
                          .classes('badge-section') \
                          .style(f'background-color: {stroke_color}; border-color: {stroke_color}; color: white;')
                        ui.element('div').classes('section-divider')
                    
                    # Metric Rows
                    with ui.column().classes('metrics-grid'):
                        for crit_name, crit_data in sec.criteria.items():
                            with ui.row().classes('metric-row'):
                                with ui.row().classes('metric-content'):
                                    icon_name = 'check_circle' if crit_data.pass_status else 'cancel'
                                    color_class = 'text-pass' if crit_data.pass_status else 'text-fail'
                                    ui.icon(icon_name).classes(f'metric-icon {color_class}')
                                    with ui.column().classes('metric-text'):
                                        ui.label(crit_name).classes('metric-name')
                                        ui.label(str(crit_data.label)).classes('metric-value-bold')
                                
                                if crit_data.conf is not None and crit_data.conf > 0:
                                    with ui.column().classes('metric-conf'):
                                        ui.label('CONF').classes('conf-label')
                                        ui.label(f"{int(crit_data.conf * 100)}%").classes('conf-value')

        with ui.expansion("Raw JSON Report", icon="data_object").classes('json-expansion'):
            # Remove the heavy mask data from the display JSON to keep it readable
            display_json = dict(raw_json)
            if 'sections' in display_json:
                display_json['sections'] = [
                    {k: v for k, v in sec.items() if k != 'mask'}
                    for sec in display_json['sections']
                ]
            ui.code(json.dumps(display_json, indent=2), language='json').classes('json-code')
