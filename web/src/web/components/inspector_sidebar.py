from nicegui import ui
from web.models.schemas import PipelineResult
import json
from nicegui import ui
from web.models.schemas import PipelineResult

def set_inspector_idle(inspector_container):
    inspector_container.clear()
    with inspector_container:
        with ui.column().classes('viewport-idle'):
            ui.icon('info', size='2rem')
            ui.label('Select an image or run batch to view informatics')

def set_inspector_pending(inspector_container):
    inspector_container.clear()
    with inspector_container:
        with ui.column().classes('viewport-idle'):
            ui.icon('info', size='2rem')
            ui.label('Image pending processing...')

def set_inspector_error(inspector_container, msg):
    inspector_container.clear()
    with inspector_container:
        ui.label(msg).classes('text-red-600 font-bold')

def render_inspector_sidebar():
    """Renders the right sidebar and returns the inspector container."""
    with ui.right_drawer(fixed=True).classes('sidebar') as right_drawer:
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('terminal').classes('text-accent text-lg')
                ui.label('Inspector').classes('sidebar-title-text')
        
        # Store reference so the callback can push data here
        inspector_container = ui.column().classes('inspector-content')
        set_inspector_idle(inspector_container)
    
    return right_drawer, inspector_container

def update_inspector_sidebar(inspector_container, result: PipelineResult, raw_json: dict):
    """Updates the inspector panel with new QC results."""
    inspector_container.clear()
    with inspector_container:
        display_qc_result(result, raw_json)

def display_qc_result(result: PipelineResult, raw_json: dict):
    """Renders the QC results breakdown on the screen."""
    
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
                with ui.column().classes('section-container'):
                    with ui.row().classes('section-header'):
                        ui.label(f'SECTION {i + 1}').classes('badge-section')
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

                    # System Notes
                    with ui.column().classes('system-notes'):
                        ui.label('System Notes').classes('notes-title')
                        if shape_crit := sec.criteria.get('shape'):
                            if shape_crit.message:
                                ui.label(f'"{shape_crit.message}"').classes('notes-text')
                        if knife_crit := sec.criteria.get('knife_mark'):
                            if knife_crit.reason:
                                ui.label(f"! {knife_crit.reason}").classes('notes-error')

        with ui.expansion("Raw JSON Report", icon="data_object").classes('json-expansion'):
            # Remove the heavy mask data from the display JSON to keep it readable
            display_json = dict(raw_json)
            if 'sections' in display_json:
                display_json['sections'] = [
                    {k: v for k, v in sec.items() if k != 'mask'}
                    for sec in display_json['sections']
                ]
            ui.code(json.dumps(display_json, indent=2), language='json').classes('json-code')
