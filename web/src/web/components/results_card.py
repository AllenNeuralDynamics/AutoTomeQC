#Components (Presentation): Reusable blocks of UI code. 
# They take data and draw it on the screen, but they don't fetch the data themselves.
import json
from nicegui import ui
from web.protocol.schemas import PipelineResult

def display_qc_result(result: PipelineResult, raw_json: dict, img_src: str, image_container, inspector_container):
    """Renders the QC results breakdown on the screen."""
    
    # 1. UPDATE MAIN WORKSPACE (Image)
    image_container.clear()
    with image_container:
        with ui.element('div').classes('image-wrapper'):
            ui.image(img_src).classes('image-preview')
            
            with ui.element('div').classes('image-overlay-text'):
                ui.label('XY: 0,0')
                ui.label('Format: JPEG 8bit')
                ui.label('Sensor: AutoTome-Cam-01')

    # 2. UPDATE INSPECTOR PANEL
    inspector_container.clear()
    with inspector_container:
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
                ui.code(json.dumps(raw_json, indent=2), language='json').classes('json-code')
