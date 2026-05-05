#Components (Presentation): Reusable blocks of UI code. 
# They take data and draw it on the screen, but they don't fetch the data themselves.
import json
from nicegui import ui
from web.protocol.schemas import PipelineResult

def display_qc_result(result: PipelineResult, raw_json: dict, img_src: str):
    """Renders the QC results breakdown on the screen."""
    
    ui.image(img_src).style("max-width: 600px; margin-bottom: 1rem;")
    
    if result.qc_summary == "PASS":
        ui.label(f"QC Summary: {result.qc_summary}").classes('text-green-600 text-2xl font-bold')
    else:
        ui.label(f"QC Summary: {result.qc_summary} | Reason: {result.fail_reason}").classes('text-red-600 text-2xl font-bold')
    
    # Display nicely formatted sections breakdown
    if result.sections:
        with ui.card().classes('w-full mt-4'):
            ui.label("Section Breakdown").classes('text-xl font-bold mb-2')
            for i, sec in enumerate(result.sections):
                with ui.expansion(f"Section {i} ({sec.qc_result}) - Area: {sec.area_in_pixels}px", icon="science").classes('w-full bg-gray-50 font-semibold'):
                    for crit_name, crit_data in sec.criteria.items():
                        with ui.row().classes('items-center ml-4 mb-1'):
                            icon_name = "check_circle" if crit_data.pass_status else "cancel"
                            color = "green" if crit_data.pass_status else "red"
                            ui.icon(icon_name, color=color, size='sm')
                            ui.label(f"{crit_name}: {crit_data.label}").classes('text-base text-black font-medium')

    with ui.expansion("Raw JSON Report", icon="data_object").classes('w-full mt-4'):
        ui.code(json.dumps(raw_json, indent=2), language='json').classes('w-full')
