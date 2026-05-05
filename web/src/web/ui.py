import os
import base64
import argparse
import httpx
import asyncio
from nicegui import ui
import json

# Parse arguments for port mapping
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8501)
args, _ = parser.parse_known_args()

@ui.page('/')
def index():
    # Read from environment variable, fallback to localhost for local development
    BACKEND_URL = os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000/api/v1/process")

    ui.label("AutoTomeQC Dashboard").classes("text-3xl font-bold mb-6")

    results_container = ui.column()

    async def handle_upload(e):
        results_container.clear()
        with results_container:
            ui.spinner('dots', size='lg')
        
        # --- NiceGUI Version Compatibility ---
        # NiceGUI recently changed their upload API and renamed the file object.
        # This checks for the correct attribute dynamically.
        if hasattr(e, 'content'):
            file_obj = e.content
        elif hasattr(e, 'file'):
            file_obj = e.file
        elif hasattr(e, 'stream'):
            file_obj = e.stream
        else:
            ui.notify(f"Unknown upload format. Attributes available: {dir(e)}", type='negative')
            return
            
        if hasattr(file_obj, 'read'):
            read_result = file_obj.read()
            # If the read method is async (returns a coroutine), we must await it!
            if asyncio.iscoroutine(read_result):
                file_bytes = await read_result
            else:
                file_bytes = read_result
        else:
            file_bytes = file_obj

        file_name = getattr(e, 'name', 'uploaded_image.jpg')
        
        # Convert bytes to base64 so NiceGUI can render it natively
        base64_img = base64.b64encode(file_bytes).decode('utf-8')
        img_src = f"data:image/jpeg;base64,{base64_img}"
        
        try:
            with httpx.Client() as client:
                files = {"file": (file_name, file_bytes, "image/jpeg")}
                response = client.post(BACKEND_URL, files=files, timeout=60.0)
            
            results_container.clear()
            with results_container:
                ui.image(img_src).style("max-width: 600px; margin-bottom: 1rem;")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("qc_summary") == "PASS":
                        ui.label(f"QC Summary: {result['qc_summary']}").classes('text-green-600 text-2xl font-bold')
                    else:
                        ui.label(f"QC Summary: {result['qc_summary']} | Reason: {result['fail_reason']}").classes('text-red-600 text-2xl font-bold')
                    
                    # Display nicely formatted sections breakdown
                    sections = result.get("sections", [])
                    if sections:
                        with ui.card().classes('w-full mt-4'):
                            ui.label("Section Breakdown").classes('text-xl font-bold mb-2')
                            for i, sec in enumerate(sections):
                                sec_status = sec.get('qc_result', 'UNKNOWN')
                                with ui.expansion(f"Section {i} ({sec_status}) - Area: {sec.get('area_in_pixels', 0)}px", icon="science").classes('w-full bg-gray-50 font-semibold'):
                                    criteria = sec.get("criteria", {})
                                    for crit_name, crit_data in criteria.items():
                                        with ui.row().classes('items-center ml-4 mb-1'):
                                            passed = crit_data.get('pass_status')
                                            icon_name = "check_circle" if passed else "cancel"
                                            color = "green" if passed else "red"
                                            ui.icon(icon_name, color=color, size='sm')
                                            ui.label(f"{crit_name}: {crit_data.get('label', 'N/A')}").classes('text-base text-black font-medium')

                    with ui.expansion("Raw JSON Report", icon="data_object").classes('w-full mt-4'):
                        ui.code(json.dumps(result, indent=2), language='json').classes('w-full')
                else:
                    ui.label(f"Backend Error: {response.text}").classes('text-red-600 font-bold')
                    
        except httpx.RequestError:
            results_container.clear()
            with results_container:
                ui.label(f"Failed to connect to the backend at {BACKEND_URL}").classes('text-red-600 font-bold')

    ui.upload(on_upload=handle_upload, label="Upload a section image", auto_upload=True).props('accept=".jpg,.jpeg,.png,.tif,.tiff"')
    
if __name__ in {"__main__", "__mp_main__"}:
    # Start the NiceGUI engine outside the page route
    ui.run(port=args.port, title="AutoTomeQC", show=False, reload=False)
