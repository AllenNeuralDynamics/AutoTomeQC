import asyncio
import base64
import json
import uuid
from pathlib import Path
from nicegui import ui

from web.models.status import app_state
from web.services.api import analyze_image
from web.models.schemas import PipelineResult, QueuedFile
from web.protocol.events import image_selected, image_pending, image_error, clear_views

class UploaderController:
    """Handles the state and logic for the batch uploader and queue processing."""
    
    def __init__(self,):
        # UI Containers (Injected later by the Orchestrator)
        self.queue_container = None
        self.empty_state = None

    def remove_file(self, file_id):
        info = app_state.queued_files.pop(file_id, None)
        if info:
            is_active = 'bg-[#1A1A1A]' in info.row_ui.classes
            info.row_ui.delete()
            info.path.unlink(missing_ok=True)
            if info.json_path:
                info.json_path.unlink(missing_ok=True)
                
            # Clear the main workspace if we deleted the image we were actively viewing
            if is_active or not app_state.queued_files:
                clear_views.emit(None)
                        
        if not app_state.queued_files:
            self.empty_state.set_visibility(True)
            
    def load_result(self, file_id):
        info = app_state.queued_files.get(file_id)
        if not info: return
            
        try:
            for f_info in app_state.queued_files.values():
                f_info.row_ui.classes(remove='active')
            info.row_ui.classes(add='active')

            json_path = info.json_path
            if json_path and json_path.exists():
                with open(json_path, 'r') as f:
                    raw_json = json.load(f)
                result = PipelineResult.model_validate(raw_json)
                
                # Emit success event
                image_selected.emit((info.path, result, raw_json))
            else:
                # Emit pending event
                image_pending.emit(info.img_src)
        except Exception as e:
            ui.notify(f"Error loading result: {e}", type='negative')

    def build_file_row(self, file_id, file_name, file_path, img_src):
        self.empty_state.set_visibility(False)
        with self.queue_container:
            with ui.row().classes('queue-item shrink-0').on('click', lambda e, fid=file_id: self.load_result(fid)) as row_ui:
                
                with ui.element('div').classes('queue-thumb'):
                    ui.image(img_src).classes('queue-img')
                
                with ui.element('div').classes('queue-details'):
                    ui.label(file_name).classes('queue-filename')
                    with ui.row().classes('queue-status-row'):
                        spinner = ui.spinner('dots', size='1em', color='blue-400')
                        spinner.set_visibility(False)
                        status_label = ui.label('PENDING').classes('queue-status-text')
                
                delete_btn = ui.button(icon='delete', color='red') \
                    .props('flat dense') \
                    .classes('btn-delete') \
                    .on('click.stop', lambda e, fid=file_id: self.remove_file(fid))

        app_state.queued_files[file_id] = QueuedFile(
            name=file_name,
            path=file_path,
            img_src=img_src,
            row_ui=row_ui,
            status_label=status_label,
            spinner=spinner,
            delete_btn=delete_btn,
        )

    async def handle_upload(self, e):
        if hasattr(e, 'content'): file_obj = e.content
        elif hasattr(e, 'file'): file_obj = e.file
        elif hasattr(e, 'stream'): file_obj = e.stream
        else:
            ui.notify(f"Unknown upload format. Attributes available: {dir(e)}", type='negative')
            return
            

        raw_name = getattr(e, 'name', None) or \
                   getattr(e, 'filename', None) or \
                   getattr(file_obj, 'name', None) or \
                   getattr(file_obj, 'filename', None)
                   
        file_name = str(raw_name) if raw_name else None
        
        if not file_name:
            file_name = f"image_{uuid.uuid4().hex[:6]}.jpg"
            ui.notify(f"Browser stripped filename. Used: {file_name}", type='warning')
            
        if hasattr(file_obj, 'read'):
            read_result = file_obj.read()
            file_bytes = await read_result if asyncio.iscoroutine(read_result) else read_result
        else:
            file_bytes = file_obj

        file_id = uuid.uuid4().hex
        file_path = app_state.temp_upload_dir / file_name
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
        ext = file_name.lower().split('.')[-1]
        mime_type = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"
        base64_img = base64.b64encode(file_bytes).decode('utf-8')
        img_src = f"data:{mime_type};base64,{base64_img}"
        
        self.build_file_row(file_id, file_name, file_path, img_src)
        
        # Clear the browser's internal file queue so the exact same file can be re-uploaded if deleted
        e.sender.run_method('removeUploadedFiles')

    async def process_batch(self, e):
        if not app_state.queued_files:
            ui.notify("Please upload images first.", type='warning')
            return
            
        e.sender.disable()
        for info in app_state.queued_files.values():
            info.delete_btn.set_visibility(False)
            
        ui.notify("Processing images...")
        
        for file_id, info in app_state.queued_files.items():
            if info.status_label.text in ['PASS', 'FAIL']: continue
                
            for f_info in app_state.queued_files.values():
                f_info.row_ui.classes(remove='active')
            info.row_ui.classes(add='active')
            info.status_label.set_text('PROCESSING')
            info.status_label.style('color: #60a5fa !important')
            info.spinner.set_visibility(True)
            image_pending.emit(None) # Signal spinner
            
            try:
                # Pass the temporary file path directly to the API
                result, raw_json = await analyze_image(app_state.process_url, str(info.path))
                
                json_path = info.path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(raw_json, f)
                info.json_path = json_path
                
                image_selected.emit((info.path, result, raw_json))
                status = result.qc_summary
                info.status_label.set_text(status)
                info.status_label.style(f'color: var(--{"pass" if status == "PASS" else "fail"}-color) !important')
            except Exception as exc:
                info.status_label.set_text('ERROR')
                info.status_label.style('color: var(--fail-color) !important')
                image_error.emit("Backend Error")
                    
            info.spinner.set_visibility(False)        
            await asyncio.sleep(1.0)
            
        e.sender.enable()
        for info in app_state.queued_files.values():
            info.delete_btn.set_visibility(True)
        ui.notify("Batch processing complete!", type='positive')