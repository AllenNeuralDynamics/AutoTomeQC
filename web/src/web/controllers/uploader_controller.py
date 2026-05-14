# web/controllers/uploader_controller.py
import asyncio
import base64
import json
import uuid
from nicegui import ui

from web.models.status import app_state
from web.services.api import analyze_image
from web.models.schemas import PipelineResult, QueuedFile
from web.protocol.events import image_selected, image_pending, image_error, clear_views

class UploaderController:
    """Handles the state and logic. Does NOT manipulate UI elements directly."""
    
    def __init__(self, refresh_ui_callback):
        self.refresh_ui = refresh_ui_callback

    def remove_file(self, file_id):
        info = app_state.queued_files.pop(file_id, None)
        if info:
            info.path.unlink(missing_ok=True)
            if info.json_path:
                info.json_path.unlink(missing_ok=True)
                
            if info.is_active or not app_state.queued_files:
                clear_views.emit(None)
                
        self.refresh_ui()
            
    def load_result(self, file_id):
        info = app_state.queued_files.get(file_id)
        if not info: return
            
        try:
            # Update state
            for f_info in app_state.queued_files.values():
                f_info.is_active = False
            info.is_active = True
            
            self.refresh_ui()

            json_path = info.json_path
            if json_path and json_path.exists():
                with open(json_path, 'r') as f:
                    raw_json = json.load(f)
                result = PipelineResult.model_validate(raw_json)
                image_selected.emit((info.path, result, raw_json))
            else:
                image_pending.emit(info.img_src)
        except Exception as e:
            ui.notify(f"Error loading result: {e}", type='negative')

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
        
        # Instantiate pure data model instead of UI elements
        app_state.queued_files[file_id] = QueuedFile(
            name=file_name,
            path=file_path,
            img_src=img_src,
            status='PENDING',
            is_active=False
        )
        
        self.refresh_ui()
        e.sender.run_method('removeUploadedFiles')

    async def process_batch(self, e):
        if not app_state.queued_files:
            ui.notify("Please upload images first.", type='warning')
            return
            
        e.sender.disable()
        ui.notify("Processing images...")
        
        for file_id, info in app_state.queued_files.items():
            if info.status in ['PASS', 'FAIL']: continue
                
            # Update state for current processing item
            for f_info in app_state.queued_files.values():
                f_info.is_active = False
            info.is_active = True
            info.status = 'PROCESSING'
            self.refresh_ui()
            image_pending.emit(None) 
            
            try:
                result, raw_json = await analyze_image(app_state.process_url, str(info.path))
                
                json_path = info.path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(raw_json, f)
                info.json_path = json_path
                
                image_selected.emit((info.path, result, raw_json))
                info.status = result.qc_summary
            except Exception as exc:
                info.status = 'ERROR'
                image_error.emit("Backend Error")
                    
            self.refresh_ui()        
            await asyncio.sleep(1.0)
            
        e.sender.enable()
        ui.notify("Batch processing complete!", type='positive')