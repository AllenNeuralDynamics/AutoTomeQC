# web/controllers/uploader_controller.py
import asyncio
import base64
import json
import uuid
from nicegui import ui

from web.models.status import app_state
from web.services.api import analyze_image
from web.models.backend_schemas import PipelineResult
from web.models.status import QueuedFile

class UploaderController:
    """Handles the state and logic. Mutates app_state and triggers UI refreshes."""
    
    def __init__(self, refresh_ui_callback, refresh_workspace, refresh_inspector):
        self.refresh_ui = refresh_ui_callback
        self.refresh_workspace = refresh_workspace
        self.refresh_inspector = refresh_inspector
        self.set_view_state('idle')

    def set_view_state(self, status, error=None, result=None, raw_json=None):
        """Updates the global state for the active view and triggers UI refreshes."""
        app_state.view_status = status
        app_state.view_error = error
        app_state.view_result = result
        app_state.view_raw_json = raw_json
        
        self.refresh_workspace()
        self.refresh_inspector()

    def remove_file(self, file_id):
        info = app_state.queued_files.pop(file_id, None)
        if info:
            info.path.unlink(missing_ok=True)
            if info.json_path:
                info.json_path.unlink(missing_ok=True)
                
            # Clear views if the deleted item was currently active
            if info.is_active or not app_state.queued_files:
                self.set_view_state('idle')
                
        self.refresh_ui()
            
    def load_result(self, file_id):
        info = app_state.queued_files.get(file_id)
        if not info: return
            
        try:
            # Update active flags
            for f_info in app_state.queued_files.values():
                f_info.is_active = False
            info.is_active = True

            json_path = info.json_path
            if json_path and json_path.exists():
                with open(json_path, 'r') as f:
                    raw_json = json.load(f)
                result = PipelineResult.model_validate(raw_json)
                self.set_view_state('result', result=result, raw_json=raw_json)
            else:
                self.set_view_state('pending')
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
        # If already running, this click means "Stop/Pause"
        if app_state.is_processing:
            app_state.is_processing = False
            return
    
        if not app_state.queued_files:
            ui.notify("Please upload images first.", type='warning')
            return
            
        app_state.is_processing = True
        ui.notify("Processing images...")
        
        # 1. Clear active state from ALL items exactly ONCE before the loop starts
        for f_info in app_state.queued_files.values():
            f_info.is_active = False
            
        previous_info = None # Track the previously active item
        
        for file_id, info in app_state.queued_files.items():
            if not app_state.is_processing:
                print("[debug] Stop signal detected. Breaking loop.")
                break

            if info.status in ['PASS', 'FAIL']: 
                continue
                
            info.status = 'PROCESSING'
            
            # 2. Effortlessly switch active states without an inner loop
            if previous_info:
                previous_info.is_active = False # Turn off the old one
                
            info.is_active = True # Turn on the new one
            previous_info = info  # Remember this one for the next cycle
            
            # Update view to processing and refresh the sidebar UI
            self.set_view_state('processing')
            
            try:
                result, raw_json = await analyze_image(app_state.process_url, str(info.path))
                
                json_path = info.path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(raw_json, f)
                info.json_path = json_path
                info.status = result.qc_summary
                
                self.set_view_state('result', result=result, raw_json=raw_json)
                    
            except Exception as exc:
                info.status = 'ERROR'
                self.set_view_state('error', error="Backend Error")
                    
            await asyncio.sleep(1.0)
        
        # Final cleanup at the end of process_batch loop
        if app_state.is_processing:
            app_state.is_processing = False
            self.refresh_ui() 
            ui.notify("Batch complete!", type='positive')  # Green
        else:
            self.refresh_ui() 
            ui.notify("Processing paused.", type='warning')  # Yellow