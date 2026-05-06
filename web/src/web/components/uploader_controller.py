import asyncio
import base64
import httpx
import json
import tempfile
import uuid
from pathlib import Path
from nicegui import ui

from web.services.api import analyze_image
from web.components.results_card import display_qc_result
from web.protocol.schemas import PipelineResult

class UploaderController:
    """Handles the state and logic for the batch uploader and queue processing."""
    
    def __init__(self, backend_url, image_container, inspector_container, queue_container, empty_state):
        self.backend_url = backend_url
        self.image_container = image_container
        self.inspector_container = inspector_container
        self.queue_container = queue_container
        self.empty_state = empty_state
        
        # State tracking
        self.temp_dir = Path(tempfile.mkdtemp(prefix="autotome_"))
        self.queued_files = {}

    def remove_file(self, file_id):
        info = self.queued_files.pop(file_id, None)
        if info:
            is_active = 'bg-[#1A1A1A]' in info['row_ui'].classes
            info['row_ui'].delete()
            info['path'].unlink(missing_ok=True)
            if info.get('json_path'):
                info['json_path'].unlink(missing_ok=True)
                
            # Clear the main workspace if we deleted the image we were actively viewing
            if is_active or not self.queued_files:
                self.image_container.clear()
                with self.image_container:
                    with ui.column().classes('viewport-idle'):
                        ui.icon('aspect_ratio', size='6rem')
                        ui.label('VIEWPORT_IDLE')
                self.inspector_container.clear()
                with self.inspector_container:
                    with ui.column().classes('viewport-idle'):
                        ui.icon('info', size='2rem')
                        ui.label('Select an image or run batch to view informatics')
                        
        if not self.queued_files:
            self.empty_state.set_visibility(True)
            
    def load_result(self, file_id):
        info = self.queued_files.get(file_id)
        if not info: return
            
        try:
            for f_info in self.queued_files.values():
                f_info['row_ui'].classes(add='active')

            self.image_container.clear()
            self.inspector_container.clear()
            
            json_path = info.get('json_path')
            if json_path and json_path.exists():
                with open(json_path, 'r') as f:
                    raw_json = json.load(f)
                result = PipelineResult.model_validate(raw_json)
                display_qc_result(result, raw_json, info['img_src'], self.image_container, self.inspector_container)
            else:
                with self.image_container:
                    with ui.element('div').classes('image-wrapper'):
                        ui.image(info['img_src']).classes('image-preview')
                with self.inspector_container:
                    with ui.column().classes('viewport-idle'):
                        ui.icon('info', size='2rem')
                        ui.label('Image pending processing...')
        except Exception as e:
            ui.notify(f"Error loading result: {e}", type='negative')

    def build_file_row(self, file_id, file_name, file_path, img_src):
        self.empty_state.set_visibility(False)
        with self.queue_container:
            with ui.row().classes('queue-item').on('click', lambda e, fid=file_id: self.load_result(fid)) as row_ui:
                
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

        self.queued_files[file_id] = {
            'name': file_name,
            'path': file_path,
            'json_path': None,
            'img_src': img_src,
            'row_ui': row_ui,
            'status_label': status_label,
            'spinner': spinner,
            'delete_btn': delete_btn
        }

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
        unique_filename = f"{file_id}_{file_name}"
        file_path = self.temp_dir / unique_filename
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
        if not self.queued_files:
            ui.notify("Please upload images first.", type='warning')
            return
            
        e.sender.disable()
        for info in self.queued_files.values():
            info['delete_btn'].set_visibility(False)
            
        ui.notify("Processing images...")
        
        for file_id, info in self.queued_files.items():
            if info['status_label'].text in ['PASS', 'FAIL']: continue
                
            for f_info in self.queued_files.values():
                f_info['row_ui'].classes(remove='active')
            info['row_ui'].classes(add='active')
            info['status_label'].set_text('PROCESSING')
            info['status_label'].style('color: #60a5fa !important')
            info['spinner'].set_visibility(True)
            
            self.image_container.clear()
            self.inspector_container.clear()
            with self.image_container:
                ui.spinner('dots', size='lg')
            
            with open(info['path'], "rb") as f:
                file_bytes = f.read()
            
            try:
                result, raw_json = await analyze_image(self.backend_url, info['name'], file_bytes)
                
                json_path = info['path'].with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(raw_json, f)
                info['json_path'] = json_path
                
                display_qc_result(result, raw_json, info['img_src'], self.image_container, self.inspector_container)
                status = result.qc_summary
                info['status_label'].set_text(status)
                info['status_label'].style(f'color: var(--{"pass" if status == "PASS" else "fail"}-color) !important')
            except Exception as exc:
                info['status_label'].set_text('ERROR')
                info['status_label'].style('color: var(--fail-color) !important')
                self.inspector_container.clear()
                with self.inspector_container:
                    ui.label(f"Backend Error").classes('text-red-600 font-bold')
                    
            info['spinner'].set_visibility(False)        
            await asyncio.sleep(1.0)
            
        e.sender.enable()
        for info in self.queued_files.values():
            info['delete_btn'].set_visibility(True)
        ui.notify("Batch processing complete!", type='positive')