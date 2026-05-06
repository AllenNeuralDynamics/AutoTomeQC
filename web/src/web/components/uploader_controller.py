import asyncio
import base64
import httpx
import tempfile
import uuid
from pathlib import Path
from nicegui import ui

from web.services.api import analyze_image
from web.components.results_card import display_qc_result

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
            info['row_ui'].delete()
            info['path'].unlink(missing_ok=True)
        if not self.queued_files:
            self.empty_state.set_visibility(True)

    def build_file_row(self, file_id, file_name, file_path, img_src):
        self.empty_state.set_visibility(False)
        with self.queue_container:
            with ui.row().classes('group relative flex items-center gap-3 p-3 rounded cursor-pointer transition-all border border-transparent hover:bg-[#151515] hover:border-[#333333] w-full flex-nowrap overflow-hidden') as row_ui:
                
                with ui.element('div').classes('w-10 h-10 rounded overflow-hidden border border-[#333333] shrink-0 bg-black flex items-center justify-center'):
                    ui.image(img_src).classes('w-full h-full object-cover opacity-80')
                
                with ui.element('div').classes('flex-1 min-w-0'):
                    ui.label(file_name).classes('text-xs font-mono truncate uppercase tracking-tight text-white')
                    with ui.row().classes('items-center gap-2 mt-1 m-0 p-0'):
                        spinner = ui.spinner('dots', size='1em', color='blue-400')
                        spinner.set_visibility(False)
                        status_label = ui.label('PENDING').classes('text-[10px] text-gray-500 uppercase font-bold')
                
                delete_btn = ui.button(icon='delete', color='red', on_click=lambda: self.remove_file(file_id)) \
                    .props('flat dense') \
                    .classes('opacity-0 group-hover:opacity-100 transition-all absolute right-2 bg-[#151515] z-10')

        self.queued_files[file_id] = {
            'name': file_name,
            'path': file_path,
            'img_src': img_src,
            'row_ui': row_ui,
            'status_label': status_label,
            'spinner': spinner,
            'delete_btn': delete_btn
        }

    async def handle_upload(self, e):
        file_name = getattr(e, 'name', 'uploaded_image.jpg')
        
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            ui.notify(f"Skipped {file_name}: Only image files are allowed.", type='warning')
            return
            
        if hasattr(e, 'content'): file_obj = e.content
        elif hasattr(e, 'file'): file_obj = e.file
        elif hasattr(e, 'stream'): file_obj = e.stream
        else:
            ui.notify(f"Unknown upload format. Attributes available: {dir(e)}", type='negative')
            return
            
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
                f_info['row_ui'].classes(add='border-transparent hover:bg-[#151515] hover:border-[#333333]', remove='bg-[#1A1A1A] border-[#F27D26]/30')
            info['row_ui'].classes(add='bg-[#1A1A1A] border-[#F27D26]/30', remove='border-transparent hover:bg-[#151515] hover:border-[#333333]')
            info['status_label'].set_text('PROCESSING')
            info['status_label'].classes(add='text-blue-400', remove='text-gray-500 text-red-500 text-green-500')
            info['spinner'].set_visibility(True)
            
            self.image_container.clear()
            self.inspector_container.clear()
            with self.image_container:
                ui.spinner('dots', size='lg')
            
            with open(info['path'], "rb") as f:
                file_bytes = f.read()
            
            try:
                result, raw_json = await analyze_image(self.backend_url, info['name'], file_bytes)
                display_qc_result(result, raw_json, info['img_src'], self.image_container, self.inspector_container)
                status = result.qc_summary
                info['status_label'].set_text(status)
                info['status_label'].classes(add=f"text-{'green' if status == 'PASS' else 'red'}-500", remove='text-blue-400 text-gray-500')
            except Exception as exc:
                info['status_label'].set_text('ERROR')
                info['status_label'].classes(add='text-red-500', remove='text-blue-400 text-gray-500')
                self.inspector_container.clear()
                with self.inspector_container:
                    ui.label(f"Backend Error").classes('text-red-600 font-bold')
                    
            info['spinner'].set_visibility(False)        
            await asyncio.sleep(1.0)
            
        e.sender.enable()
        for info in self.queued_files.values():
            info['delete_btn'].set_visibility(True)
        ui.notify("Batch processing complete!", type='positive')