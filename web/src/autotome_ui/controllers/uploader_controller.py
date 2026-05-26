# web/controllers/uploader_controller.py
import asyncio
import json
import shutil
import uuid
import logging
import imagesize  # type: ignore[import-untyped]
from nicegui import ui
from pathlib import Path

from autotome_ui.models.status import app_state
from autotome_ui.services.api import analyze_image
from autotome_ui.models.backend_schemas import PipelineResult
from autotome_ui.models.status import QueuedFile


class UploaderController:
    # Update Init parameters
    def __init__(self, add_ui_callback, remove_ui_callback, set_active_ui_callback, refresh_workspace, refresh_inspector):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.add_ui = add_ui_callback
        self.remove_ui = remove_ui_callback
        self.set_active_ui = set_active_ui_callback
        self.refresh_workspace = refresh_workspace
        self.refresh_inspector = refresh_inspector
        self._set_view_state('idle')

    def _set_view_state(self, status, error=None, result=None, raw_json=None):
        """Updates the global state for the active view and triggers UI refreshes."""
        app_state.view.status = status
        app_state.view.error = error
        app_state.view.result = result
        app_state.view.raw_json = raw_json
        if self.set_active_ui:
            self.set_active_ui(app_state.active_file_id)
        
        self.refresh_workspace()
        self.refresh_inspector()
        
    def remove_file(self, file_ids: list[str]):
        # 1. Perform the deletion FIRST
        for file_id in file_ids:
            info = app_state.queued_files.pop(file_id, None)
            if info:
                info.path.unlink(missing_ok=True)
                if info.json_path:
                    info.json_path.unlink(missing_ok=True)
        
        # 2. NOW check if the active file was one of the ones just deleted
        if app_state.active_file_id not in app_state.queued_files:
            # The active file is now missing!
            if app_state.queued_files:
                # There are still files left, pick one
                self.load_next() 
            else:
                # Queue is completely empty
                app_state.active_file_id = None
                self._set_view_state('idle')
        else:
            # The active file still exists, just refresh the UI counters
            self.refresh_workspace()

        # 3. Finally, update the UI component
        self.remove_ui(file_ids)

    def load_result(self, file_id):
        info = app_state.queued_files.get(file_id)
        if not info:
            return
            
        try:
            app_state.active_file_id = file_id

            json_path = info.json_path
            if json_path and json_path.exists():
                with open(json_path, 'r') as f:
                    raw_json = json.load(f)
                result = PipelineResult.model_validate(raw_json)
                self._set_view_state('result', result=result, raw_json=raw_json)
            else:
                self._set_view_state('pending')
        except Exception as e:
            ui.notify(f"Error loading result: {e}", type='negative')

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
        
        app_state.active_file_id = None
    
        for file_id, info in app_state.queued_files.items():
            if not app_state.is_processing:
                break

            if not info:
                continue
                
            if info.status in ['PASS', 'FAIL']: 
                continue
                
            info.status = 'PROCESSING'
            app_state.active_file_id = file_id
            
            # Update view to processing and refresh the sidebar UI
            self._set_view_state('processing')
            
            try:
                result, raw_json = await analyze_image(app_state.process_url, str(info.path))
                
                json_path = info.path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(raw_json, f)
                info.json_path = json_path
                info.status = result.qc_summary
                
                self._set_view_state('result', result=result, raw_json=raw_json)
                    
            except Exception:
                info.status = 'ERROR'
                self._set_view_state('error', error="Backend Error")
                    
            await asyncio.sleep(1.0)

        if app_state.is_processing:
            app_state.is_processing = False
            ui.notify("Batch complete!", type='positive')
        else:
            ui.notify("Processing paused.", type='warning')

    def load_next(self):
        """Finds the active file and shifts state to the next item in the queue."""
        files_list = list(app_state.queued_files.keys())
        if not files_list:
            return

        # Find the ID of the currently active file
        current_id = app_state.active_file_id
        
        if current_id is None or current_id not in files_list:
            # Fallback: if nothing is active, select the first file
            target_id = files_list[0]
        else:
            current_idx = files_list.index(current_id)
            # Calculate next index with wrapping (loops around to 0)
            new_idx = (current_idx + 1) % len(files_list)
            target_id = files_list[new_idx]

        # Use existing controller logic to update active flags, load json, and change view state
        self.load_result(target_id)

    def load_prev(self):
        """Finds the active file and shifts state to the previous item in the queue."""
        files_list = list(app_state.queued_files.keys())
        if not files_list:
            return

        # Find the ID of the currently active file
        current_id = app_state.active_file_id
        
        if current_id is None or current_id not in files_list:
            # Fallback: if nothing is active, select the last file
            target_id = files_list[-1]
        else:
            current_idx = files_list.index(current_id)
            # Calculate previous index with wrapping (loops around to last index)
            new_idx = (current_idx - 1) % len(files_list)
            target_id = files_list[new_idx]

        # Use existing controller logic to update active flags, load json, and change view state
        self.load_result(target_id)

    async def handle_upload(self, data):
        """Unified entry point for both Native (list of paths) and Web (NiceGUI event)."""
        # 1. Determine if we are handling a list of strings (native) or an event (web)
        if app_state.is_native:
            file_items = data  # list of paths
            # For native, we don't need to clear any component
            sender = None 
        else:
            file_items = data.files
            sender = data.sender

        if not file_items:
            return
        
        app_state.view.status = 'uploading'
        self.refresh_workspace()

        # 2. Process each item
        results = []
        for item in file_items:
            # _process_item handles updating app_state.queued_files
            file_id = await self._process_item(item)
            if file_id:
                results.append(file_id)

        # 3. Batch Update
        added_ids = [res for res in results if res is not None]
        if added_ids:
            self.add_ui(added_ids)
            ui.notify(f"Successfully added {len(added_ids)} files", type='positive')

        app_state.view.status = 'idle'
        self.refresh_workspace()

        if sender:
            sender.run_method('removeUploadedFiles')

    async def _process_item(self, item):
        """Generic logic that works for both local paths and upload objects."""
        is_path = isinstance(item, str)
        file_name = Path(item).name if is_path else item.name
        
        # Duplicate Check
        if file_name in {info.name for info in app_state.queued_files.values()}:
            self.logger.warning("Duplicate: %s", file_name)
            return None

        dest_path = app_state.temp_upload_dir / file_name

        # Load and Measure (the part that differs between native and web)
        try:
            if is_path:
                # Native: copy if not exists
                def copy_and_measure():
                    if not dest_path.exists():
                        shutil.copy2(item, dest_path)
                    return imagesize.get(str(dest_path))
                width, height = await asyncio.to_thread(copy_and_measure)
            else:
                # Web: read bytes and save
                file_bytes = await item.read()
                def save_and_measure():
                    with open(dest_path, "wb") as f:
                        f.write(file_bytes)
                    return imagesize.get(str(dest_path))
                width, height = await asyncio.to_thread(save_and_measure)
        except Exception as ex:
            self.logger.error("Failed to process %s: %s", file_name, ex)
            return None

        # Update State
        file_id = uuid.uuid4().hex
        app_state.queued_files[file_id] = QueuedFile(
            name=file_name, path=dest_path, img_src=f"/temp_uploads/{file_name}",
            status='PENDING', width=width, height=height
        )
        return file_id  