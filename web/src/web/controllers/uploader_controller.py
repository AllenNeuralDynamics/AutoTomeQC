# web/controllers/uploader_controller.py
import asyncio
import json
import uuid
import imagesize
from nicegui import ui

from web.models.status import app_state
from web.services.api import analyze_image
from web.models.backend_schemas import PipelineResult
from web.models.status import QueuedFile

class UploaderController:
    # Update Init parameters
    def __init__(self, add_ui_callback, remove_ui_callback, set_active_ui_callback, refresh_workspace, refresh_inspector):
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
    

    async def handle_upload(self, e):
        """Processes files one-by-one to maintain simplicity and UI stability."""
        print("\n\nlen of files received:", len(e.files) if hasattr(e, 'files') else 'unknown'  )
        files = e.files
        if not files:
            return

        # 1. Start all tasks simultaneously
        # We process files in parallel, allowing the OS to handle concurrent writes
        tasks = [self.process_single_file(f) for f in files]
        
        # 2. Await all results
        # asyncio.gather returns a list of the returned file_ids
        results = await asyncio.gather(*tasks)
        print("Gathered results:")

        # 3. Filter out None results (duplicates or errors)
        added_ids = [file_id for file_id in results if file_id is not None]
        print("Added file IDs:", len(added_ids))

        # 2. Batch Update UI: Only call this once for the whole batch
        if added_ids:
            # Update the Renderer state and trigger the one-time UI sync
            self.add_ui(added_ids)
            ui.notify(f"Successfully processed {len(added_ids)} files", type='positive')
            
        e.sender.run_method('removeUploadedFiles')

    async def process_single_file(self, f):
        file_name = f.name
        
        # 1. DUPLICATE CHECK
        existing_names = {info.name for info in app_state.queued_files.values()}
        if file_name in existing_names:
            print(f"Duplicate found, skipping: {file_name}")
            return None
        
        # 2. READ BYTES: Use a more robust await pattern
        try:
            # Most SmallFileUpload objects have an async read method
            # We must explicitly await it.
            file_bytes = await f.read() 
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            return None

        file_id = uuid.uuid4().hex
        file_path = app_state.temp_upload_dir / file_name

        # 3. DISK I/O: Keep this in a thread to keep the event loop free
        def save_and_measure():
            with open(file_path, "wb") as f_handle:
                f_handle.write(file_bytes)
            w, h = imagesize.get(str(file_path))
            return w, h

        try:
            width, height = await asyncio.to_thread(save_and_measure)
        except Exception as ex:
            # FIX: Use ui.notify here. It IS thread-safe, 
            # but ensure it's not being called inside a deep non-GUI thread if possible.
            # Usually, ui.notify works fine from background tasks.
            print(f"Failed to save {file_name}: {ex}")
            return None

        # 4. UPDATE STATE (Keep this simple)
        app_state.queued_files[file_id] = QueuedFile(
            name=file_name, path=file_path, img_src=f"/temp_uploads/{file_name}",
            status='PENDING', width=width, height=height
        )
        return file_id

    async def handle_upload_(self, e):
        # THE BREATHER: Yield control to the event loop immediately 
        await asyncio.sleep(0.01)

        # Extract file object
        if hasattr(e, 'content'):
            file_obj = e.content
        elif hasattr(e, 'file'):
            file_obj = e.file
        elif hasattr(e, 'stream'):
            file_obj = e.stream
        else:
            return

        # Extract filename
        raw_name = getattr(e, 'name', None) or getattr(e, 'filename', None) or \
                   getattr(file_obj, 'name', None) or getattr(file_obj, 'filename', None)
        file_name = str(raw_name) if raw_name else f"image_{uuid.uuid4().hex[:6]}.jpg"

        # DUPLICATE CHECK
        # Check against existing names before we spend any time reading bytes or saving to disk.
        existing_names = {info.name for info in app_state.queued_files.values()}
        if file_name in existing_names:
            # Skip this file entirely
            return

        # READ BYTES
        if hasattr(file_obj, 'read'):
            read_result = file_obj.read()
            file_bytes = await read_result if asyncio.iscoroutine(read_result) else read_result
        else:
            file_bytes = file_obj

        file_id = uuid.uuid4().hex
        file_path = app_state.temp_upload_dir / file_name
        img_src = f"/temp_uploads/{file_name}"

        # DISK I/O THREAD
        def process_heavy_data():
            # Save to disk
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            # Measure image dimensions
            w, h = imagesize.get(str(file_path))
            return w, h
        try:
            # Run the heavy disk work in the background so the server doesn't freeze
            width, height = await asyncio.to_thread(process_heavy_data)
        except Exception as ex:
            ui.notify(f"Failed to save {file_name}: {ex}", type='negative')
            return

        # UPDATE STATE
        app_state.queued_files[file_id] = QueuedFile(
            name=file_name, path=file_path, img_src=img_src,
            status='PENDING', width=width, height=height
        )
        print("Length of queue after upload:", len(app_state.queued_files))

        # UPDATE UI
        self.add_ui(file_id)

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
    
        for file_id in list(app_state.queued_files.keys()):
            if not app_state.is_processing:
                break

            info = app_state.queued_files.get(file_id)
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
    