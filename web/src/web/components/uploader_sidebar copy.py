# web/components/uploader_sidebar.py
from nicegui import ui
import asyncio
from web.models.status import app_state

class QueueRenderer:
    def __init__(self):
        self.container = None
        self.empty_state = None
        self.rendered_rows = {}
        self.on_click = None
        self.on_delete = None
        self.current_active_id = None

        self.render_lock = asyncio.Lock()

    def mount(self, on_click, on_delete):
        """Mounts the persistent container once when the sidebar is created."""
        self.on_click = on_click
        self.on_delete = on_delete
        self.container = ui.column().classes('queue-container custom-scrollbar gap-0')
        
        with self.container:
            self.empty_state = ui.column().classes('queue-empty-state')
            with self.empty_state:
                with ui.element('div').classes('queue-empty-icon-wrapper'):
                    ui.icon('image', size='sm').classes('text-white')
                ui.label('NO DATA LOADED').classes('queue-empty-text')

            
            # Render initial files if resuming a session
            if app_state.queued_files:
                self.empty_state.set_visibility(False)
                for fid in app_state.queued_files:
                    self.add_item(fid)
            
        

    def add_item(self, file_id):
        """O(1) appending of a single item."""
        if self.empty_state:
            self.empty_state.set_visibility(False)
            
        if self.container:
            info = app_state.queued_files.get(file_id)
            if info:
                with self.container:
                    # Call the internal method
                    row = self._render_file_row(file_id, info)
                    self.rendered_rows[file_id] = row
        
        print("Length of rendered_rows after add:", len(self.rendered_rows))
    
    def remove_item(self, file_id):
        """O(1) deletion of a single item."""
        if file_id in self.rendered_rows and self.container:
            self.container.remove(self.rendered_rows[file_id])
            del self.rendered_rows[file_id]
            
        if not app_state.queued_files and self.empty_state:
            self.empty_state.set_visibility(True)
        
    def set_active(self, active_file_id):
        """Globally updates which row has the 'active' class."""
        # 1. SOLVE TODO: Remove 'active' from the PREVIOUS row only (O(1) time)
        if self.current_active_id and self.current_active_id in self.rendered_rows:
            self.rendered_rows[self.current_active_id].classes(remove='active')
            
        # Update our tracker
        self.current_active_id = active_file_id

        # 2. Add 'active' to the NEW row and update its status
        if active_file_id in self.rendered_rows:
            active_row = self.rendered_rows[active_file_id]
            active_row.classes(add='active')
            
            # Fetch the latest info
            info = app_state.queued_files.get(active_file_id)
            if not info:
                return

            # 3. UPDATE THE STATUS LABEL DIRECTLY
            active_row.status_label.set_text(info.status)
            
            # Update the color based on the status
            if info.status == 'PROCESSING':
                active_row.status_label.style('color: #60a5fa !important')
            elif info.status == 'PASS':
                active_row.status_label.style('color: var(--pass-color) !important')
            elif info.status in ['FAIL', 'ERROR']:
                active_row.status_label.style('color: var(--fail-color) !important')
            else:
                # Reset to default for PENDING
                active_row.status_label.classes('queue-status-text')
            

    def _render_file_row(self, file_id, info):
        """Renders a single file row using reactive bindings for high performance."""
        is_initially_active = (app_state.active_file_id == file_id)
        row_classes = 'queue-item shrink-0 active' if is_initially_active else 'queue-item shrink-0'
        
        row = ui.row().classes(row_classes).props(f'id="row-{file_id}"')

        def handle_click(e):
            if app_state.is_processing:
                ui.notify("Cannot select images while processing batch", type='warning')
                return
                
            # Trigger the Python logic using the stored callback
            if self.on_click:
                self.on_click(file_id)

        row.on('click', handle_click)

        with row:
            with ui.element('div').classes('queue-thumb'):
                ui.image(info.img_src).classes('queue-img')
            with ui.element('div').classes('queue-details'):
                ui.label(info.name).classes('queue-filename')
                with ui.row().classes('queue-status-row'):
                    row.status_label = ui.label('PENDING').classes('queue-status-text')
            del_btn = ui.button(icon='delete', color='red') \
                .props('flat dense') \
                .classes('btn-delete') \
                .on('click.stop', lambda e, fid=file_id: self.on_delete(fid) if self.on_delete else None)
        return row

async def _show_delete_all_dialog(on_delete_all_callback=None):
    """Shows a confirmation dialog before deleting all items."""
    with ui.dialog() as confirm_dialog, ui.card().classes('p-6'):
        ui.label('Are you sure you want to delete all items in the queue?').classes('text-lg')
        ui.label('This action cannot be undone.')
        with ui.row().classes('w-full justify-end gap-4 pt-4'):
            ui.button('Cancel', on_click=confirm_dialog.close).props('flat')
            ui.button('Delete All', on_click=lambda: confirm_dialog.submit('yes'), color='negative')
    result = await confirm_dialog
    if result == 'yes' and on_delete_all_callback:
        on_delete_all_callback()

def render_uploader_sidebar(renderer: QueueRenderer, on_upload_callback, on_process_callback, on_item_click, on_item_delete, on_delete_all_callback=None):
    """Renders the static sidebar wrapper."""
    with ui.left_drawer(fixed=True).classes('sidebar') as left_drawer:
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')
            with ui.row().classes('items-center'):
                upload_btn = ui.button(icon='upload', color=None).classes('btn-upload') \
                        .props('flat dense round') \
                        .tooltip('Upload images') \
                        .classes('text-gray-400 hover:text-white')
                upload_btn.bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)
                delete_all_btn = ui.button(icon='delete_sweep', color=None, on_click=lambda: _show_delete_all_dialog(on_delete_all_callback)) \
                    .props('flat dense round') \
                    .tooltip('Delete all items') \
                    .classes('btn-delete-all text-gray-400 hover:text-white')
                delete_all_btn.bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)

        # Mount using the passed-in renderer
        renderer.mount(on_item_click, on_item_delete)
        
        if on_upload_callback:
            uploader = ui.upload(on_upload=on_upload_callback,
                                 multiple=True,
                                 auto_upload=True) \
            .props('accept="image/*" max-connections="5"').classes('hidden-uploader')

            def handle_batch_finish():
                uploader.run_method('removeUploadedFiles') # Clean up the UI
                ui.notify(f"Finished uploading batch", type='positive')

            uploader.on('finish', handle_batch_finish)
        upload_btn.on('click', lambda: uploader.run_method('pickFiles'))
        
        with ui.row().classes('sidebar-footer'):
            if on_process_callback:
                btn = ui.button('', on_click=on_process_callback).classes('btn-process')
                btn.bind_icon_from(app_state, 'is_processing', backward=lambda proc: 'pause' if proc else 'play_arrow')
                btn.bind_text_from(app_state, 'is_processing', backward=lambda proc: 'PAUSE BATCH' if proc else 'PROCESS BATCH')
        
    return left_drawer