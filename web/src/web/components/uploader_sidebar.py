from nicegui import ui
from web.models.status import app_state

# 1. Create a QueueRenderer to handle O(1) appends and deletes
class QueueRenderer:
    def __init__(self):
        self.container = None
        self.empty_state = None
        self.rendered_rows = {}
        self.on_click = None
        self.on_delete = None

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
            info = app_state.queued_files[file_id]
            with self.container:
                row = _render_file_row(file_id, info, self.on_click, self.on_delete)
                self.rendered_rows[file_id] = row

    def remove_item(self, file_id):
        """O(1) deletion of a single item."""
        if file_id in self.rendered_rows and self.container:
            self.container.remove(self.rendered_rows[file_id])
            del self.rendered_rows[file_id]
            
        if not app_state.queued_files and self.empty_state:
            self.empty_state.set_visibility(True)

# Expose a global instance of our renderer
queue_renderer = QueueRenderer()

def _render_file_row(file_id, info, on_click_callback, on_delete_callback):
    """Renders a single file row using reactive bindings for high performance."""
    is_initially_active = (app_state.active_file_id == file_id)
    row_classes = 'queue-item shrink-0 active' if is_initially_active else 'queue-item shrink-0'
    
    row = ui.row().classes(row_classes).props(f'id="row-{file_id}"')
    print("[debug] Rendering row for file_id:", file_id, "with status:", info.status)

    def handle_click(e):
        if app_state.is_processing:
            ui.notify("Cannot select images while processing batch", type='warning')
            return
            
        prev_active_id = app_state.active_file_id
        
        # If there is a currently active row, remove its 'active' class
        if prev_active_id and prev_active_id in queue_renderer.rendered_rows:
            queue_renderer.rendered_rows[prev_active_id].classes(remove='active')
            
        # Add the 'active' class to the newly clicked row
        row.classes(add='active')

        # Trigger the Python logic in the background
        on_click_callback(file_id)

    # Bind the click event to the entire row, but prevent it from triggering when clicking the delete button
    row.on('click', handle_click)

    with row:
        with ui.element('div').classes('queue-thumb'):
            ui.image(info.img_src).classes('queue-img')

        with ui.element('div').classes('queue-details'):
            ui.label(info.name).classes('queue-filename')
            
            with ui.row().classes('queue-status-row'):
                # 2. Bind the spinner visibility to the 'PROCESSING' status
                spinner = ui.spinner('dots', size='1em', color='blue-400')
                spinner.bind_visibility_from(info, 'status', backward=lambda s: s == 'PROCESSING')
                
                # 3. Create dedicated status labels and bind their visibility to their respective states.
                # This is much faster and cleaner than trying to dynamically rewrite CSS variables on the fly.
                lbl_proc = ui.label('PROCESSING').classes('queue-status-text').style('color: #60a5fa !important')
                lbl_proc.bind_visibility_from(info, 'status', backward=lambda s: s == 'PROCESSING')
                
                lbl_pass = ui.label('PASS').classes('queue-status-text').style('color: var(--pass-color) !important')
                lbl_pass.bind_visibility_from(info, 'status', backward=lambda s: s == 'PASS')
                
                lbl_fail = ui.label('FAIL').classes('queue-status-text').style('color: var(--fail-color) !important')
                lbl_fail.bind_visibility_from(info, 'status', backward=lambda s: s == 'FAIL')

                lbl_err = ui.label('ERROR').classes('queue-status-text').style('color: var(--fail-color) !important')
                lbl_err.bind_visibility_from(info, 'status', backward=lambda s: s == 'ERROR')

                lbl_pend = ui.label('PENDING').classes('queue-status-text')
                lbl_pend.bind_visibility_from(info, 'status', backward=lambda s: s == 'PENDING')

        # 4. Bind the delete button visibility inversely to the global 'is_processing' state
        del_btn = ui.button(icon='delete', color='red') \
            .props('flat dense') \
            .classes('btn-delete') \
            .on('click.stop', lambda e, fid=file_id: on_delete_callback(fid))
            
        del_btn.bind_visibility_from(app_state, 'is_processing', backward=lambda is_proc: not is_proc)

    return row

def render_uploader_sidebar(on_upload_callback, on_process_callback, on_item_click, on_item_delete):
    """Renders the static sidebar wrapper."""
    with ui.left_drawer(fixed=True).classes('sidebar') as left_drawer:
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')
            upload_btn = ui.button(icon='upload', color=None).classes('btn-upload')
        
        # 2. Mount the high-performance structural renderer
        queue_renderer.mount(on_item_click, on_item_delete)
        
        if on_upload_callback:
            uploader = ui.upload(on_upload=on_upload_callback, multiple=True, auto_upload=True).props('accept="image/*"').classes('hidden-uploader')
        upload_btn.on('click', lambda: uploader.run_method('pickFiles'))
        
        with ui.row().classes('sidebar-footer'):
            if on_process_callback:
                btn = ui.button('', on_click=on_process_callback).classes('btn-process')
                btn.bind_icon_from(app_state, 'is_processing', backward=lambda proc: 'pause' if proc else 'play_arrow')
                btn.bind_text_from(app_state, 'is_processing', backward=lambda proc: 'PAUSE BATCH' if proc else 'PROCESS BATCH')
        
    return left_drawer