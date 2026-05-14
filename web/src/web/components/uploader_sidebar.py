# web/components/uploader_sidebar.py
from nicegui import ui
from web.models.status import app_state

@ui.refreshable
def render_queue_list(on_item_click, on_item_delete):
    """Dynamically renders the queue list based on app_state."""
    if not app_state.queued_files:
        with ui.column().classes('queue-empty-state'):
            with ui.element('div').classes('queue-empty-icon-wrapper'):
                ui.icon('image', size='sm').classes('text-white')
            ui.label('NO DATA LOADED').classes('queue-empty-text')
        return

    for file_id, info in app_state.queued_files.items():
        _render_file_row(file_id, info, on_item_click, on_item_delete)
    
def _render_file_row(file_id, info, on_click_callback, on_delete_callback):
    """Renders a single file row using reactive bindings for high performance."""
    # Check if this specific row should be active when the list is rendered
    row_classes = 'queue-item shrink-0 active' if info.is_active else 'queue-item shrink-0'
    
    # Create the row AND assign it a unique HTML ID
    row = ui.row().classes(row_classes).props(f'id="row-{file_id}"')
    
    # Create the click handler (MUST accept 'e' to prevent Python errors)
    def handle_click(e):
        # Instantly update UI via Javascript
        # Use java script to avoid rendering again when user clickes on the active item. 
        # Because JavaScript runs directly in the user's browser, 
        # allowing instantly update visual elements without waiting for a server round-trip.
        ui.run_javascript(f'''
            document.querySelectorAll('.queue-item').forEach(el => el.classList.remove('active'));
            const targetRow = document.getElementById('row-{file_id}');
            if (targetRow) {{
                targetRow.classList.add('active');
            }} else {{
                console.warn('Could not find row with ID: row-{file_id}');
            }}
        ''')
        # Trigger the heavy Python logic in the background
        on_click_callback(file_id)
        
    # Bind our new handler to the row
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

def render_uploader_sidebar(on_upload_callback, on_process_callback, on_item_click, on_item_delete):
    """Renders the static sidebar wrapper."""
    with ui.left_drawer(fixed=True).classes('sidebar') as left_drawer:
        
        with ui.row().classes('sidebar-header'):
            with ui.row().classes('sidebar-title'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')
            
            upload_btn = ui.button(icon='upload', color=None).classes('btn-upload')
        
        with ui.column().classes('queue-container custom-scrollbar gap-0'):
            # Mount the refreshable component here
            render_queue_list(on_item_click, on_item_delete)
                
        if on_upload_callback:
            uploader = ui.upload(on_upload=on_upload_callback, multiple=True, auto_upload=True).props('accept="image/*"').classes('hidden-uploader')
        upload_btn.on('click', lambda: uploader.run_method('pickFiles'))
        
        with ui.row().classes('sidebar-footer'):
            if on_process_callback:
                ui.button('PROCESS BATCH', icon='play_arrow', color=None, on_click=on_process_callback).classes('btn-process')
            
    return left_drawer