from nicegui import ui
import asyncio
from web.models.status import app_state

class QueueRenderer:
    def __init__(self):
        self.grid = None
        self.on_click = None
        self.on_delete = None
        self.current_active_id = None

    def mount(self, on_click, on_delete):
        self.on_click = on_click
        self.on_delete = on_delete
        # 1. Absolute simplest AG grid config
        cell_renderer_js = "(params) => params.value ? '<img src=\"' + params.value + '\" style=\"height: 30px; width: 30px; object-fit: cover; border-radius: 4px; margin-top: 5px;\">' : ''"
        cell_style_js = "(params) => { if (params.value === 'PROCESSING') return {color: '#60a5fa', fontWeight: 'bold'}; if (params.value === 'PASS') return {color: '#4ade80', fontWeight: 'bold'}; if (params.value === 'FAIL' || params.value === 'ERROR') return {color: '#f87171', fontWeight: 'bold'}; return {color: '#9ca3af'}; }"

        options = {
            'columnDefs': [
                {'field': 'img_src', 'width': 65, 'headerName': 'Img', 'cellRenderer': cell_renderer_js},
                {'field': 'name', 'headerName': 'File Name'},
                {'field': 'status', 'headerName': 'Status', 'cellStyle': cell_style_js}
            ],
            'rowData': app_state.grid_row_data,
            'rowSelection': {'mode': 'multiRow'},
        }

        # Pass theme directly as an argument inside the parentheses
        # Initialize the grid
        self.grid = ui.aggrid(
            options, 
            html_columns=[0], # Tell NiceGUI it is safe to inject the <img> tag
            theme="balham"
        ).classes('w-full h-full min-h-[300px]')

    def add_item(self, file_id):
        """O(1) appending of a single item using grid transactions."""
        info = app_state.queued_files.get(file_id)
        if info and self.grid:
            print("self.grid:", self.grid, file_id)
            row_data = {"id": file_id, **info.model_dump(mode='json')}
            print("Adding row to grid:", row_data)
            # applyTransaction adds row client-side without sending the whole array over websocket
            self.grid.run_grid_method('applyTransaction', {'add': [row_data]})

    def remove_item(self, file_id):
        """O(1) deletion of a single item."""
        if self.grid:
            self.grid.run_grid_method('applyTransaction', {'remove': [{'id': file_id}]})

    def set_active(self, active_file_id):
        """Updates status and selects the active row visually."""
        self.current_active_id = active_file_id
        info = app_state.queued_files.get(active_file_id)
        
        if info and self.grid:
            row_data = {"id": active_file_id, **info.model_dump(mode='json')}
            # Update cell values dynamically
            self.grid.run_grid_method('applyTransaction', {'update': [row_data]})
            
            # Visually select the row in the grid
            self.grid.run_grid_method('forEachNode', f'''(node) => {{
                if (node.data.id === "{active_file_id}") {{
                    node.setSelected(true);
                }}
            }}''')

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
    with ui.left_drawer(fixed=True).classes('sidebar flex flex-col') as left_drawer:
        with ui.row().classes('sidebar-header w-full items-center justify-between'):
            with ui.row().classes('sidebar-title items-center'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')
                
            with ui.row().classes('items-center gap-1'):
                upload_btn = ui.button(icon='upload', color=None).classes('btn-upload') \
                        .props('flat dense round') \
                        .tooltip('Upload images') \
                        .classes('text-gray-400 hover:text-white')
                upload_btn.bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)
                
                # NEW: Delete Selected Button (Since we removed the row-level delete icon)
                async def handle_delete_selected():
                    selected = await renderer.grid.get_selected_rows()
                    if selected and len(selected) > 0 and on_item_delete:
                        on_item_delete(selected[0]['id'])
                        
                delete_selected_btn = ui.button(icon='remove_circle', color=None, on_click=handle_delete_selected) \
                    .props('flat dense round') \
                    .tooltip('Delete selected item') \
                    .classes('text-gray-400 hover:text-white')
                delete_selected_btn.bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)

                delete_all_btn = ui.button(icon='delete_sweep', color=None, on_click=lambda: _show_delete_all_dialog(on_delete_all_callback)) \
                    .props('flat dense round') \
                    .tooltip('Delete all items') \
                    .classes('btn-delete-all text-gray-400 hover:text-white')
                delete_all_btn.bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)

        # GRID CONTAINER: Flex grow ensures ag-grid takes up the exact remaining height
        with ui.element('div').classes('flex-grow w-full overflow-hidden'):
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
        
        with ui.row().classes('sidebar-footer w-full p-2'):
            if on_process_callback:
                btn = ui.button('', on_click=on_process_callback).classes('btn-process w-full')
                btn.bind_icon_from(app_state, 'is_processing', backward=lambda proc: 'pause' if proc else 'play_arrow')
                btn.bind_text_from(app_state, 'is_processing', backward=lambda proc: 'PAUSE BATCH' if proc else 'PROCESS BATCH')
        
    return left_drawer