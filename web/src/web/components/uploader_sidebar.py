# web/components/uploader_sidebar.py
from nicegui import ui
import asyncio
from web.models.status import app_state

class QueueRenderer:
    def __init__(self):
        self.table = None
        self.on_click = None
        self.on_delete = None
        self.current_active_id = None
        self._update_task = None # Tracks background updates to prevent WebSocket flooding

    def mount(self, on_click, on_delete):
        """Mounts the Native Table container once when the sidebar is created."""
        self.on_click = on_click
        self.on_delete = on_delete

        columns = [
            {'name': 'img_src', 'label': 'Img', 'field': 'img_src', 'align': 'left', 'style': 'width: 15px; padding: 0 2px 0 0;'},
            {'name': 'name', 'label': 'File Name', 'field': 'name', 'align': 'left', 'sortable': True, 'classes': 'ellipsis', 'style': 'font-size: 0.75rem; padding: 0 2px;'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'align': 'left', 'sortable': True, 'style': 'width: 95px; min-width: 80px; max-width: 95px; padding: 0 2px; font-size: 0.75rem;'},
        ]

        # Initialize the native NiceGUI table
        self.table = ui.table(
            columns=columns,
            rows=app_state.grid_row_data,
            row_key='id',
            selection='multiple', # Enables checkboxes
            pagination=None,
        ).classes('w-full h-full custom-scrollbar')

        # Enable Virtual Scrolling for massive queues
        self.table.props('virtual-scroll :virtual-scroll-item-size="36" flat dense')

        # 1. Native Vue Slot for Images
        self.table.add_slot('body-cell-img_src', '''
            <q-td :props="props" style="padding: 1px 1px 1px 0;">
                <img v-if="props.row.img_src" :src="props.row.img_src" style="height: 32px; width: 32px; border-radius: 2px; display: block;" fit="cover" />
            </q-td>
        ''')

        # 2. Status Colors
        self.table.add_slot('body-cell-status', '''
            <q-td :props="props" style="padding: 1px 2px;">
                <span :style="{
                    color: props.row.status === 'PROCESSING' ? '#60a5fa' : 
                           props.row.status === 'PASS' ? '#4ade80' : 
                           (props.row.status === 'FAIL' || props.row.status === 'ERROR') ? '#f87171' : '#9ca3af'
                }">
                    {{ props.row.status }}
                </span>
            </q-td>
        ''')

        # Handle row clicks (Clicking the row text/image loads the file, clicking the checkbox selects it)
        self.table.on('rowClick', lambda e: self.on_click(e.args[1]['id']) if self.on_click else None)

    def add_item(self, file_id):
        """Appends the data to Python state and requests a batched UI update to prevent crashes."""
        info = app_state.queued_files.get(file_id)
        if info and self.table:
            row_data = {"id": file_id, **info.model_dump(mode='json')}
            self.table.rows.append(row_data)
            print("Length of table rows after add:", len(self.table.rows))
            if self._update_task is None or self._update_task.done():
                self._update_task = asyncio.create_task(self._delayed_update())

    async def _delayed_update(self):
        print("Batching UI update...")
        """Batches rows together and syncs the UI once every 1 second during heavy uploads."""
        await asyncio.sleep(0.5)
        if self.table:
            self.table.update()

    def remove_items(self, file_ids: list[str]):
        """Removes multiple items using NiceGUI's optimized remove_rows."""
        if not self.table:
            return
        
        # Identify which rows need to be removed based on IDs
        ids_to_remove = set(file_ids)
        rows_to_remove = [row for row in self.table.rows if row.get('id') in ids_to_remove]
        if not rows_to_remove:
            return

        # Use NiceGUI's native optimized method
        self.table.remove_rows(rows_to_remove)
        
        # Clean up the selection list (this is still required as NiceGUI 
        self.table.update()

    def set_active(self, active_file_id):
        self.current_active_id = active_file_id
        if self.table:
            for row in self.table.rows:
                info = app_state.queued_files.get(row['id'])
                if info and row['status'] != info.status:
                    row['status'] = info.status

            # This will check the box of the active file being processed
            self.table.selected = [{'id': active_file_id}]
            self.table.update()
            
            try:
                row_index = next(i for i, row in enumerate(self.table.rows) if row.get('id') == active_file_id)
                self.table.run_method('scrollTo', row_index, 'center')
            except StopIteration:
                pass

def render_uploader_sidebar(renderer: QueueRenderer, on_upload_callback, on_process_callback, on_item_click, on_item_delete):
    
    # NEW: Function to handle deleting specifically selected rows
    async def handle_delete_selected():
        selected_rows = renderer.table.selected
        if not selected_rows:
            ui.notify('No items selected to delete', type='warning')
            return
            
        with ui.dialog() as confirm_dialog, ui.card().classes('p-6'):
            ui.label(f'Delete {len(selected_rows)} selected item(s)?').classes('text-lg')
            ui.label('This action cannot be undone.')
            with ui.row().classes('w-full justify-end gap-4 pt-4'):
                ui.button('Cancel', on_click=confirm_dialog.close).props('flat')
                ui.button('Delete', on_click=lambda: confirm_dialog.submit('yes'), color='negative')
                
        result = await confirm_dialog
        if result == 'yes':
            if on_item_delete:
                removed_ids = [row['id'] for row in selected_rows]
                on_item_delete(removed_ids)
            # Clear selection after deletion
            renderer.table.selected = []
            renderer.table.update()

    with ui.left_drawer(fixed=True).classes('sidebar flex flex-col no-wrap') as left_drawer:
        
        # 1. Initialize Uploader completely hidden and OUTSIDE the flex row
        with ui.element('div').classes('hidden'):
            if on_upload_callback:
                uploader = ui.upload(on_upload=on_upload_callback, multiple=True, auto_upload=True) \
                    .props('accept="image/*" max-connections="2"')
                    
                def handle_batch_finish():
                    uploader.run_method('removeUploadedFiles') 
                    ui.notify("Finished uploading batch", type='positive')
                
                uploader.on('finish', handle_batch_finish)
            else:
                uploader = None

        # 2. Header Row
        with ui.row().classes('sidebar-header w-full items-center justify-between shrink-0 no-wrap'):
            
            # FIXED: Added 'shrink-0' so the title doesn't get crushed
            with ui.row().classes('sidebar-title items-center shrink-0'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')

                
            # FIXED: Added 'shrink-0 no-wrap' here. This absolutely forbids Flexbox from squishing your buttons together!
            with ui.row().classes('items-center justify-end gap-1 shrink-0 no-wrap'):
                
                if uploader:
                    upload_btn = ui.button(icon='upload', on_click=lambda: uploader.run_method('pickFiles')) \
                        .props('flat dense round') \
                        .tooltip('Upload images')
                    upload_btn.bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)
                
                if on_item_delete:
                    delete_btn = ui.button(icon='delete', on_click=handle_delete_selected) \
                        .props('flat dense round') \
                        .tooltip('Delete selected items')
                delete_btn.bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)

        with ui.element('div').classes('flex-grow w-full overflow-hidden min-h-0'):
            renderer.mount(on_item_click, on_item_delete)
        
        # 4. Footer Row
        with ui.row().classes('sidebar-footer w-full p-2 shrink-0'):
            if on_process_callback:
                btn = ui.button('', on_click=on_process_callback).classes('btn-process w-full')
                btn.bind_icon_from(app_state, 'is_processing', backward=lambda proc: 'pause' if proc else 'play_arrow')
                btn.bind_text_from(app_state, 'is_processing', backward=lambda proc: 'PAUSE BATCH' if proc else 'PROCESS BATCH')
        
    return left_drawer