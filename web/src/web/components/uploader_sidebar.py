# web/components/uploader_sidebar.py
from nicegui import ui
import asyncio
import logging
from web.models.status import app_state


class QueueRenderer:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
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

    def add_item(self, file_ids: list[str]):
        """Appends the data to Python state and requests a batched UI update to prevent crashes."""
        if not self.table:
            return

        self.log.debug("Len of file_ids to add: %d", len(file_ids))

        new_rows = []
        for file_id in file_ids:
            info = app_state.queued_files.get(file_id)
            if info:
                row_data = {"id": file_id, **info.model_dump(mode='json')}
                new_rows.append(row_data)
                
        if new_rows:
            self.table.add_rows(new_rows)
    
    def add_item_(self, file_id):
        """Appends the data to Python state and requests a batched UI update to prevent crashes."""
        info = app_state.queued_files.get(file_id)
        if info and self.table:
            row_data = {"id": file_id, **info.model_dump(mode='json')}
            self.table.rows.append(row_data)
            self.log.debug("Length of table rows after add: %d", len(self.table.rows))
            if self._update_task is None or self._update_task.done():
                self._update_task = asyncio.create_task(self._delayed_update())

    async def _delayed_update(self):
        self.log.debug("Batching UI update...")
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
        self.log.debug("Removed %d rows", len(rows_to_remove))
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



def render_uploader_sidebar(renderer, on_upload_callback, on_process_callback, on_item_click, on_item_delete):
    
    async def handle_delete_selected():
        selected_rows = renderer.table.selected
        if not selected_rows:
            ui.notify('No items selected', type='warning')
            return
            
        with ui.dialog() as confirm_dialog, ui.card().classes('p-6'):
            ui.label(f'Delete {len(selected_rows)} items?').classes('text-lg')
            with ui.row().classes('w-full justify-end gap-4 pt-4'):
                ui.button('Cancel', on_click=confirm_dialog.close).props('flat')
                ui.button('Delete', on_click=lambda: confirm_dialog.submit('yes'), color='negative')
                
        if await confirm_dialog == 'yes':
            removed_ids = [row['id'] for row in selected_rows]
            if on_item_delete:
                on_item_delete(removed_ids)
            # Explicit context for UI updates
            with renderer.table:
                renderer.table.selected = []
                renderer.table.update()

    with ui.left_drawer(fixed=True).classes('sidebar flex flex-col no-wrap') as left_drawer:
        
        # 1. Initialize Uploader (Hidden via CSS)
        uploader = None
        def _handle_rejection():
            ui.notify("Limit reached: Only 1000 files allowed per upload.", type='warning')
            # Reset the uploader so it can be used again
            uploader.run_method('removeUploadedFiles')
            uploader.update()

        if on_upload_callback:
            with ui.element('div').style('position: absolute; left: -9999px;'):
                uploader = ui.upload(
                    on_multi_upload=on_upload_callback,
                    on_rejected=_handle_rejection,
                    multiple=True,
                    max_files=1000,
                    auto_upload=True
                ).props('accept="image/*" max-connections="1" batch="true" batch-size="50"')

        # 2. Header Row
        with ui.row().classes('sidebar-header w-full items-center justify-between shrink-0 no-wrap'):
            with ui.row().classes('sidebar-title items-center shrink-0'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')

            with ui.row().classes('items-center justify-end gap-1 shrink-0 no-wrap'):
                if uploader:
                    ui.button(icon='upload', on_click=lambda: uploader.run_method('pickFiles')) \
                        .props('flat dense round') \
                        .tooltip('Upload images') \
                        .bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)
                
                if on_item_delete:
                    ui.button(icon='delete', on_click=handle_delete_selected) \
                        .props('flat dense round') \
                        .tooltip('Delete selected items') \
                        .bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)

        # 3. Main Content Area
        with ui.element('div').classes('flex-grow w-full overflow-hidden min-h-0'):
            renderer.mount(on_item_click, on_item_delete)
        
        # 4. Footer Row
        if on_process_callback:
            with ui.row().classes('sidebar-footer w-full p-2 shrink-0'):
                btn = ui.button('', on_click=on_process_callback).classes('btn-process w-full')
                btn.bind_icon_from(app_state, 'is_processing', backward=lambda p: 'pause' if p else 'play_arrow')
                btn.bind_text_from(app_state, 'is_processing', backward=lambda p: 'PAUSE BATCH' if p else 'PROCESS BATCH')
        
    return left_drawer