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
            {'name': 'name', 'label': 'File Name', 'field': 'name', 'align': 'left', 'classes': 'ellipsis', 'style': 'font-size: 0.75rem; padding: 0 2px;'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'align': 'left', 'style': 'width: 95px; min-width: 80px; max-width: 95px; padding: 0 2px; font-size: 0.75rem;'},
            {'name': 'actions', 'label': '', 'field': 'id', 'align': 'right', 'style': 'width: 30px; padding: 0;'}
        ]

        # Initialize the native NiceGUI table
        self.table = ui.table(
            columns=columns,
            rows=app_state.grid_row_data,
            row_key='id',
            selection='single',
            pagination=None,
        ).classes('w-full h-full custom-scrollbar')

        # Enable Virtual Scrolling for massive queues
        self.table.props('virtual-scroll :virtual-scroll-item-size="36" flat dense')

        # Clear the selection slots so the checkboxes disappear
        self.table.add_slot('header-selection', '')
        self.table.add_slot('body-selection', '')

        # 1. Native Vue Slot for Images
        self.table.add_slot('body-cell-img_src', '''
            <q-td :props="props" style="padding: 1px 1px 1px 0;">
                <img v-if="props.row.img_src" :src="props.row.img_src" style="height: 32px; width: 32px; border-radius: 2px; display: block;" fit="cover" />
            </q-td>
        ''')

        # 2. Native Vue Slot for Status Colors
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

        self.table.add_slot('body-cell-actions', '''
            <q-td :props="props" style="padding: 2px 2px;">
                <q-btn flat round dense color="red" icon="delete" size="sm" @click.stop="$parent.$emit('delete_row', props.row.id)" />
            </q-td>
        ''')

        # Handle row clicks
        self.table.on('rowClick', lambda e: self.on_click(e.args[1]['id']) if self.on_click else None)
        
        # Handle the custom delete button click
        self.table.on('delete_row', lambda e: self.on_delete(e.args) if self.on_delete else None)

    def add_item(self, file_id):
        """Appends the data to Python state and requests a batched UI update to prevent crashes."""
        print("Add item called for file_id:", file_id, len(self.table.rows) if self.table else 'No table')
        info = app_state.queued_files.get(file_id)
        if info and self.table:
            row_data = {"id": file_id, **info.model_dump(mode='json')}
            self.table.rows.append(row_data)
            
            if self._update_task is None or self._update_task.done():
                self._update_task = asyncio.create_task(self._delayed_update())

    async def _delayed_update(self):
        """Batches rows together and syncs the UI once every 1 second during heavy uploads."""
        await asyncio.sleep(1.0)
        if self.table:
            self.table.update()

    def remove_item(self, file_id):
        print("Remove item called for file_id:", file_id)
        if self.table:
            self.table.rows = [row for row in self.table.rows if row.get('id') != file_id]
            self.table.update()

    def set_active(self, active_file_id):
        print("Active file ID set to:", active_file_id)
        self.current_active_id = active_file_id
        
        if self.table:
            for row in self.table.rows:
                info = app_state.queued_files.get(row['id'])
                if info and row['status'] != info.status:
                    row['status'] = info.status

            self.table.selected = [{'id': active_file_id}]
            self.table.update()
            
            try:
                row_index = next(i for i, row in enumerate(self.table.rows) if row.get('id') == active_file_id)
                self.table.run_method('scrollTo', row_index, 'center')
            except StopIteration:
                pass


async def _show_delete_all_dialog(on_delete_all_callback=None):
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
    # FIXED: Added 'no-wrap' so elements strictly stay in their zones
    with ui.left_drawer(fixed=True).classes('sidebar flex flex-col no-wrap') as left_drawer:
        
        # FIXED: Added 'shrink-0' so the header never squishes
        with ui.row().classes('sidebar-header w-full items-center justify-between shrink-0'):
            with ui.row().classes('sidebar-title items-center'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')
                
            with ui.row().classes('items-center gap-1'):
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

        # FIXED: Added 'min-h-0'. This is the Flexbox magic that prevents the table from pushing the footer off the screen!
        with ui.element('div').classes('flex-grow w-full overflow-hidden min-h-0'):
            renderer.mount(on_item_click, on_item_delete)
        
        if on_upload_callback:
            uploader = ui.upload(on_upload=on_upload_callback,
                                 multiple=True,
                                 auto_upload=True) \
            .props('accept="image/*" max-connections="2"').classes('hidden-uploader')

            def handle_batch_finish():
                uploader.run_method('removeUploadedFiles') 
                ui.notify(f"Finished uploading batch", type='positive')

            uploader.on('finish', handle_batch_finish)
            
        upload_btn.on('click', lambda: uploader.run_method('pickFiles'))
        
        # FIXED: Added 'shrink-0' to keep the footer permanently pinned to the bottom
        with ui.row().classes('sidebar-footer w-full p-2 shrink-0'):
            if on_process_callback:
                btn = ui.button('', on_click=on_process_callback).classes('btn-process w-full')
                btn.bind_icon_from(app_state, 'is_processing', backward=lambda proc: 'pause' if proc else 'play_arrow')
                btn.bind_text_from(app_state, 'is_processing', backward=lambda proc: 'PAUSE BATCH' if proc else 'PROCESS BATCH')
        
    return left_drawer