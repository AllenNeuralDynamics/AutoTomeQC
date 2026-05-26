
from nicegui import ui
import logging
from autotome_ui.models.status import app_state


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
            {'name': 'name', 'label': 'File Name', 'field': 'name', 'align': 'left', 'classes': 'ellipsis', 'style': 'font-size: 0.75rem; padding: 0 2px;'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'align': 'left', 'classes': 'ellipsis', 'style': 'width: 95px; min-width: 80px; max-width: 95px; padding: 0 2px; font-size: 0.75rem;'},
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
