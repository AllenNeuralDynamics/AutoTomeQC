# web/components/uploader_sidebar.py
from nicegui import ui
from autotome_ui.models.status import app_state
from nicegui import app

def render_uploader_sidebar(renderer, on_upload_callback, on_process_callback, on_item_click, on_item_delete):
    
    async def _handle_delete_selected():
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
            with renderer.table:
                renderer.table.selected = []
                renderer.table.update()

    # --- Native File Picker Helper ---
    async def _pick_files():
        files = await app.native.main_window.create_file_dialog(
            allow_multiple=True,
            file_types=('Image files (*.jpg;*.png;*.jpeg;*.tiff;*.bmp;*.gif)', 'All files (*.*)')
        )
        if files:
            await on_upload_callback(list(files))

    # --- Web Uploader Rejection Handler ---
    def _handle_rejection():
        ui.notify("Limit reached: Only 1000 files allowed per upload.", type='warning')

    with ui.left_drawer(fixed=True).classes('sidebar flex flex-col no-wrap') as left_drawer:
        
        # 2. Header Row
        with ui.row().classes('sidebar-header w-full items-center justify-between shrink-0 no-wrap'):
            with ui.row().classes('sidebar-title items-center shrink-0'):
                ui.icon('monitor_heart').classes('text-accent text-xl')
                ui.label('AutoTome-QC').classes('sidebar-title-text')

            with ui.row().classes('items-center justify-end gap-1 shrink-0 no-wrap'):
                
                # Dynamic Uploader: Native vs Web
                if app_state.is_native:
                    ui.button(icon='upload', on_click=_pick_files) \
                        .props('flat dense round') \
                        .tooltip('Select images') \
                        .bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)
                else:
                    # Hidden uploader triggered by a button
                    with ui.element('div').style('position: absolute; left: -9999px;'):
                        uploader = ui.upload(
                            on_multi_upload=on_upload_callback,
                            on_rejected=_handle_rejection,
                            multiple=True,
                            max_files=1000,
                            auto_upload=True
                        ).props('accept="image/*" max-connections="1" batch="true" batch-size="50"')
                    
                    ui.button(icon='upload', on_click=lambda: uploader.run_method('pickFiles')) \
                        .props('flat dense round') \
                        .tooltip('Upload images') \
                        .bind_visibility_from(app_state, 'is_processing', backward=lambda p: not p)
                
                # Delete Button
                if on_item_delete:
                    ui.button(icon='delete', on_click=_handle_delete_selected) \
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