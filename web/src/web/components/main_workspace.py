# web/components/main_workspace.py
from nicegui import ui
from typing import Union
from pathlib import Path
from PIL import Image
from web.models.status import app_state

@ui.refreshable
def render_main_workspace():
    # Standard container for centering everything
    image_container = ui.element('div').classes(
        'w-[640px] h-[640px] flex-none '
        'flex items-center justify-center '
        'bg-black overflow-hidden m-auto '
        'self-center rounded-lg border border-[#222222]'
    )
    
    status = getattr(app_state, 'view_status', 'idle')

    with image_container:
        # --- IDLE: No image selected ---
        if status == 'idle':
            with ui.column().classes('items-center justify-center text-gray-500'):
                ui.icon('aspect_ratio', size='6rem')
                ui.label('VIEWPORT_IDLE')

        # --- PENDING: Show raw image for manual inspection (no spinner) ---
        elif status == 'pending':
            img_src = next((f.img_src for f in app_state.queued_files.values() if f.is_active), None)
            if img_src:
                with ui.element('div').classes('image-wrapper'):
                    ui.image(img_src).classes('image-preview').style('width: 640px; height: 640px; object-fit: contain;')
            else:
                ui.spinner('dots', size='lg')

        # --- PROCESSING: Show dimmed image with "Analyzing" overlay ---
        elif status == 'processing':
            active_info = next((f for f in app_state.queued_files.values() if f.is_active), None)
            
            with ui.element('div').classes('relative w-[640px] h-[640px] flex items-center justify-center'):
                if active_info and active_info.img_src:
                    # Dimmed background image
                    ui.image(active_info.img_src).style('width: 640px; height: 640px; object-fit: contain; opacity: 0.4;')
                    
                    # Active overlay
                    with ui.column().classes('absolute inset-0 flex items-center justify-center'):
                        ui.spinner('dots', size='lg', color='orange')
                        ui.label('ANALYZING...').classes('text-orange font-bold mt-2 tracking-widest')
                else:
                    ui.spinner('dots', size='lg', color='orange')

        # --- RESULT: Show image with interactive SVG masks ---
        elif status == 'result':
            result = getattr(app_state, 'view_result', None)
            
            # Find the active image path
            img_src = None
            for info in app_state.queued_files.values():
                if info.is_active:
                    img_src = info.path
                    break
            
            if img_src:
                if isinstance(img_src, Path) and img_src.exists():
                    img_src = Image.open(img_src).resize((640, 640))
                elif isinstance(img_src, str) and Path(img_src).exists():
                    img_src = Image.open(img_src).resize((640, 640))

                with ui.element('div').classes('relative'):
                    image_view = ui.interactive_image(img_src).style('width: 640px; height: 640px;')
                    
                    if result and result.sections:
                        svg_content = ""
                        for sec in result.sections:
                            if sec.mask:
                                points_str = " ".join([f"{p[0]},{p[1]}" for p in sec.mask])
                                svg_content += f'<polygon points="{points_str}" fill="rgba(242, 125, 38, 0.2)" stroke="#F27D26" stroke-width="2" />'
                        
                        if svg_content:
                            image_view.content = svg_content
                            
                            def toggle_mask():
                                if image_view.content:
                                    image_view.content = ""
                                    toggle_btn._props['icon'] = 'visibility_off'
                                else:
                                    image_view.content = svg_content
                                    toggle_btn._props['icon'] = 'visibility'
                                toggle_btn.update()
                                
                            toggle_btn = ui.button(icon='visibility', on_click=toggle_mask) \
                                .props('flat round color=white') \
                                .classes('absolute top-2 right-2 bg-black/50 z-10')

        # --- ERROR: Show failure message ---
        elif status == 'error':
            msg = getattr(app_state, 'view_error', 'Backend Error')
            ui.label(msg).classes('text-red-600 font-bold')
