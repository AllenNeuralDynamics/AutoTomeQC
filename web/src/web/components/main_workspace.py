# web/components/main_workspace.py
from nicegui import ui
from pathlib import Path
from PIL import Image
from web.models.status import app_state

@ui.refreshable
def render_main_workspace(on_next_callback=None, on_prev_callback=None):
    # Standard container for centering everything
    image_container = ui.element('div').classes(
        'w-full h-full max-w-[1200px] max-h-[1200px] aspect-square flex-none '
        'flex items-center justify-center '
        'bg-black overflow-hidden m-auto '
        'self-center rounded-lg border border-[#222222]'
    )
    # TODO display image size
    status = getattr(app_state.view, 'status', 'idle')
 
    with image_container:
        # --- IDLE: No image selected ---
        if status == 'idle':
            with ui.column().classes('items-center justify-center text-gray-500'):
                ui.icon('aspect_ratio', size='6rem')
                ui.label('VIEWPORT IDLE')

        # --- PENDING: Show raw image for manual inspection (no spinner) ---
        elif status == 'pending':
            active_file = app_state.queued_files.get(app_state.active_file_id)
            if active_file and active_file.img_src:
                with ui.element('div').classes('image-wrapper'):
                    ui.image(active_file.img_src).classes('image-preview').style('width: 640px; height: 640px; object-fit: contain;')
            else:
                ui.spinner('dots', size='lg')

        # --- PROCESSING: Show dimmed image with "Analyzing" overlay ---
        elif status == 'processing':
            active_file = app_state.queued_files.get(app_state.active_file_id)
            with ui.element('div').classes('relative w-[640px] h-[640px] flex items-center justify-center'):
                if active_file and active_file.img_src:
                    # Dimmed background image
                    ui.image(active_file.img_src).style('width: 640px; height: 640px; object-fit: contain; opacity: 0.4;')
                    
                    # Active overlay
                    with ui.column().classes('absolute inset-0 flex items-center justify-center'):
                        ui.spinner('dots', size='lg', color='orange')
                        ui.label('ANALYZING...').classes('text-orange font-bold mt-2 tracking-widest')
                else:
                    ui.spinner('dots', size='lg', color='orange')

        # --- RESULT: Show image with interactive SVG masks ---
        # TODO: move into helper and use app_state for hardcoded values
        elif status == 'result':
            result = getattr(app_state.view, 'result', None)
            info = app_state.queued_files.get(app_state.active_file_id)
            
            if info:
                img_src = info.path  # This is a Path object or string path
                native_width = info.width
                native_height = info.height

                if isinstance(img_src, (Path, str)) and Path(img_src).exists():
                    # FIXED: Open the image with PIL just to read its native dimensions
                    with Image.open(img_src) as img:
                        native_width, native_height = img.size
                else:
                    # Fallback defaults if the path is an external URL or missing
                    native_width, native_height = 640, 640

                with ui.element('div').classes('w-full h-full relative'):
                    # Pass the high-res path directly to NiceGUI so it renders crisp
                    image_view = ui.interactive_image(img_src).classes('w-full h-full').style('object-fit: contain;')
                    
                    # Match the SVG viewbox exactly to the original high-res dimensions
                    image_view.view_box = [0, 0, native_width, native_height]
                    
                    def generate_svg_string(show_masks):
                        if not show_masks or not result or not result.sections:
                            return ""
                        
                        # Calculate how much we need to scale the 640x640 mask coordinates
                        scale_x = native_width / 640
                        scale_y = native_height / 640
                        
                        svg_content = ""
                        for sec in result.sections:
                            if sec.mask:
                                # Re-scale every coordinate point back to the original aspect ratio
                                scaled_points = []
                                for p in sec.mask:
                                    x_scaled = p[0] * scale_x
                                    y_scaled = p[1] * scale_y
                                    scaled_points.append(f"{x_scaled},{y_scaled}")
                                
                                points_str = " ".join(scaled_points)
                                svg_content += f'<polygon points="{points_str}" fill="rgba(242, 125, 38, 0.2)" stroke="#F27D26" stroke-width="2" />'
                        return svg_content

                    # Bind content
                    image_view.bind_content_from(
                        app_state.view, 
                        'show_masks', 
                        backward=generate_svg_string
                    )

        # --- ERROR: Show failure message ---
        elif status == 'error':
            msg = getattr(app_state, 'view_error', 'Backend Error')
            ui.label(msg).classes('text-red-600 font-bold')

    # --- NAVIGATION BAR (Fires simple callbacks, no controller info here) ---
    if len(app_state.queued_files) > 1:
        with ui.row().classes('w-full justify-center items-center gap-6 mt-2'):
            ui.button(icon='chevron_left', on_click=on_prev_callback if on_prev_callback else lambda: None) \
                .props('flat round dense size=lg').classes('text-gray-400 hover:text-white')
            
            files_keys = list(app_state.queued_files.keys())
            try:
                current_idx = files_keys.index(app_state.active_file_id)
            except ValueError:
                current_idx = 0
                
            ui.label(f"{current_idx + 1} / {len(files_keys)}").classes('text-gray-400 font-medium')
            
            ui.button(icon='chevron_right', on_click=on_next_callback if on_next_callback else lambda: None) \
                .props('flat round dense size=lg').classes('text-gray-400 hover:text-white')
