# web/components/main_workspace.py
from nicegui import ui
from web.models.status import app_state

class MainWorkspace:
    def __init__(self, on_next_callback=None, on_prev_callback=None):
        self.on_next = on_next_callback
        self.on_prev = on_prev_callback

        if app_state.config and app_state.config.qc and app_state.config.qc.yolo and app_state.config.qc.yolo.img_dim:
            self.yolo_w = app_state.config.qc.yolo.img_dim[0]
            self.yolo_h = app_state.config.qc.yolo.img_dim[1]
        else:
            self.yolo_w, self.yolo_h = 640, 640

        # Register the global keyboard event listener
        ui.keyboard(on_key=self._handle_key)

    @ui.refreshable
    def render(self):
        """Main entry point for rendering the workspace."""
        status = getattr(app_state.view, 'status', 'idle')
        """
        # Standard container for centering everything
        image_container = ui.element('div').classes(
            'w-full h-full max-w-[1200px] max-h-[1200px] aspect-square flex-none '
            'flex items-center justify-center '
            'bg-black overflow-hidden m-auto '
            'self-center rounded-lg border border-[#222222]'
        )"""

        
        with ui.column().classes('w-full h-full flex-nowrap items-center justify-between pb-4'):
            # 2. flex-1 allows the container to grow/shrink. min-h-0 prevents it from overflowing the parent.
            # Removed 'aspect-square' and 'h-full' to allow dynamic resizing.
            image_container = ui.element('div').classes(
                'w-full flex-1 min-h-0 max-w-[1200px] '
                'flex items-center justify-center '
                'bg-black overflow-hidden m-auto '
                'rounded-lg border border-[#222222]'
            )

            with image_container:
                if status == 'idle':
                    self._render_idle()
                elif status == 'pending':
                    self._render_pending()
                elif status == 'processing':
                    self._render_processing()
                elif status == 'result':
                    self._render_result()
                elif status == 'error':
                    self._render_error()

        self._render_navigation()

    def _render_idle(self):
        with ui.column().classes('items-center justify-center text-gray-500'):
            ui.icon('aspect_ratio', size='6rem')
            ui.label('VIEWPORT IDLE')

    def _render_pending(self):
        active_file = app_state.queued_files.get(app_state.active_file_id)
        # Use a full-size flex container to center contents
        with ui.element('div').classes('w-full h-full relative flex items-center justify-center'):
            if active_file and active_file.img_src:
                # Expand image to full container size and contain aspect ratio
                ui.image(active_file.img_src).classes('w-full h-full').style('object-fit: contain;')
            else:
                ui.spinner('dots', size='lg')

    def _render_processing(self):
        active_file = app_state.queued_files.get(app_state.active_file_id)

        with ui.element('div').classes('w-full h-full relative flex items-center justify-center'):
            if active_file and active_file.img_src:
                # Expand image to full container size
                ui.image(active_file.img_src).classes('w-full h-full').style('object-fit: contain; opacity: 0.4;')
                
                # The absolute inset-0 overlay will now cover the entire w-full h-full parent
                with ui.column().classes('absolute inset-0 flex items-center justify-center'):
                    ui.spinner('dots', size='lg', color='orange')
                    ui.label('ANALYZING...').classes('text-orange font-bold mt-2 tracking-widest')
            else:
                ui.spinner('dots', size='lg', color='orange')

    def _render_result(self):
        result = getattr(app_state.view, 'result', None)
        info = app_state.queued_files.get(app_state.active_file_id)
        
        if not info:
            return

        img_src = info.img_src
        native_width = info.width
        native_height = info.height

        with ui.element('div').classes('w-full h-full relative'):
            image_view = ui.interactive_image(img_src).classes('w-full h-full').style('object-fit: contain;')
            image_view.view_box = [0, 0, native_width, native_height]
            
            # Bind using a lambda to pass the necessary context to our class method
            image_view.bind_content_from(
                app_state.view, 
                'show_masks', 
                backward=lambda show: self._generate_svg_string(show, result, native_width, native_height)
            )

    def _generate_svg_string(self, show_masks, result, native_width, native_height):
        """Helper method to generate SVG masks. Now safely isolated."""
        if not show_masks or not result or not result.sections:
            return ""

        scale_x = native_width / self.yolo_w
        scale_y = native_height / self.yolo_h
        
        svg_content = ""
        colors = app_state.view.section_colors
        for i, sec in enumerate(result.sections):
            if sec.mask:
                stroke_color, fill_color = colors[i % len(colors)]
                scaled_points = [f"{p[0] * scale_x},{p[1] * scale_y}" for p in sec.mask]
                points_str = " ".join(scaled_points)
                svg_content += f'<polygon points="{points_str}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" />'
        return svg_content

    def _render_error(self):
        msg = getattr(app_state, 'view_error', 'Backend Error')
        ui.label(msg).classes('text-red-600 font-bold')

    def _render_navigation(self):
        if len(app_state.queued_files) <= 1:
            return
            
        with ui.row().classes('w-full justify-center items-center gap-6 mt-2'):
            ui.button(icon='chevron_left', on_click=self.on_prev if self.on_prev else lambda: None) \
                .props('flat round dense size=lg').classes('text-gray-400 hover:text-white')
            
            files_keys = list(app_state.queued_files.keys())
            try:
                current_idx = files_keys.index(app_state.active_file_id)
            except ValueError:
                current_idx = 0
                
            ui.label(f"{current_idx + 1} / {len(files_keys)}").classes('text-gray-400 font-medium')
            
            ui.button(icon='chevron_right', on_click=self.on_next if self.on_next else lambda: None) \
                .props('flat round dense size=lg').classes('text-gray-400 hover:text-white')
            
    def _handle_key(self, e):
        """Handle keyboard events for navigation."""
        # Only trigger on key down to prevent duplicate calls on key up
        if not e.action.keydown:
            return

        # Trigger corresponding callbacks based on arrow keys
        if e.key.arrow_right and self.on_next:
            self.on_next()
        elif e.key.arrow_left and self.on_prev:
            self.on_prev()