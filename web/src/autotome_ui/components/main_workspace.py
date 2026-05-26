from nicegui import ui
from autotome_ui.models.status import app_state
import logging


class MainWorkspace:
    def __init__(self, on_next_callback=None, on_prev_callback=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.on_next = on_next_callback
        self.on_prev = on_prev_callback

        if app_state.config and app_state.config.qc and app_state.config.qc.yolo and app_state.config.qc.yolo.img_dim:
            self.yolo_w = app_state.config.qc.yolo.img_dim[0]
            self.yolo_h = app_state.config.qc.yolo.img_dim[1]
        else:
            self.yolo_w, self.yolo_h = 640, 640

        # Zoom & Pan State
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_dragging = False
        self.start_x = 0
        self.start_y = 0
        self.image_view = None

        # Register the global keyboard event listener
        ui.keyboard(on_key=self._handle_key)

    @ui.refreshable
    def render(self):
        """Main entry point for rendering the workspace."""
        status = getattr(app_state.view, 'status', 'idle')

        # Standard container for centering everything
        image_container = ui.element('div').classes(
            'w-full h-[calc(100vh-160px)] flex-none '
            'flex items-center justify-center '
            'bg-black overflow-hidden m-auto '
            'self-center rounded-lg border border-[#222222]'
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
        self._render_zoomable_viewport(active_file, status='pending')

    def _render_processing(self):
        """Static render for processing state (no zoom/pan)."""
        active_file = app_state.queued_files.get(app_state.active_file_id)

        with ui.element('div').classes('w-full h-full relative flex items-center justify-center'):
            if active_file and active_file.img_src:
                # Expand image to full container size
                ui.image(active_file.img_src).classes('w-full h-full').props('fit="contain"').style('opacity: 0.4;')
                
                # The absolute inset-0 overlay will now cover the entire w-full h-full parent
                with ui.column().classes('absolute inset-0 flex items-center justify-center'):
                    ui.spinner('dots', size='lg', color='orange')
                    ui.label('ANALYZING...').classes('text-orange font-bold mt-2 tracking-widest')
            else:
                ui.spinner('dots', size='lg', color='orange')

    def _render_result(self):
        active_file = app_state.queued_files.get(app_state.active_file_id)
        self._render_zoomable_viewport(active_file, status='result')

    def _render_zoomable_viewport(self, active_file, status):
        """Reusable method to render the interactive image with zoom/pan for pending and result states."""
        with ui.element('div').classes('w-full h-full relative flex items-center justify-center'):
            if not active_file or not active_file.img_src:
                ui.spinner('dots', size='lg')
                return

            img_src = active_file.img_src
            native_width = getattr(active_file, 'width', 1000)
            native_height = getattr(active_file, 'height', 1000)

            self.zoom_level = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0

            # 1. Added 'flex items-center justify-center' to keep the image centered
            with ui.element('div').classes('w-full h-full relative flex items-center justify-center overflow-hidden group cursor-move') as wrapper:
                
                # 2. Changed w-full h-full to max-w-full max-h-full
                self.image_view = ui.interactive_image(img_src).classes('max-w-full max-h-full')
                
                # 3. Enforce the exact original aspect ratio directly on the component
                aspect_ratio = f"{native_width}/{native_height}"
                self.image_view.style(f'aspect-ratio: {aspect_ratio}; transform-origin: center; transition: transform 0.05s ease-out;')
                self.image_view.view_box = [0, 0, native_width, native_height]
                
                # Event Listeners for Zoom & Pan
                wrapper.on('wheel.prevent', self._handle_wheel, ['deltaY'])
                wrapper.on('mousedown.prevent', self._handle_mousedown, ['clientX', 'clientY', 'button'])
                wrapper.on('mousemove.prevent', self._handle_mousemove, ['clientX', 'clientY'])
                wrapper.on('mouseup', self._handle_mouseup)
                wrapper.on('mouseleave', self._handle_mouseup)

                # Floating Zoom Controls
                with ui.row().classes(
                    'absolute bottom-4 right-4 gap-2 opacity-0 group-hover:opacity-100 '
                    'transition-opacity duration-300 z-50 bg-black/70 p-2 rounded-lg'
                ):
                    ui.button(icon='remove', on_click=self.zoom_out).props('flat round color=white size=sm')
                    ui.button(icon='fit_screen', on_click=self.reset_zoom).props('flat round color=white size=sm')
                    ui.button(icon='add', on_click=self.zoom_in).props('flat round color=white size=sm')
                
                if status == 'result':
                    result = getattr(app_state.view, 'result', None)
                    self.image_view.bind_content_from(
                        app_state.view, 
                        'show_masks', 
                        backward=lambda show: self._generate_svg_string(show, result, native_width, native_height)
                    )

                self._apply_transform()

    # --- Zoom & Pan Event Handlers ---

    def _apply_transform(self):
        """Updates the CSS transform for zooming and panning."""
        if hasattr(self, 'image_view') and self.image_view:
            self.image_view.style(
                f'transform: translate({self.pan_x}px, {self.pan_y}px) scale({self.zoom_level});'
            )

    def zoom_in(self):
        self.zoom_level = min(self.zoom_level * 1.25, 10.0) # Max 10x Zoom
        self._apply_transform()

    def zoom_out(self):
        self.zoom_level = max(self.zoom_level / 1.25, 0.2) # Min 0.2x Zoom
        self._apply_transform()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._apply_transform()

    def _handle_wheel(self, e):
        delta = e.args.get('deltaY', 0)
        if delta > 0:
            self.zoom_out()
        elif delta < 0:
            self.zoom_in()

    def _handle_mousedown(self, e):
        button = e.args.get('button', 0)
        
        # If middle mouse button (1) is clicked, reset zoom and return
        if button == 1:
            self.reset_zoom()
            return
            
        # Otherwise, initiate dragging (usually left click -> 0)
        self.is_dragging = True
        self.start_x = e.args.get('clientX', 0) - self.pan_x
        self.start_y = e.args.get('clientY', 0) - self.pan_y

    def _handle_mousemove(self, e):
        if self.is_dragging:
            self.pan_x = e.args.get('clientX', 0) - self.start_x
            self.pan_y = e.args.get('clientY', 0) - self.start_y
            self._apply_transform()

    def _handle_mouseup(self, e):
        self.is_dragging = False

    # --- SVG Masks & Utilities ---

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
        msg = app_state.view.error or 'Backend Error'
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
        if not e.action.keydown:
            return

        if e.key.arrow_right and self.on_next:
            self.on_next()
        elif e.key.arrow_left and self.on_prev:
            self.on_prev()