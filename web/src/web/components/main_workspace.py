from nicegui import ui
from typing import Union
from pathlib import Path

def render_main_workspace():
    """Renders the central image viewing area and returns the image container."""
    ui.query('.q-page').classes('main-workspace bg-grid')
    image_container = ui.column().classes('image-container')
    with image_container:
        with ui.column().classes('viewport-idle'):
            ui.icon('aspect_ratio', size='6rem')
            ui.label('VIEWPORT_IDLE')
    return image_container

def update_main_workspace(image_container, img_src: Union[str, Path]):
    """Updates the central image viewing area with a new image."""
    image_container.clear()
    with image_container:
        with ui.element('div').classes('image-wrapper'):
            ui.image(img_src).classes('image-preview')