from nicegui import ui
from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image

"""
def render_main_workspace():
    ui.query('.q-page').classes('main-workspace bg-grid')
    image_container = ui.column().classes('image-container')
    with image_container:
        with ui.column().classes('viewport-idle'):
            ui.icon('aspect_ratio', size='6rem')
            ui.label('VIEWPORT_IDLE')
    return image_container

def update_main_workspace(image_container, img_src: Union[str, Path]):
    image_container.clear()
    with image_container:
        with ui.element('div').classes('image-wrapper flex justify-center items-center'): # Re-add this wrapper
            # Both URL strings and local Paths work here. We constrain it to 640x640.
            ui.image(img_src).classes('image-preview').style('max-width: 640px; max-height: 640px; width: 100%; object-fit: contain;')
            print("img_src for main workspace:", img_src)
            image = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
            ui.image(image).classes('w-32')

"""

def render_main_workspace():
    ui.query('.q-page').classes('flex flex-col items-stretch bg-[#0A0A0A]')
    image_container = ui.element('div').classes('w-full flex-grow flex items-center justify-center bg-black overflow-hidden min-h-[600px]')
    with image_container:
        ui.icon('aspect_ratio', size='6rem').classes('opacity-10 text-white')
    return image_container

def update_main_workspace(image_container, img_src: Union[str, Path]):
    image_container.clear()
    with image_container:
        #with ui.element('div').classes('flex-1 w-full h-full flex items-center justify-center'):
        #ui.image(img_src).classes('image-preview').style('w-full h-full object-contain')
        #image = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        ui.image(img_src).classes('w-full h-full object-cover')
    return        