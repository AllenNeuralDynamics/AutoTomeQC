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
    #image_container = ui.element('div').classes('w-full flex-grow flex items-center justify-center bg-black overflow-hidden min-h-[600px]')
    #image_container = ui.element('div').classes(
    #    'w-full flex-grow flex-1 h-full min-h-0 flex items-center justify-center bg-black overflow-hidden'
    #)
    # Flex container that fills available space, centers content, and has a black background
    image_container = ui.element('div').classes(
        'w-full h-full flex-1 flex-grow min-h-0 ' # Expansion
        'flex items-center justify-center '       # Content Centering
        'bg-black overflow-hidden self-center'    # Alignment & Style
    )
    # Non-stretching image with object-fit to contain, wrapped in a div that centers it
    image_container = ui.element('div').classes(
        'w-[640px] h-[640px] flex-none '       # Strict Fixed Dimensions
        'flex items-center justify-center '    # Centers the icon/label inside
        'bg-black overflow-hidden m-auto '     # Centers the div itself in the parent
        'self-center rounded-lg border border-[#222222]'   # Style
    )
    # Fixed-size image with object-fit to contain, wrapped in a div that centers it

    with image_container:
        with ui.column().classes('viewport-idle'):
            ui.icon('aspect_ratio', size='6rem')
            ui.label('VIEWPORT_IDLE')
    return image_container

def update_main_workspace(image_container, img_src: Union[str, Path]):
    image_container.clear()
    with image_container:
        #with ui.element('div').classes('flex-1 w-full h-full flex items-center justify-center'):
        #ui.image(img_src).classes('image-preview').style('w-full h-full object-contain')
        #image = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        # Expand
        #ui.image(img_src).classes('w-full h-full object-cover')
        #ui.image(img_src).props('fit=scale-down')
        
        # Fixed image size
        # Check img_src dimension: f
        # TODO get size
        ui.image(img_src).props('width=640 height=640')
    return        