from nicegui import ui
from typing import Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
from web.models.schemas import PipelineResult

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

def set_workspace_idle(image_container):
    image_container.clear()
    with image_container:
        with ui.column().classes('viewport-idle'):
            ui.icon('aspect_ratio', size='6rem')
            ui.label('VIEWPORT_IDLE')

def set_workspace_pending(image_container, img_src=None):
    image_container.clear()
    with image_container:
        if img_src:
            with ui.element('div').classes('image-wrapper'):
                ui.image(img_src).classes('image-preview')
        else:
            ui.spinner('dots', size='lg')

def set_workspace_error(image_container, msg):
    image_container.clear()
    with image_container:
        ui.label(msg).classes('text-red-600 font-bold')

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

    set_workspace_idle(image_container)
    return image_container

def update_main_workspace(image_container, img_src: Union[str, Path], result: Optional[PipelineResult] = None):
    image_container.clear()
    
    # Resize the image to 640x640 to match YOLO's coordinate space for the SVG masks
    if isinstance(img_src, Path) and img_src.exists():
        img_src = Image.open(img_src).resize((640, 640))
    elif isinstance(img_src, str) and Path(img_src).exists():
        img_src = Image.open(img_src).resize((640, 640))

    with image_container:
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
    return