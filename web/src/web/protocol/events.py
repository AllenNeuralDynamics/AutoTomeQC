# Event definition
from typing import Callable, Generic, TypeVar, Tuple, Optional
from pathlib import Path
from web.models.schemas import PipelineResult
from nicegui import Event

# Global UI Events
image_selected = Event[Tuple[Path, PipelineResult, dict]]()
image_pending = Event[Optional[str]]()
image_error = Event[str]()
clear_views = Event[None]()
config_requested = Event[None]()
export_requested = Event[None]()