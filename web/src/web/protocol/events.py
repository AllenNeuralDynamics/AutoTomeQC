# Event definition
from typing import Any, Optional, Tuple
from pathlib import Path
from nicegui import Event

from web.models.schemas import PipelineResult

# --- Global UI Events ---
clear_views = Event[None]()

# --- Image/Workspace Events ---
image_selected = Event[Tuple[Path, "PipelineResult", dict]]()
image_pending = Event[Optional[str]]()
image_error = Event[str]()
