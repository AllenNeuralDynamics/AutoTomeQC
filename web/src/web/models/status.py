import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict, computed_field
from web.models.backend_schemas import PipelineResult, AppConfig

class QueuedFile(BaseModel):
    """Represents a file in the upload queue."""
    #model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    path: Path
    img_src: str
    json_path: Optional[Path] = None  # just save the path to save memory, load content on demand
    status: str = 'PENDING'  # 'PENDING', 'PROCESSING', 'PASS', 'FAIL', 'ERROR'
    width: Optional[int] = None
    height: Optional[int] = None

class ViewState(BaseModel):
    """Encapsulates UI rendering and volatile screen states."""
    status: str = 'idle'  # 'idle', 'pending', 'result', 'error', 'processing'
    error: Optional[str] = None
    result: Optional[PipelineResult] = None
    raw_json: Optional[Dict] = None
    show_masks: bool = True

    # Centralized UI color palette for section rendering (Stroke Hex, Fill RGBA)
    section_colors: List[Tuple[str, str]] = Field(default_factory=lambda: [
        ("#F27D26", "rgba(242, 125, 38, 0.2)"),   # Orange (Original)
        ("#26A69A", "rgba(38, 166, 154, 0.2)"),   # Teal
        ("#EF5350", "rgba(239, 83, 80, 0.2)"),    # Red
        ("#42A5F5", "rgba(66, 165, 245, 0.2)"),   # Blue
        ("#AB47BC", "rgba(171, 71, 188, 0.2)"),   # Purple
        ("#9CCC65", "rgba(156, 204, 101, 0.2)"),  # Light Green
    ])

class AppState(BaseModel):
    """Holds the global state of the frontend using Pydantic for validation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Backend state
    is_backend_ready: bool = False
    config: Optional[AppConfig] = None
    backend_url: str = Field(default_factory=lambda: os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000"))

    # Upload queue state
    queued_files: Dict[str, QueuedFile] = Field(default_factory=dict)
    active_file_id: Optional[str] = None
    is_processing: bool = False

    # UI view state
    view: ViewState = Field(default_factory=ViewState)

    # Storage paths
    temp_upload_dir: Path = Field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="autotome_")))
    print("temp upload dir:", temp_upload_dir)

    @computed_field
    @property
    def temp_upload_url_prefix(self) -> str:
        return f"/temp_files/{self.temp_upload_dir.name}"

    @computed_field
    @property
    def process_url(self) -> str:
        return f"{self.backend_url}/api/v1/process"

    @computed_field
    @property
    def is_ready_url(self) -> str:
        return f"{self.backend_url}/api/v1/is_ready"

    @computed_field
    @property
    def config_url(self) -> str:
        return f"{self.backend_url}/api/v1/config"

# Create a global instance to be imported and used across frontend components
app_state = AppState()