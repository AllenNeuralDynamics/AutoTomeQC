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

    # Centralized UI color palette for section rendering - Allen Brand color 
    section_colors: List[Tuple[str, str]] = Field(default_factory=lambda: [
        ("#FF6E00", "rgba(255, 110, 0, 0.1)"),   # Orange
        ("#6464FF", "rgba(100, 100, 255, 0.1)"), # Blue
        ("#FF00FF", "rgba(255, 0, 255, 0.1)"),   # Rose
        ("#C0DB05", "rgba(192, 219, 5, 0.1)"),   # Green
        ("#CD0F55", "rgba(205, 15, 85, 0.1)"),   # Maroon
        ("#00A998", "rgba(0, 169, 152, 0.1)"),    # Teal
        ("#8246E1", "rgba(130, 70, 225, 0.1)"),  # Violet
        ("#FFE823", "rgba(255, 232, 35, 0.1)"),  # Yellow
        ("#DC9E00", "rgba(220, 158, 0, 0.1)"),   # Ochre
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