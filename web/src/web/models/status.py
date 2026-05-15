import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict, computed_field
from web.models.backend_schemas import PipelineResult, AppConfig

class QueuedFile(BaseModel):
    """Represents a file in the upload queue."""
    #model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    path: Path
    img_src: str
    json_path: Optional[Path] = None
    status: str = 'PENDING'  # 'PENDING', 'PROCESSING', 'PASS', 'FAIL', 'ERROR'
    is_active: bool = False


class AppState(BaseModel):
    """Holds the global state of the frontend using Pydantic for validation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Backend state
    is_backend_ready: bool = False
    config: Optional[AppConfig] = None
    backend_url: str = Field(default_factory=lambda: os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000"))

    # Upload queue state
    queued_files: Dict[str, QueuedFile] = Field(default_factory=dict)
    is_processing: bool = False

    # View State
    view_status: str = 'idle'  # 'idle', 'pending', 'result', 'error', 'processing'
    view_error: Optional[str] = None
    view_result: Optional[PipelineResult] = None
    view_raw_json: Optional[Dict] = None
    view_show_masks: bool = True

    # Storage paths
    temp_upload_dir: Path = Field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="autotome_")))

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