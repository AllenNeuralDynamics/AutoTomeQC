# Store Config from backend
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict
from web.models.schemas import AppConfig, QueuedFile

class AppState:
    """Holds the global state of the frontend, including the backend configuration."""

    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.queued_files: Dict[str, QueuedFile] = {}
        self.is_backend_ready: bool = False

        # Centralize backend URL configuration
        self.backend_url: str = os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000")
        self.process_url: str = f"{self.backend_url}/api/v1/process"
        self.is_ready_url: str = f"{self.backend_url}/api/v1/is_ready"
        self.config_url: str = f"{self.backend_url}/api/v1/config"

        # Create a persistent temporary directory for uploaded files for the session
        self.temp_upload_dir = Path(tempfile.mkdtemp(prefix="autotome_"))
        self.temp_upload_url_prefix = f"/temp_files/{self.temp_upload_dir.name}"

# Create a global instance to be imported and used across frontend components
app_state = AppState()