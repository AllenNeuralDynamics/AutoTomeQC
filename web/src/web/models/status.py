# Store Config from backend
import os
from typing import Optional
from web.models.schemas import AppConfig

class AppState:
    """Holds the global state of the frontend, including the backend configuration."""

    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.is_backend_ready: bool = False

        # Centralize backend URL configuration
        self.backend_url: str = os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000")
        self.process_url: str = f"{self.backend_url}/api/v1/process"
        self.is_ready_url: str = f"{self.backend_url}/api/v1/is_ready"
        self.config_url: str = f"{self.backend_url}/api/v1/config"

# Create a global instance to be imported and used across frontend components
app_state = AppState()