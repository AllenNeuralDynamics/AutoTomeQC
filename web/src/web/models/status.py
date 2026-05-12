# Store Config from backend
from typing import Optional
from web.models.schemas import AppConfig

class AppState:
    """Holds the global state of the frontend, including the backend configuration."""
    
    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.is_backend_ready: bool = False

# Create a global instance to be imported and used across frontend components
app_state = AppState()