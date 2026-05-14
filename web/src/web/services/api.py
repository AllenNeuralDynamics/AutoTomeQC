# Services (Logic): Code that talks to the outside world. 
# It shouldn't know about anything related to UI.

import httpx
from typing import Tuple
from web.models.backend_schemas import PipelineResult, AppConfig
from web.models.status import app_state


async def analyze_image(process_url: str, file_path: str) -> Tuple[PipelineResult, dict]:
    """
    Sends the image path to the FastAPI backend, validates the response, 
    and returns both the Pydantic object and the raw JSON dictionary.
    """
    async with httpx.AsyncClient() as client:
        # Send the POST request asynchronously with the file path as a query parameter
        params = {"img_path": file_path}
        response = await client.post(process_url, params=params, timeout=60.0)
        
        # Raise an exception if the status code is not 200 OK
        response.raise_for_status()
        
        raw_json = response.json()
        #print("Raw JSON response from backend:", raw_json)  # Debugging statement
        return PipelineResult.model_validate(raw_json), raw_json

async def fetch_config_async(config_url: str) -> AppConfig:
    """Fetches the active configuration from the backend."""
    async with httpx.AsyncClient() as client:
        response = await client.get(config_url, timeout=5.0)
        response.raise_for_status()
        
        return AppConfig.model_validate(response.json())

async def is_running_async(health_url: str) -> bool:
    """Asynchronous health check for the backend."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(health_url, timeout=2.0)
            if response.status_code == 200 and response.json().get("status") == "running":
                return True
    except Exception:
        pass
    return False
