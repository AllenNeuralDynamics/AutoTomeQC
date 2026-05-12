# Services (Logic): Code that talks to the outside world. 
# It shouldn't know about anything related to UI.

import httpx
import time
from typing import Tuple
from web.protocol.schemas import PipelineResult

async def analyze_image(backend_url: str, file_path: str) -> Tuple[PipelineResult, dict]:
    """
    Sends the image path to the FastAPI backend, validates the response, 
    and returns both the Pydantic object and the raw JSON dictionary.
    """
    async with httpx.AsyncClient() as client:
        # Send the POST request asynchronously with the file path as a query parameter
        params = {"img_path": file_path}
        response = await client.post(backend_url, params=params, timeout=60.0)
        
        # Raise an exception if the status code is not 200 OK
        response.raise_for_status()
        
        raw_json = response.json()
        #print("Raw JSON response from backend:", raw_json)  # Debugging statement
        return PipelineResult.model_validate(raw_json), raw_json

def check_health(health_url: str) -> bool:
    """Synchronous health check for the backend."""
    try:
        with httpx.Client() as client:
            response = client.get(health_url, timeout=2.0)
            if response.status_code == 200 and response.json().get("status") == "ready":
                return True
    except Exception:
        pass
    return False

async def check_health_async(health_url: str) -> bool:
    """Asynchronous health check for the backend."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(health_url, timeout=2.0)
            if response.status_code == 200 and response.json().get("status") == "ready":
                return True
    except Exception:
        pass
    return False

def wait_for_backend(health_url: str, timeout_sec: int = 120) -> bool:
    """Polls the backend health endpoint until it is ready."""
    start_ts = time.time()
    while time.time() - start_ts < timeout_sec:
        if check_health(health_url):
            return True
        time.sleep(2.0)
    return False
