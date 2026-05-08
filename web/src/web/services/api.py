# Services (Logic): Code that talks to the outside world. 
# It shouldn't know about anything related to UI.

import httpx
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
