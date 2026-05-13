# src/autotomeqc/interface/server.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from autotomeqc.core.autotome_service import AutoTomeService
from autotomeqc.config.config_loader import load_app_config

# 1. Load the Base Config (from YAML)
base_config = load_app_config()

# 2. Override settings using Environment Variables
# The Web UI launcher sets these, otherwise they default to standard behavior
base_config.qc.save_qc_json = os.getenv("AUTOTOME_SAVE_QC_JSON", "False").lower() == "true"
base_config.qc.save_segmented_images = os.getenv("AUTOTOME_SAVE_SEGMENTED", "False").lower() == "true"
base_config.qc.save_input_images = os.getenv("AUTOTOME_SAVE_INPUT", "False").lower() == "true"
base_config.qc.return_mask_data = os.getenv("AUTOTOME_RETURN_MASK", "True").lower() == "true"

# 3. Instantiate the service ONCE with the finalized config
# This is "locked in" for the life of the server process.
service = AutoTomeService(config=base_config)

# Manage the service lifecycle (Start up & Shut down)
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not service.start():
        raise RuntimeError("Failed to start AutoTomeService.")
    yield
    service.stop()

app = FastAPI(title="AutoTomeQC API", lifespan=lifespan)


# Endpoint to check if the service is ready
@app.get("/api/v1/is_ready")
def running_check():
    if service.running:
        return {"status": "running"}
    raise HTTPException(status_code=503, detail="Service is not running or still initializing.")

# Endpoint to fetch the current active configuration
@app.get("/api/v1/config")
def get_config():
    return service.config.model_dump()

# Create the endpoint
@app.post("/api/v1/process")
def process_image(img_path: str):
    try:
        # Pass the file path directly to the pipeline
        future_ticket = service.process(img_path=img_path)
        result = future_ticket.result()  # Wait for the worker thread to finish
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
