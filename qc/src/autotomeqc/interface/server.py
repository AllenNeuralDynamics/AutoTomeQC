# src/autotomeqc/interface/server.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from autotomeqc.core.autotome_service import AutoTomeService
from autotomeqc.config.config_loader import load_app_config
from autotomeqc.utils.logging_utils import setup_logging

# logging setup: Read log level from env var, default to INFO
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
setup_logging(default_level=log_level_name)

# Setup Config
base_config = load_app_config()
base_config.qc.save_qc_json = os.getenv("AUTOTOME_SAVE_QC_JSON", "False").lower() == "true"
base_config.qc.save_segmented_images = os.getenv("AUTOTOME_SAVE_SEGMENTED", "False").lower() == "true"
base_config.qc.save_input_images = os.getenv("AUTOTOME_SAVE_INPUT", "False").lower() == "true"
base_config.qc.return_mask_data = os.getenv("AUTOTOME_RETURN_MASK", "True").lower() == "true"

# Initialize the service
service = AutoTomeService(config=base_config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not service.start():
        raise RuntimeError("Failed to start AutoTomeService.")
    yield
    service.stop()

# Create FastAPI app with lifespan
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
