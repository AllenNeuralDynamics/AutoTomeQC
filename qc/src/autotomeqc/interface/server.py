from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from autotomeqc.core.autotome_service import AutoTomeService

# Instantiate the service globally
service = AutoTomeService()

# Manage the service lifecycle (Start up & Shut down)
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not service.start():
        raise RuntimeError("Failed to start AutoTomeService.")
    yield
    service.stop()

app = FastAPI(title="AutoTomeQC API", lifespan=lifespan)

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
