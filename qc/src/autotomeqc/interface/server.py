from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2

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
def process_image(file: UploadFile = File(...)):
    try:
        # TODO: Return segmented images as well.
        # Read the uploaded image into a numpy array (cv2 frame)
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file provided.")

        # Pass the raw frame to your existing pipeline
        future_ticket = service.process(frame=frame)
        result = future_ticket.result()  # Wait for the worker thread to finish
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
