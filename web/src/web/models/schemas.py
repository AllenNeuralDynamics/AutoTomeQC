from pydantic import BaseModel
from typing import List, Dict, Optional

class QCCriteria(BaseModel):
    pass_status: bool
    label: str
    message: Optional[str] = None
    conf: Optional[float] = None
    metric: Optional[float] = None
    reason: Optional[str] = None

class SectionResult(BaseModel):
    qc_result: str
    segmentation_conf: float
    area_in_pixels: int
    overlap_ratio: float
    criteria: Dict[str, QCCriteria]
    mask: Optional[List[List[float]]] = None
    
class PipelineResult(BaseModel):
    filename: str
    timestamp: str
    qc_summary: str
    fail_reason: str
    processing_time_sec: Optional[float] = None
    sections: List[SectionResult] = []

class UIOutputSettings(BaseModel):
    """Lightweight schema for frontend UI state management and API requests."""
    save_qc_json: bool = True
    save_segmented_images: bool = True
    save_input_images: bool = True
    return_mask_data: bool = False
