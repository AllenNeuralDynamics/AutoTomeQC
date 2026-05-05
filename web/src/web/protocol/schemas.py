from pydantic import BaseModel
from typing import List, Dict, Optional

class QCCriteria(BaseModel):
    pass_status: bool
    label: str
    message: Optional[str] = None
    conf: Optional[float] = None
    metric: Optional[float] = None

class SectionResult(BaseModel):
    qc_result: str
    segmentation_conf: float
    area_in_pixels: int
    overlap_ratio: float
    criteria: Dict[str, QCCriteria]

class PipelineResult(BaseModel):
    filename: str
    timestamp: str
    qc_summary: str
    fail_reason: str
    processing_time_sec: Optional[float] = None
    sections: List[SectionResult] = []
