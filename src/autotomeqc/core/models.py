from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

class QCCriteria(BaseModel):
    pass_status: bool
    label: str
    conf: Optional[float] = None
    metric: Optional[Any] = None
    message: Optional[str] = None
    reason: Optional[str] = None

class SectionResult(BaseModel):
    qc_result: str  # "PASS" or "FAIL"
    segmentation_conf: float
    area_in_pixels: int
    overlap_ratio: float
    criteria: Dict[str, QCCriteria]

class PipelineResult(BaseModel):
    filename: str
    timestamp: str
    qc_summary: str
    fail_reason: str = "N/A"
    processing_time_sec: Optional[float] = None
    sections: List[SectionResult] = Field(default_factory=list)