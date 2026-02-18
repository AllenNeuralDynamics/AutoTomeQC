from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

"""
class QCCriteria(BaseModel):
    # Standard Pydantic V2 configuration
    model_config = ConfigDict(populate_by_name=True)

    pass_status: bool = Field(False)
    label: Optional[str] = None
    conf: Optional[float] = None
    metric: Optional[Union[int, float]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    reason: Optional[str] = None

class PipelineResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    filename: str
    timestamp: str
    qc_summary: str
    processing_time_sec: Optional[float] = None
    segmentation_conf: float = 0.0
    overlap_ratio: float = 0.0
    criteria: Dict[str, QCCriteria] = Field(default_factory=dict)
    fail_reason: Optional[str] = None

"""

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
    sections: Dict[str, SectionResult] = Field(default_factory=dict)  # "sections": { "0": {...} } structure