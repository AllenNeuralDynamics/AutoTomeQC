from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Optional, List
from concurrent.futures import Future
import numpy as np

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

class PipelineTask(BaseModel):
    #  Allow non-primitive types like np.ndarray and Future
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame: Optional[np.ndarray] = None
    filename: str = "unknown"
    timestamp: str = "N/A"
    start_ts: float
    future: Future