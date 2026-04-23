from pydantic import BaseModel, Field, ConfigDict, model_validator
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

class Detection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    class_name: str
    class_id: int = Field(alias="class")
    confidence: float
    bbox: List[float] = Field(default_factory=list)
    track_id: int = Field(default=0, alias="id")
    mask: List[List[float]] = Field(default_factory=list)
    overlap_ratio: float = 0.0
    area_in_pixels: int = 0
    section_image: Optional[np.ndarray] = None

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

class ProcessInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    img_path: Optional[str] = None
    frame: Optional[np.ndarray] = None

    @model_validator(mode='after')
    def check_exclusive_input(self) -> 'ProcessInput':
        if (self.img_path is None) == (self.frame is None):
            raise ValueError("Ambiguous input: Provide either 'img_path' OR 'frame', not both/neither.")
        return self