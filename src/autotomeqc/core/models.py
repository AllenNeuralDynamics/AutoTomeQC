from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, Union

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
    error_reason: Optional[str] = None