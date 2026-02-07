from pydantic import BaseModel, Field, computed_field
from typing import Dict, Any, Optional, Union
from datetime import datetime

class QCCriteria(BaseModel):
    """
    Schema for individual algorithm check results.
    Handles varying keys like 'metric', 'conf', or 'message'.
    """
    pass_status: bool = Field(..., alias="pass")
    label: Optional[str] = None
    conf: Optional[float] = None
    metric: Optional[Union[int, float]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    reason: Optional[str] = None

    class Config:
        # Allows using 'pass' in input dicts while mapping to 'pass_status'
        populate_by_name = True

class PipelineResult(BaseModel):
    """
    Main schema for AutoTomeQC pipeline results.
    Reflects the structure used in AutoTomeQC core logic.
    """
    filename: str
    timestamp: str
    qc_summary: str  # "PASS" or "FAIL"
    processing_time_sec: Optional[float] = None
    segmentation_conf: float = 0.0
    overlap_ratio: float = 0.0
    
    # Typed dictionary to validate each algorithm output
    criteria: Dict[str, QCCriteria] = Field(default_factory=dict)
    
    # Optional field used during pipeline failures
    error_reason: Optional[str] = None

    @computed_field
    def log_status(self) -> str:
        """Standardized log message for pipeline reporting."""
        if self.qc_summary == "FAIL" and self.error_reason:
            return f"[{self.filename}] Pipeline Rejected: {self.error_reason}"
        return f"[{self.filename}] Pipeline {self.qc_summary}"

    class Config:
        populate_by_name = True