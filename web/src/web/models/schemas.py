# Schema for json ouput after processing an image through the pipeline. 
# This is what the backend sends to the frontend.
from autotomeqc.config.schemas import resolve_path
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional

from pydantic_settings import BaseSettings

# --- JSON Result --

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

# -- Config Schema --

class YoloSettings(BaseModel):
    weights_path: str
    conf_thresh: float
    img_size: int
    img_dim: List[int]
    max_det: int

class PostProcessingSettings(BaseModel):
    """Schema for the yolo_post_processing section in yaml."""
    out_dim: List[int]
    loop_bbox_margin: int
    allow_no_loop: bool
    overlap_threshold: float

class AlgorithmSettings(BaseModel):
    weights_path: Optional[str]
    img_size: int
    img_dim: List[int]
    pass_labels: List[str]
    min_confidence: float = 0.5

class ShapeSettings(BaseModel):
    """Specific settings for the ShapeQC module."""
    save_debug_img: bool = True

class QCSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    output_dir: str
    save_qc_json: bool = True
    save_segmented_images: bool = True
    save_input_images: bool = True
    return_mask_data: bool = False
    yolo: YoloSettings
    yolo_post_processing: PostProcessingSettings # type: ignore
    # Map the algorithm configs
    section_coverage: AlgorithmSettings
    knife_mark: AlgorithmSettings
    thickness_consistency: AlgorithmSettings
    thickness: AlgorithmSettings
    shape: ShapeSettings

class AppConfig(BaseSettings):
    qc: QCSettings