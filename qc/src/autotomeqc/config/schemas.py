from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path

# Dynamically locate the 'qc' directory based on this file's location
# schemas.py -> config -> autotomeqc -> src -> qc
QC_DIR = Path(__file__).resolve().parent.parent.parent.parent

def resolve_path(v: str | None) -> str | None:
    if not v:
        return v
    p = Path(v)
    if p.is_absolute():
        return v
    return str((QC_DIR / p).resolve())

class YoloSettings(BaseModel):
    weights_path: str
    conf_thresh: float = 0.25
    img_size: int = 640
    img_dim: List[int] = [640, 640]
    max_det: int = 10

    @field_validator('weights_path')
    @classmethod
    def make_absolute(cls, v):
        return resolve_path(v)

class PostProcessingSettings(BaseModel):
    """Schema for the yolo_post_processing section in yaml."""
    out_dim: List[int] = [640, 640]
    loop_bbox_margin: int = 30
    allow_no_loop: bool = True
    overlap_threshold: float = 0.5

class AlgorithmSettings(BaseModel):
    weights_path: Optional[str] = None
    img_size: int = 224
    img_dim: List[int] = [224, 224]
    pass_labels: List[str]
    min_confidence: float = 0.5

    @field_validator('weights_path')
    @classmethod
    def make_absolute(cls, v):
        return resolve_path(v)

class ShapeSettings(BaseModel):
    """Specific settings for the ShapeQC module."""
    save_debug_img: bool = True

class QCSettings(BaseModel):
    output_dir: str = "example/output"
    save_segmented_images: bool = True
    save_input_images: bool = True
    yolo: YoloSettings
    yolo_post_processing: PostProcessingSettings # type: ignore
    # Map the algorithm configs
    section_coverage: AlgorithmSettings
    knife_mark: AlgorithmSettings
    thickness_consistency: AlgorithmSettings
    thickness: AlgorithmSettings
    shape: ShapeSettings

    @field_validator('output_dir')
    @classmethod
    def make_absolute(cls, v):
        return resolve_path(v)

class AppConfig(BaseSettings):
    qc: QCSettings