# src/autotomeqc/config/config_loader.py
import yaml  # type: ignore[import-untyped]
from pathlib import Path
from typing import Optional
from autotomeqc.config.schemas import AppConfig


def load_app_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Accepts a string path or None. 
    Loads the YAML and validates it via Pydantic.
    """
    if config_path is None:
        target_path = Path(__file__).resolve().parent / 'yolo-config.yaml'  # Default config
    else:
        target_path = Path(config_path)

    # Basic verification
    if not target_path.exists():
        raise FileNotFoundError(f"Config not found at: {target_path.absolute()}")

    # Load and Return as Pydantic Object
    with open(target_path, 'r') as f:
        raw_yaml = yaml.safe_load(f)

    return AppConfig(**raw_yaml)