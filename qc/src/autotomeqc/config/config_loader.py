# Temp file for testing
# src/autotomeqc/config/config_loader.py

import yaml
from pathlib import Path
from autotomeqc.config.schemas import AppConfig

# TODO Use the adapter or validate using pydantic
def load_app_config() -> AppConfig:
    # Get the directory where THIS file (config_loader.py) lives
    config_dir = Path(__file__).resolve().parent

    # Point directly to the yaml in the same folder
    config_path = config_dir / 'yolo-config.yaml'

    if not config_path.exists():
        print(f"DEBUG: Looking for config at: {config_path.absolute()}")
        raise FileNotFoundError(f"Config missing at: {config_path}")

    with open(config_path, 'r') as f:
        raw_yaml = yaml.safe_load(f)

    return AppConfig(**raw_yaml)

CONFIG = load_app_config()