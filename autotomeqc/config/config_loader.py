# src/autotomeqc/config/config_loader.py

import yaml
from pathlib import Path

def load_app_config(filename: str = 'yolo-config.yaml') -> dict:
    """
    Finds the project root and loads the specified configuration file 
    from the project's root 'config' directory.
    """
    
    this_file_path = Path(__file__).resolve()
    project_root = this_file_path.parent.parent
    
    # 3. Construct the full path to the configuration file
    config_path = project_root / 'config' / filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # 4. Load and return the YAML content
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Define the global configuration dictionary (loaded only once)
try:
    CONFIG = load_app_config()
except FileNotFoundError as e:
    print(f"FATAL CONFIG ERROR: {e}")
    CONFIG = {} # Fallback to empty dictionary