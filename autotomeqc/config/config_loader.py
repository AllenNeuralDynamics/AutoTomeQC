# src/autotomeqc/config/config_loader.py

import yaml
from pathlib import Path
import os


TEST_IMG_DIR = Path(r"C:\Users\hanna.lee\Documents\01_lasso\001_data\Training_data_segmask\Trained\qc3_postpickup_20251027_cropped")
TEST_OUT_DIR = Path(r"C:\Users\hanna.lee\Documents\01_lasso\001_data\Training_data_segmask\Trained\qc3_postpickup_20251027_cropped_out")
os.makedirs(TEST_OUT_DIR, exist_ok=True)

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