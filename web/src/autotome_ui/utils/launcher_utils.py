import os
import socket
import logging
import sys
from typing import Optional

def get_local_ip() -> str:
    """Helper to get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually connect; just triggers route selection
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def configure_logging(level: str = "INFO", name: Optional[str] = None) -> None:
    """Configures the root logger based on a string level."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True
    )

def set_launcher_environment(backend_port: int, log_level: str) -> None:
    """Sets necessary environment variables for the backend and frontend processes."""
    os.environ["AUTOTOME_BACKEND_URL"] = f"http://localhost:{backend_port}"
    os.environ["AUTOTOME_SAVE_QC_JSON"] = "False"
    os.environ["AUTOTOME_SAVE_SEGMENTED"] = "False"
    os.environ["AUTOTOME_SAVE_INPUT"] = "False"
    os.environ["AUTOTOME_RETURN_MASK"] = "True"
    os.environ["LOG_LEVEL"] = log_level.upper()