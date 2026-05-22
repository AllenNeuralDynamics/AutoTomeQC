import os
import socket
import argparse
import logging
import subprocess
from pathlib import Path

def get_local_ip() -> str:
    """Helper to get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def configure_logging(level: str) -> None:
    """Configures the root logger based on a string level."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def set_environment_variables(backend_port: int, log_level: str) -> None:
    """Sets necessary environment variables for the backend and frontend processes."""
    # Provide the Base Backend URL dynamically to the Web UI
    os.environ["AUTOTOME_BACKEND_URL"] = f"http://localhost:{backend_port}"

    # Tell the Backend to disable disk writes and return mask data for the UI
    os.environ["AUTOTOME_SAVE_QC_JSON"] = "False"
    os.environ["AUTOTOME_SAVE_SEGMENTED"] = "False"
    os.environ["AUTOTOME_SAVE_INPUT"] = "False"
    os.environ["AUTOTOME_RETURN_MASK"] = "True"
    os.environ["LOG_LEVEL"] = log_level.upper()

def main():
    """
    Entry point for the 'autotome-ui' command defined in pyproject.toml.
    This launches both the FastAPI Backend and the NiceGUI Web UI.
    """
    parser = argparse.ArgumentParser(prog="autotome-ui", description="AutoTomeQC - Web UI & Backend Launcher")
    parser.add_argument("--backend-port", type=int, default=8000, help="Port for the FastAPI backend. (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8080, help="Port for the NiceGUI web UI. (default: 8080)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for both UI and backend. (default: INFO)"
    )
    args, unknown = parser.parse_known_args()

    configure_logging(level=args.log_level)
    log = logging.getLogger("autotome-ui")

    # Setup environment
    set_environment_variables(args.backend_port, args.log_level)

    # Backend) Start the Backend API in the background using the same python environment
    backend_cmd = [
        "uv", "run", "uvicorn", "autotomeqc.interface.server:app", "--host", "0.0.0.0",
        "--port", str(args.backend_port),
        "--log-level", args.log_level.lower(),
    ]
    log.info(f"Launching Backend API on port {args.backend_port}...")
    backend_process = subprocess.Popen(backend_cmd)

    # Frontend) Launch the Web UI in the foreground
    local_ip = get_local_ip()
    log.info("Launching Web UI...")
    log.info(f"Web UI: http://localhost:{args.ui_port}")
    if local_ip != "127.0.0.1":
        log.info(f"        or http://{local_ip}:{args.ui_port}")
    
    script_path = Path(__file__).resolve().parent / "main.py"
    frontend_cmd = [
        "uv", "run", "python",
        str(script_path), 
        "--port", str(args.ui_port)
    ]
    frontend_process = subprocess.Popen(frontend_cmd)

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        log.info("\nInterrupted by user. Shutting down services...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()
