import os
import socket
import argparse
import logging
import subprocess
from pathlib import Path
from web.utils.launcher_utils import (
    get_local_ip,
    set_launcher_environment
)

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
    parser.add_argument("--web", action="store_true", help="Run in web browser instead of native window")
    args = parser.parse_known_args()[0]

    # Setup environment
    set_launcher_environment(args.backend_port, args.log_level)

    # Backend) Start the Backend API in the background using the same python environment
    backend_cmd = [
        "uv", "run", "uvicorn", "autotomeqc.interface.server:app", "--host", "0.0.0.0",
        "--port", str(args.backend_port),
        "--log-level", args.log_level.lower(),
    ]
    backend_process = subprocess.Popen(backend_cmd)
    
    # Frontend) Start the NiceGUI frontend in the background
    script_path = Path(__file__).resolve().parent / "main.py"
    frontend_cmd = [
        "uv", "run", "python",
        str(script_path), 
        "--port", str(args.ui_port),
        "--log-level", args.log_level
    ]
    if args.web:
        frontend_cmd.append("--web")
    frontend_process = subprocess.Popen(frontend_cmd)

    # Print logging
    local_ip = get_local_ip()
    print("="*50)
    print("AutoTomeUI is now running!")
    print(f"Local:   http://localhost:{args.ui_port}")
    if local_ip != "127.0.0.1":
        print(f"Network: http://{local_ip}:{args.ui_port}")
    print("="*50)

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down services...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()
