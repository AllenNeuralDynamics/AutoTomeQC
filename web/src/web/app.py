import os
import sys
import argparse
import logging
import subprocess
from streamlit.web import cli as stcli

def get_local_ip() -> str:
    """Helper to get the local IP address of the machine."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def configure_logging(level: int, fmt: str = "%(message)s", datefmt: str = "[%X]") -> None:
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def main():
    """
    Entry point for the 'autotome-ui' command defined in pyproject.toml.
    This launches both the FastAPI Backend and the Streamlit Web UI.
    """
    parser = argparse.ArgumentParser(prog="autotome-ui", description="AutoTomeQC - Web UI & Backend Launcher")
    parser.add_argument("--backend-port", type=int, default=8000, help="Port for the FastAPI backend (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8501, help="Port for the Streamlit UI (default: 8501)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # We use parse_known_args because Streamlit also reads sys.argv later
    args, unknown = parser.parse_known_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(level=log_level)
    log = logging.getLogger("autotome-ui")

    log.info("Starting AutoTomeQC Services...")

    # Provide the Backend URL dynamically to the Streamlit UI
    os.environ["AUTOTOME_BACKEND_URL"] = f"http://localhost:{args.backend_port}/api/v1/process"

    # Backend) Start the Backend API in the background using the same python environment
    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "autotomeqc.interface.server:app", 
        "--host", "0.0.0.0", 
        "--port", str(args.backend_port)
    ]
    if args.debug:
        backend_cmd.extend(["--log-level", "debug"])  
    log.info(f"Launching Backend API on port {args.backend_port}...")
    backend_process = subprocess.Popen(backend_cmd)

    # Frontend) Launch the Web UI in the foreground
    local_ip = get_local_ip()
    log.info(f"Launching Web UI...")
    log.info(f"Web UI: http://localhost:{args.ui_port}")
    if local_ip != "127.0.0.1":
        log.info(f"        or http://{local_ip}:{args.ui_port}")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_tmp.py")
    sys.argv = ["streamlit", "run", script_path, "--server.port", str(args.ui_port)]
    
    try:
        sys.exit(stcli.main())
    except KeyboardInterrupt:
        log.info("\nInterrupted by user.")
    finally:
        # Ensure the Backend cleanly shuts down when you exit the UI
        log.info("Shutting down Backend API...")
        backend_process.terminate()
        backend_process.wait()

if __name__ == "__main__":
    main()
