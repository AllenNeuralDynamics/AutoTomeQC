import os
import streamlit as st
import requests
import sys
import argparse
import logging
import subprocess
import logging
from streamlit.web import cli as stcli

def render_ui():
    # Read from environment variable, fallback to localhost for local development
    BACKEND_URL = os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8000/api/v1/process")

    st.set_page_config(page_title="AutoTomeQC", layout="centered")
    st.title("AutoTomeQC Dashboard")

    uploaded_file = st.file_uploader("Upload a section image", type=["jpg", "jpeg", "png", "tif", "tiff"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Input Image", width="stretch")
        
        if st.button("Run QC", type="primary"):
            with st.spinner("Processing in backend..."):
                # We send the raw bytes of the image over the network!
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                
                try:
                    response = requests.post(BACKEND_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"QC Summary: {result['qc_summary']} | Reason: {result['fail_reason']}")
                        st.json(result)  # Display the full JSON report beautifully
                    else:
                        st.error(f"Backend Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend. Is the FastAPI server running on port 8000?")

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

    # Provide the Backend URL dynamically to the Streamlit code above
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
    backend_process = subprocess.Popen(backend_cmd, cwd="qc")

    # Frontend) Launch the Web UI in the foreground
    local_ip = get_local_ip()
    log.info(f"Launching Web UI...")
    log.info(f"Web UI: http://localhost:{args.ui_port}")
    if local_ip != "127.0.0.1":
        log.info(f"        or http://{local_ip}:{args.ui_port}")
    script_path = os.path.abspath(__file__)
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
    render_ui()
