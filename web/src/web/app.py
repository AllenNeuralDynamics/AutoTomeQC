import os
import streamlit as st
import requests

# Read from environment variable, fallback to localhost for local development
BACKEND_URL = os.getenv("AUTOTOME_BACKEND_URL", "http://localhost:8080/api/v1/process")

st.set_page_config(page_title="AutoTomeQC", layout="centered")
st.title("AutoTomeQC Dashboard")

uploaded_file = st.file_uploader("Upload a section image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input Image", use_container_width=True)
    
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

def main():
    """
    Entry point for the 'web' command defined in pyproject.toml.
    This allows running the UI by simply typing `uv run web` in the terminal.
    """
    import sys
    from streamlit.web import cli as stcli
    
    script_path = os.path.abspath(__file__)
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())
