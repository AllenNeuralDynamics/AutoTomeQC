import os

import httpx
import streamlit as st

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
                    with httpx.Client() as client:
                        response = client.post(BACKEND_URL, files=files, timeout=60.0)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"QC Summary: {result['qc_summary']} | Reason: {result['fail_reason']}")
                        st.json(result)  # Display the full JSON report beautifully
                    else:
                        st.error(f"Backend Error: {response.text}")
                        
                except httpx.RequestError:
                    st.error(f"Failed to connect to the backend. Is the FastAPI server running on {BACKEND_URL}?")

if __name__ == "__main__":
    render_ui()