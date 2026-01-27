import subprocess
import time
import sys
from pathlib import Path

# --- Configuration ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
input_dir = current_dir / "input_images"

def main():
    # Validation
    if not input_dir.exists():
        print(f"[MASTER] Error: Input directory not found at {input_dir}")
        return

    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print("[MASTER] No images found to process.")
        return
    
    print(f"[MASTER] Found {len(image_files)} images.")
    print("[MASTER] Launching AutoTomeQC Service...")
    
    # Launch Service as a Subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "autotomeqc"],
        stdin=subprocess.PIPE,
        text=True,     # Work with strings, not bytes
        bufsize=1,     # Line buffered (send commands immediately)
        cwd=project_root
    )

    # Warmup
    # The service needs a moment to load YOLO/PyTorch models
    print("[MASTER] Waiting 5s for service initialization...")
    time.sleep(5)

    # Batch Processing Loop
    print(f"[MASTER] Starting batch submission...")
    try:
        for i, img_path in enumerate(image_files):
            print(f"\n[MASTER] >>> Command: Process {img_path.name}")
            process.stdin.write(f"{img_path.absolute()}\n")
            time.sleep(2) 

        # Shutdown
        print("\n[MASTER] >>> Command: Stop")
        process.stdin.write("stop\n")
        
    except BrokenPipeError:
        print("[MASTER] Error: Service pipe closed unexpectedly.")
    except KeyboardInterrupt:
        print("\n[MASTER] Interrupted. Closing service...")
        
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait() # Wait for service to finish its cleanup
        print("[MASTER] Batch run complete.")

if __name__ == "__main__":
    main()