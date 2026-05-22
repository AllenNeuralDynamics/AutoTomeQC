# example/example_subprocess.py
import json
import subprocess
from pathlib import Path


# --- Configuration to simulate user input and environment ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
config_path = project_root / "qc" / "src" / "autotomeqc" / "config" / "yolo-config.yaml"
input_dir = current_dir / "input_images"
READY_SIGNAL = "System Ready"

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
    cmd = ["uv", "run", "autotomeqc"]
    if config_path.exists():
        cmd.extend(["--config", str(config_path)])

    # 3. Launch Service as a Subprocess
    # We use pipes to communicate with the headless service
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,     # Work with strings (JSON), not bytes
        bufsize=1      # Line buffered for real-time interaction
    )

    # Warmup
    print("[MASTER] Waiting for initialization...")
    while True:
        line = process.stdout.readline()
        if not line:
            break  # Process died
        if READY_SIGNAL in line:
            print("[MASTER] >>> Service Ready.")
            break

    # Batch Processing Loop
    print("[MASTER] Starting batch submission...")
    try:
        for i, img_path in enumerate(image_files):
            print(f"\n[MASTER] >>> Command: Process {img_path.name}")

            # Send image path to service
            process.stdin.write(f"{img_path.absolute()}\n")
            process.stdin.flush()

            # Receive (Blocks until done)
            result = json.loads(process.stdout.readline())
            print(f"[MASTER] Result: {result}")

        # Shutdown
        print("\n[MASTER] >>> Command: Stop")
        process.stdin.write("stop\n")
        process.stdin.flush()
        
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