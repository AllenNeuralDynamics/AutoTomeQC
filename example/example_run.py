import subprocess
import time
from pathlib import Path

# --- Configuration ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
input_dir = current_dir / "input_images"

def main():
    if not input_dir.exists():
        print(f"MASTER: Error - Input directory not found: {input_dir}")
        return

    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print("MASTER: No images found.")
        return
    
    print("MASTER: Launching AutoTomeQC Service...")
    
    # We ONLY pipe stdin (to send commands). 
    # We let stdout/stderr go directly to the console so you can see them naturally.
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", "autotomeqc"],
        stdin=subprocess.PIPE,
        text=True,
        bufsize=0  # Unbuffered
    )
    print("MASTER: Waiting 10s for model initialization...")
    time.sleep(10)

    print(f"MASTER: Starting batch of {len(image_files)} images...")
    for img_path in image_files:
        print(f"\nMASTER: >>> Sending: {img_path.name}")
        try:
            # Write Path + Newline
            process.stdin.write(f"{img_path.absolute()}\n")
            process.stdin.flush()
        except BrokenPipeError:
            print("MASTER: Service died.")
            break
        # Wait 3s for precessing to simulate realistic pacing
        time.sleep(3.0) 

    # Stop
    print("\nMASTER: >>> Sending 'exit'.")
    try:
        process.stdin.write("exit\n")
        process.stdin.flush()
    except (BrokenPipeError, OSError):
            # Service might already be dead/closed, which is fine during shutdown
            pass
    except Exception as e:
        print(f"MASTER: Warning - failed to send exit command: {e}")

        process.wait()
        print("MASTER: Service closed.")

if __name__ == "__main__":
    main()
    # Example of running this script:
    # uv run python example/example_run.py