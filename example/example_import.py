import time
from autotomeqc.core.autotome_service import AutoTomeService

def main():
    print("Starting AutoTomeQC Service...")
    service = AutoTomeService()
    if not service.start():
        print("Failed to start service. Exiting.")
        return
    
    # Give the YOLO model a second to warm up
    print("\nWarming up models...")
    time.sleep(3)

    # Process a batch of example images
    try:
        for i in range(10):
            try:
                image_path = f"example/input_images/img{i}.jpg"
                print(f"\n[{i+1}/10] Sending: {image_path}")
                future = service.process(image_path)  # Submit the job & get the 'Ticket' (Future)
                while not future.done():
                    time.sleep(0.5)
                result_data = future.result()
                status = result_data.get("qc_summary", "UNKNOWN")
                print(f"\n   Complete! Status: {status}")
                print(f"   Details: {result_data.get('criteria', 'No Data')}")
            except Exception as e:
                print(f"Processing Failed: {image_path}: {e}")
    except KeyboardInterrupt:
        print("User interrupted.")
    finally:
        print("Stopping Service...")
        service.stop()

if __name__ == "__main__":
    main()