import time
import json
from autotomeqc.core.autotome_service import AutoTomeService

def main():
    service = AutoTomeService()
    if not service.start():
        return

    print("Warming up...")
    time.sleep(3)

    try:
        for i in range(10):
            path = f"example/input_images/img{i}.jpg"
            print(f"\nProcessing: {path}")

            # ---------------------------------------------------------
            # WAY 1: Pass by File Path (Service loads the file)
            # ---------------------------------------------------------
            future = service.process(img_path=path)

            # ---------------------------------------------------------
            # WAY 2: Pass by Raw Frame (You load the file)
            # ---------------------------------------------------------
            #frame = cv2.imread(path)
            #if frame is None: continue 
            #future = service.process(frame=frame)
            # ---------------------------------------------------------
            
            # Wait for result
            while not future.done():
                time.sleep(0.1)
            
            # Print output
            result = future.result()
            print(f"Status: {result.get('qc_summary')}")
            print(json.dumps(result.get('criteria', {}), indent=4, default=str))

    except KeyboardInterrupt:
        pass
    finally:
        service.stop()

if __name__ == "__main__":
    main()