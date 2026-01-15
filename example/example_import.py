import time
from autotomeqc.core.autotomeService import AutoTomeService

def main():
    print("Starting AutoTomeQC Service...")
    service = AutoTomeService()
    service.start()
    try:
        for i in range(5):
            service.process(f"example/input_images/img{i}.jpg")
            time.sleep(0.5)
    except Exception as e:
        print(f"Job Failed: {e}")
    finally:
        print("Stopping Service...")
        service.stop()

if __name__ == "__main__":
    main()
    # uv run python example/example_import.py