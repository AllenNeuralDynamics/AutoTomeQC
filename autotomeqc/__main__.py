import logging
import sys
import time
from autotomeqc.config.config_loader import TEST_IMG_DIR
from autotomeqc.core.pipeline import AutoTomePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main() -> None:
    pipeline = AutoTomePipeline()
    pipeline.start()

    # Temporary: Process all images in the test directory
    # This simulates feeding images to the pipeline
    image_files = list(TEST_IMG_DIR.glob("*.jpg"))

    try:
        for file_path in image_files[:2]:
            logging.info(f"Feeding image: {file_path.name}")
            pipeline.process_image(file_path)
            # Simulating capture rate.
            time.sleep(5)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        
    finally:
        pipeline.stop()
        logging.info("Pipeline finished.")

if __name__ == "__main__":
    main()