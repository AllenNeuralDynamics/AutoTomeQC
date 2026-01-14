import sys
import logging
from autotomeqc.config.cli import parse_args, print_arg_info
from autotomeqc.core.pipeline import AutoTomePipeline
from autotomeqc.utils.io import save_json_results, save_debug_image

# Configure logging once here; other modules will inherit it.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main() -> None:
    # Parse and Validate Arguments
    args = parse_args()
    print_arg_info(args)
    if not args.input_image_path.exists():
        logger.error(f"Input file not found: {args.input_image_path}")
        sys.exit(1)

    # Run pipeline
    pipeline = AutoTomePipeline()
    pipeline.start()

    try:
        logger.info(f"Processing image: {args.input_image_path.name}")
        # Returns: (metrics_dict, segmented_image_numpy_array)
        metrics, segmented_img = pipeline.process_image(args.input_image_path)

        # Save Results to Disk
        save_json_results(metrics, args.output_path)
        if args.save_segmented_image:
            save_debug_image(segmented_img, args.save_segmented_image)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        pipeline.stop()
        logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()