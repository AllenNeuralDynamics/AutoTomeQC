"""
CLI Configuration
"""

import argparse
import sys
from pathlib import Path

# Try to import version
try:
    from autotomeqc import __version__
except ImportError:
    __version__ = "0.2.0"

ASCII_ART = r"""
    _         _      _____                  ___   ____  
   / \  _   _| |_ __|_   _|__  _ __ ___    / _ \ / ___| 
  / _ \| | | | __/ _ \| |/ _ \| '_ ` _ \  | | | | |     
 / ___ \ |_| | || (_) | | (_) | | | | | | | |_| | |___  
/_/   \_\__,_|\__\___/|_|\___/|_| |_| |_|  \__\_\\____| 
"""

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the AutoTomeQC application."""
    epilog_text = f"v{__version__}\n{ASCII_ART}\n"
    parser = argparse.ArgumentParser(
        description="AutoTomeQC: Analyze mouse brain section images and export QC metrics.",
        prog="uv run python -m autotomeqc",
        usage="%(prog)s [options] input_image output_json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text
    )

    # --- Main Inputs ---
    parser.add_argument(
        "input_image_path",
        type=Path,
        metavar="input_image",
        help="Path to the input image file (e.g., .png, .jpg, .tif).",
    )

    parser.add_argument(
        "output_path",
        type=Path,
        metavar="output_json",
        help="Path where the output JSON QC results will be saved.",
    )

    # --- Optional Outputs ---
    parser.add_argument(
        "--save_segmented_image",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional path to save the debug segmented section image (e.g., ./output/seg.png).",
    )
    return parser.parse_args()


def print_arg_info(args):
    print(ASCII_ART)
    print(f"Processing Image: {args.input_image_path}")
    print(f"Output Target:    {args.output_path}")
    
    if args.save_segmented_image:
        print(f"Save Debug Img:   {args.save_segmented_image}")


if __name__ == "__main__":
    args = parse_args()
    print_arg_info(args)
    
    if not args.input_image_path.exists():
        print(f"\nError: Input file '{args.input_image_path}' does not exist.")
        sys.exit(1)
        
    print(f"\nReady to process '{args.input_image_path.name}' -> JSON... {args.output_path}")