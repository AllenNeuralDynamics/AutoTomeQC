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
    __version__ = "0.1.0"

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
        description="AutoTomeQC: Analyze mouse brain section images.",
        prog="uv run python -m autotomeqc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text
    )
    # TODO Add future CLI args here
    return parser.parse_args()


def print_arg_info(args):
    """Print configuration info."""
    print(ASCII_ART)
    print(f"Mode:             INTERACTIVE SERVICE")
    print(f"Status:           Waiting for input on Stdin...")


if __name__ == "__main__":
    args = parse_args()
    print_arg_info(args)