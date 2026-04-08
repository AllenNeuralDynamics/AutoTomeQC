# autotomeqc/__main__.py
import logging
from autotomeqc.interface.cli import run_interactive_cli

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_interactive_cli()

if __name__ == "__main__":
    main()