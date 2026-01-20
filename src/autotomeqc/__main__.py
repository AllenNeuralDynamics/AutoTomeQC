# autotomeqc/__main__.py
import logging
from autotomeqc.interface.cli import run_interactive_cli

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutoTomeMain")

if __name__ == "__main__":
    run_interactive_cli()