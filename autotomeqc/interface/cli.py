#autotomeqc/interface/cli.py
import sys
import logging
from autotomeqc.core.autotome_service import AutoTomeService

logger = logging.getLogger(__name__)


def run_interactive_cli():
    service = AutoTomeService()

    # Detect Mode
    # isatty() returns True if connected to a terminal (Human), False if piped (Robot)
    is_interactive = sys.stdin.isatty()
    if is_interactive:  # Human Mode
        logger.info("==========================================")
        logger.info("   AutoTomeQC Interactive Service         ")
        logger.info("   Input: Paste file path to process      ")
        logger.info("   Commands: exit, stop (or Ctrl+C)       ")
        logger.info("==========================================")
        prompt_text = "\nReady > "
    else:
        # Machine Mode: No header, silent
        prompt_text = ""

    # Start
    service.start()
    try:
        while service.running:
            try:
                # If human: prints "Ready > " and waits.
                # If robot: prints nothing, just waits for line from pipe.
                if is_interactive:
                    user_input = input(prompt_text).strip()
                else:
                    user_input = sys.stdin.readline().strip()
                    # If readline returns empty string immediately, pipe is closed
                    if not user_input and len(user_input) == 0:
                        break
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "stop", "q"]:  # Quit Command
                break

            try:
                input_input_path = user_input.strip('"').strip("'")
                if len(input_input_path) > 0:
                    service.process(input_input_path)
            except Exception:
                logger.error("Invalid input or path error.")

    except KeyboardInterrupt:
        if is_interactive:
            logger.info("\nInterrupted by User.")

    finally:
        service.stop()
        logger.info("Bye!")

if __name__ == "__main__":
    run_interactive_cli()