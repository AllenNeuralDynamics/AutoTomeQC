import sys
import logging

from autotomeqc.core.autotomeService import AutoTomeService

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutoTomeMain")

def main():
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
    service.handle_start()
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
                input_path_str = user_input.strip('"').strip("'")
                if len(input_path_str) > 0:
                    service.handle_process_image(input_path_str)
            except Exception:
                logger.error("Invalid input or path error.")

    except KeyboardInterrupt:
        if is_interactive:
            logger.info("\nInterrupted by User.")

    finally:
        service.handle_stop()
        logger.info("Bye!")

if __name__ == "__main__":
    main()