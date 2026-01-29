#autotomeqc/interface/cli.py
import sys
import logging
import time
import json
import autotomeqc
from autotomeqc.core.autotome_service import AutoTomeService


logger = logging.getLogger(__name__)

ASCII_ART = r"""
    _         _      _____                  ___   ____  
   / \  _   _| |_ __|_   _|__  _ __ ___    / _ \ / ___| 
  / _ \| | | | __/ _ \| |/ _ \| '_ ` _ \  | | | | |     
 / ___ \ |_| | || (_) | | (_) | | | | | | | |_| | |___  
/_/   \_\__,_|\__\___/|_|\___/|_| |_| |_|  \__\_\\____| 
"""

def print_arg_info(args):
    """Print configuration info."""
    logger.info(ASCII_ART)
    logger.info(f"version: {autotomeqc.__version__}")

def run_interactive_cli():
    service = AutoTomeService()

    # Detect Mode
    # isatty() returns True if connected to a terminal (Human), False if piped (Robot)
    is_interactive = sys.stdin.isatty()
    if is_interactive:  # Human Mode
        print_arg_info(None)
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
    if not service.start():
        logger.error("Failed to start service.")
        return

    # Process Loop
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
            except (EOFError):
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "stop", "q"]:  # Quit Command
                break

            try:
                input_path = user_input.strip('"').strip("'")
                if len(input_path) > 0:
                    future = service.process(input_path)
                    if is_interactive:
                        print("Processing...", end="", flush=True)
                        while not future.done():
                            time.sleep(0.1)
                        print(" Done.")
                    else:
                        # Robot mode: just wait efficiently
                        future.result()

                    result = future.result()
                    status = result.get("qc_summary", "UNKNOWN")
                    if status == "PASS":
                        logger.info(f"✅ PASS: {status}")
                    else:
                        top_level_error = result.get("error_reason", result.get("error"))
                        if top_level_error:
                            reason = top_level_error
                        else:
                            # Check for specific QC criteria failures
                            failed_criteria = []
                            criteria = result.get("criteria", {})
                            for name, data in criteria.items():
                                if data.get("pass") is False:
                                    msg = data.get("reason", data.get("label", "Failed"))
                                    failed_criteria.append(f"{name} ({msg})")
                            reason = ", ".join(failed_criteria) if failed_criteria else "Unknown Failure"
                        logger.info(f"❌ FAIL: {reason}")
                    logger.info(f"   Details:\n{json.dumps(result.get('criteria', {}), indent=4, default=str)}")
            except Exception as e:
                logger.error(f"Invalid input or path error: {e}")

    except KeyboardInterrupt:
        if is_interactive:
            logger.info("\nInterrupted by User.")

    finally:
        service.stop()
        logger.info("Bye!")

if __name__ == "__main__":
    run_interactive_cli()