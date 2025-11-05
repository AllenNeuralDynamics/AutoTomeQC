import logging
import sys

import autotomeqc.yolo.yolo_client as YOLOClient
from autotomeqc.config.config_loader import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def beep_boop(msg: str) -> str:
    return msg.replace('o', '0')    


def main() -> None:
    print(beep_boop("Howdy"))
    if not CONFIG:
        logger.error("Application cannot start without configuration. Exiting.")
        sys.exit(1)

    print("Loading YOLO Client with config:", CONFIG["blower"])
    #myyoloclient = YOLOClient({})


if __name__ == "__main__":
    main()
