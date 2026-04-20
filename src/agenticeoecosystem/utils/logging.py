import logging
import sys
from datetime import datetime
from pathlib import Path

import ollama


def setup_logging(script_name: str) -> logging.Logger:
    logs_dir = Path(__file__).resolve().parents[3] / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{script_name}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
        force=True,
    )
    _log = logging.getLogger(__name__)
    _log.info("Logging to %s", log_file)
    return _log


def init_logging(log_file:str=None, verbose:bool=False)-> logging.Logger:
    import os
    # Determine the logging level
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Define the logging format
    formatter = "%(asctime)s : %(levelname)s : [%(filename)s:%(lineno)s - %(funcName)s()] : %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # Setup basic configuration for logging
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.FileHandler(log_file, "w"),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.StreamHandler()
            ]
        )

    logger = logging.getLogger()
    return logger


