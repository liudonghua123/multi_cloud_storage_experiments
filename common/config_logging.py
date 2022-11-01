import logging
import sys
from os.path import dirname, join, realpath

# config logging for both stdout and file


def init_logging(file_name=join(dirname(realpath(__file__)), "app,log"), level=logging.INFO):
    logging.basicConfig(
        format="%(levelname)s %(asctime)s - %(message)s",
        level=level,
        handlers=[logging.FileHandler(file_name), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger()
    return logger