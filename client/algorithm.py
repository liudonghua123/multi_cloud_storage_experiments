import sys
import time
from os.path import dirname, join, realpath

import numpy as np
import schedule
import yaml

# import pandas as pd
# import matplotlib.pyplot as plt


# In order to run this script directly, you need to add the parent directory to the sys.path
# Or you need to run this script in the parent directory using the command: python -m client.algorithm
sys.path.append(dirname(realpath(".")))
from common.config_logging import init_logging

logger = init_logging(join(dirname(realpath(__file__)), "client.log"))

# read config.yml file using yaml
with open(
    join(dirname(realpath(__file__)), "config.yml"), mode="r", encoding="utf-8"
) as file:
    config = yaml.safe_load(file)

logger.info(f"load config: {config}")


def get_data():
    logger.info("get_data")
    time.sleep(1)
    logger.info("get_data end")


def run():
    schedule.every(10).seconds.do(get_data)

    while True:
        logger.info("schedule.run_pending")
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    run()
