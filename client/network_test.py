import sys
import time
import os
from os.path import dirname, join, realpath

import math
import numpy as np

# ndarray for type hints
from numpy import ndarray
from numpy import nan
import pandas as pd
import schedule
import yaml
import random
import snoop
import asyncio
import requests
from datetime import datetime, date
from requests.exceptions import Timeout, ConnectionError

# In order to run this script directly, you need to add the parent directory to the sys.path
# Or you need to run this script in the parent directory using the command: python -m client.algorithm
sys.path.append(dirname(realpath(".")))
from common.utility import get_latency
from common.config_logging import init_logging

logger = init_logging(join(dirname(realpath(__file__)), "client.log"))

# read config.yml file using yaml
with open(
    join(dirname(realpath(__file__)), "config.yml"), mode="r", encoding="utf-8"
) as file:
    config = yaml.safe_load(file)

logger.info(f"load config: {config}")

storage_cost: list[float] = config["storage_cost"]
read_cost: list[float] = config["read_cost"]
write_cost: list[float] = config["write_cost"]
cloud_providers: list[str] = config["cloud_providers"]


class NetworkTest:
    def __init__(
        self, start_time, end_time, interval, read=True, N=6, data_size=1024, k=3, n=2
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.interval = interval
        self.read = read
        self.N = N
        self.data_size = data_size
        self.k = k
        self.n = n

    def run(self):
        # get the current datetime
        now = datetime.now()
        # create a dataframe to store the results
        # the columns are: timestamp, cloud_id_0, cloud_id_1, ..., cloud_id_N
        header = ["timestamp", *[f"cloud_id_{i}" for i in range(self.N)]]
        df = pd.DataFrame(columns=header)
        # run the test
        for tick in range(self.start_time, self.end_time, self.interval):
            logger.info(f"tick: {tick}")
            clould_placements = np.full((self.N,), 1)
            latency_cloud = asyncio.run(get_latency(clould_placements, tick, self.N, self.k, cloud_providers, self.data_size, self.read))
            logger.info(f"latency_cloud: {latency_cloud}")
            # save the result to df
            df.loc[len(df.index)] =[tick, *latency_cloud]
            # logging the last 5 rows of df
            logger.info(f'last 5 rows of df: \n{df.iloc[-5:]}')
            time.sleep(self.interval)
        # save the result to csv file suffix with the current timestamp
        csv_file_path = join(dirname(realpath(__file__)), f"network_test_with_datasize_{self.data_size}_{'read' if self.read else 'write'}_start_at_{now.strftime('%Y_%d_%m_%H_%M_%S')}.csv")
        logger.info(f'prepare to save the result to csv file: {csv_file_path}')
        df.to_csv(csv_file_path, index=False)
        logger.info(f'save the result to csv file successfully')


if __name__ == "__main__":
    # test the network in 3 mdfdfinutes
    network_test = NetworkTest(int(time.time()) + 10, int(time.time()) + 10 + 30, 10)
    network_test.run()
