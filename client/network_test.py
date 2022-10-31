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
# import datetime
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

cloud_placements: list[list[int]] = config["network_test"]["matrix"]["cloud_placements"]
data_sizes: list[int] = config["network_test"]["matrix"]["data_sizes"]
reads: list[bool] = config["network_test"]["matrix"]["reads"]
start_datetime: datetime = config["network_test"]["start_datetime"]
end_datetime: datetime = config["network_test"]["end_datetime"]
interval: int = config["network_test"]["interval"]
N: int = config["network_test"]["N"]
k: int = config["network_test"]["k"]
n: int = config["network_test"]["n"]

class NetworkTest:
    def __init__(
        self, start_datetime, end_datetime, interval, read=True, N=6, clould_placement=[1,1,1,1,1,1], data_size=1024, k=3, n=2
    ):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.interval = interval
        self.read = read
        self.N = N
        self.clould_placement = clould_placement
        self.data_size = data_size
        self.k = k
        self.n = n

    async def run(self):
        # get the current datetime
        initial_datetime = datetime.now()
        # create a dataframe to store the results
        # the columns are: timestamp, cloud_id_0, cloud_id_1, ..., cloud_id_N
        header = ["timestamp", *[f"cloud_id_{i}" for i in range(self.N)]]
        df = pd.DataFrame(columns=header)
        
        # check if the current datetime exceeds the end datetime
        if datetime.now() > self.end_datetime:
            logger.info("The current datetime exceeds the end datetime.")
            return
        # wait for the start datetime
        while datetime.now() < self.start_datetime:
            logger.info(f"wait for the start datetime: {self.start_datetime}")
            await asyncio.sleep(self.interval / 2)
        # start the test until the end datetime
        while datetime.now() < self.end_datetime:
            tick = datetime.now()
            logger.info(f"tick: {tick}")
            latency_cloud = await get_latency(self.clould_placement, tick, self.N, self.k, cloud_providers, self.data_size, self.read)
            logger.info(f"latency_cloud: {latency_cloud}")
            # save the result to df
            df.loc[len(df.index)] =[tick, *latency_cloud]
            # logging the last 5 rows of df
            logger.info(f'last 5 rows of df: \n{df.iloc[-5:]}')
            await asyncio.sleep(self.interval)
        # save the result to csv file suffix with the current timestamp
        csv_file_path = join(dirname(realpath(__file__)), f"network_test_with_placements_{'_'.join(map(str, self.clould_placement))}_datasize_{self.data_size}_{'read' if self.read else 'write'}_start_at_{initial_datetime.strftime('%Y_%d_%m_%H_%M_%S')}.csv")
        logger.info(f'prepare to save the result to csv file: {csv_file_path}')
        df.to_csv(csv_file_path, index=False)
        logger.info(f'save the result to csv file successfully')
        return df


async def run_test():
    # create a list of NetworkTest based on the test matrix
    network_tests: list[NetworkTest] = []
    for cloud_placement in cloud_placements:
        for data_size in data_sizes:
            for read in reads:
                network_tests.append(NetworkTest(start_datetime, end_datetime, interval, read, N, cloud_placement, data_size, k, n))
    logger.info(f'create {len(network_tests)} NetworkTest instances')
    test_tasks = [asyncio.create_task(network_test.run()) for network_test in network_tests]
    logger.info(f"test tasks started at {time.strftime('%X')}")
    results = await asyncio.gather(*test_tasks)
    logger.info(f'test tasks results: {results}')
    logger.info(f"test tasks ended at {time.strftime('%X')}")

if __name__ == "__main__":
    # test the initial test matrix
    # network_test = NetworkTest(start_datetime, end_datetime, interval, reads[0], N, cloud_placements[0], data_sizes[0], k, n)
    # asyncio.run(network_test.run())
    
    # run the test matrix parallelly
    asyncio.run(run_test())
    
