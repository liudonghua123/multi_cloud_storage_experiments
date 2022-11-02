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
from threading import Thread
# import datetime
from datetime import datetime, date, timedelta
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
intermediate_save_seconds: int = config["network_test"]["intermediate_save_seconds"]

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
            # sleep for 1 second for reaching the start datetime accurately
            await asyncio.sleep(1)   
        # name csv file suffix with some configurations and current timestamp
        csv_file_path = join(dirname(realpath(__file__)), f"network_test_with_placements_{'_'.join(map(str, self.clould_placement))}_datasize_{self.data_size}_{'read' if self.read else 'write'}_start_at_{initial_datetime.strftime('%Y_%d_%m_%H_%M_%S')}.csv")
        csv_saved_count = 1
        # start the test until the end datetime
        while datetime.now() < self.end_datetime:
            tick = datetime.now()
            logger.info(f"tick: {tick}")
            latency_cloud = await get_latency(self.clould_placement, tick, self.N, self.k, cloud_providers, self.data_size, self.read)
            logger.info(f"datasize of {self.data_size} {'read' if self.read else 'write'} latency_cloud: {latency_cloud}")
            # save the result to df
            df.loc[len(df.index)] = latency_cloud
            # logging the last 5 rows of df
            logger.info(f"datasize of {self.data_size} {'read' if self.read else 'write'} last 5 rows of df: \n{df.iloc[-5:]}")
            while datetime.now() < tick + timedelta(seconds=self.interval):
                await asyncio.sleep(0.1)
            # save the df to csv file every intermediate_save_seconds in another thread
            if (datetime.now() - start_datetime).seconds >= csv_saved_count * intermediate_save_seconds:
                def _save_csv(df):
                    logger.info(f"prepare to save df to csv file intermediately: {csv_file_path}")
                    df.to_csv(csv_file_path, index=False)
                    logger.info(f"saved df to csv file: {csv_file_path}")
                t = Thread(target = _save_csv, args =(df, ))
                t.start() 
                csv_saved_count += 1
        # save the df to csv file, overwrite the previous one
        logger.info(f'prepare to save the result to csv file finally: {csv_file_path}')
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
    # results = await asyncio.gather(*test_tasks)
    results = []
    for task in asyncio.as_completed(test_tasks):
        logger.info(f"before await task")
        result = await task
        logger.info(f"after await task, result: {result}")
        results.append(result)
    # logger.info(f'test tasks results: {results}')
    logger.info(f"test tasks ended at {time.strftime('%X')}")

if __name__ == "__main__":
    # test the initial test matrix
    # network_test = NetworkTest(start_datetime, end_datetime, interval, reads[0], N, cloud_placements[0], data_sizes[0], k, n)
    # asyncio.run(network_test.run())
    
    # run the test matrix parallelly
    asyncio.run(run_test())
    
