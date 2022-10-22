import sys
import time
import os
from os.path import dirname, join, realpath

import math
import numpy as np

# ndarray for type hints
from numpy import ndarray
from numpy import nan
import schedule
import yaml
import random
import snoop
import asyncio
import requests
from requests.exceptions import Timeout, ConnectionError

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

    async def do_request(self, cloud_id) -> list:
        # make a request to cloud provider
        result = "success"
        try:
            size = int(self.data_size / self.k)
            if self.read:
                url = f"{cloud_providers[cloud_id]}/get?size={size}"
                logger.info(f"make read request url: {url}")
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    result = "fail"
            else:
                url = f"{cloud_providers[cloud_id]}/put?size={size}"
                logger.info(f"make read request url: {url}")
                response = requests.put(
                    url, files={"file": os.urandom(size)}, timeout=10
                )
                if response.status_code != 200:
                    result = "fail"
        except Timeout:
            logger.error(f"The request {url} timed out")
            result = "timeout"
        except ConnectionError as e:
            logger.error(f"The request {url} connection_error!")
            result = "connection_error"
        except:
            logger.error(f"The request {url} error!")
            result = "error"
        return cloud_id, result

    async def get_latency(self, clould_placements, tick):
        # make a parallel request to cloud providers which is enabled in clould_placements
        request_tasks = [
            asyncio.create_task(self.do_request(cloud_id))
            for cloud_id, enabled in enumerate(clould_placements)
            if enabled == 1
        ]
        logger.info(f"{tick} requests started at {time.strftime('%X')}")
        start_time = time.time()
        latency_cloud = np.zeros((self.N,))
        for task in asyncio.as_completed(request_tasks):
            cloud_id, result = await task
            if result != "success":
                logger.error(f"request to cloud {cloud_id} failed")
            else:
                latency = time.time() - start_time
                latency_cloud[cloud_id] = latency
                print(
                    f"request to cloud cloud_id: {cloud_id}, res: {result}, used {latency} seconds"
                )
        logger.info(f"{tick} requests ended at {time.strftime('%X')}")
        return latency_cloud

    def run(self):
        # run the test
        for tick in range(self.start_time, self.end_time, self.interval):
            print(f"tick: {tick}")
            clould_placements = np.full((self.N,), 1)
            latency_cloud = asyncio.run(self.get_latency(clould_placements, tick))
            print(f"latency_cloud: {latency_cloud}")
            time.sleep(self.interval)


if __name__ == "__main__":
    # test the network in 3 minutes
    network_test = NetworkTest(int(time.time()) + 10, int(time.time()) + 10 + 300, 30)
    network_test.run()
