import asyncio
import concurrent.futures
import requests
import time
import numpy as np

# async def main():

#     loop = asyncio.get_event_loop()
#     futures = [
#         loop.run_in_executor(None, requests.get, f"https://www.baidu.com/{i}")
#         for i in range(20)
#     ]
#     start_time = time.time()
#     for response in await asyncio.gather(*futures):
#         request_url = response.request.url
#         print(f"request {request_url} took {time.time() - start_time}, response: {response}")


# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())


# https://realpython.com/async-io-python/

# async def coro(job_id, seq) -> list:
#     """'IO' wait time is proportional to the max element."""
#     await asyncio.sleep(max(seq))
#     return job_id, list(reversed(seq))
  
# async def main():
#     t = asyncio.create_task(coro('job1', [3, 2, 1]))
#     t2 = asyncio.create_task(coro('job1', [10, 5, 0]))
#     print('Start:', time.strftime('%X'))
#     for res in asyncio.as_completed((t, t2)):
#         job_id, compl = await res
#         print(f'job_id: {job_id}, res: {compl} completed at {time.strftime("%X")}')
#     print('End:', time.strftime('%X'))
#     print(f'Both tasks done: {all((t.done(), t2.done()))}')

# asyncio.run(main())

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
with open(join(dirname(realpath(__file__)), "config.yml"), mode="r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

logger.info(f"load config: {config}")

storage_cost : list[float] = config['storage_cost']
read_cost : list[float] = config['read_cost']
write_cost : list[float] = config['write_cost']
cloud_providers : list[str] = config['cloud_providers']

class ASyncJOB:
      
    async def do_request(self, cloud_id) -> list:
        # make a request to cloud provider
        result = 'success'
        try:
            size = int(10240 / 2)
            if False:
                url = f"{cloud_providers[cloud_id]}/get?size={size}"
                logger.info(f"make read request url: {url}")
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    result = 'fail'
            else:
                url = f"{cloud_providers[cloud_id]}/put?size={size}"
                logger.info(f"make read request url: {url}")
                response = requests.put(url, files={"file": os.urandom(size)}, timeout=10)
                if response.status_code != 200:
                    result = 'fail'
        except Timeout:
            logger.error(f'The request {url} timed out')
            result = 'timeout'
        except ConnectionError as e:
            logger.error(f'The request {url} connection_error!')
            result = 'connection_error'
        except:
            logger.error(f'The request {url} error!')
            result = 'error'
        return cloud_id, result
        
    async def get_latency(self, clould_placements, tick):
        # make a parallel request to cloud providers which is enabled in clould_placements
        request_tasks = [asyncio.create_task(self.do_request(cloud_id)) for cloud_id, enabled in enumerate(clould_placements) if enabled == 1]
        logger.info(f"{tick} requests started at {time.strftime('%X')}")
        start_time = time.time()
        cloud_latency = np.full((6, ), np.nan)
        for task in asyncio.as_completed(request_tasks):
            cloud_id, result = await task
            if result != 'success':
                logger.error(f"request to cloud {cloud_id} failed")
            else:
                latency = time.time() - start_time
                cloud_latency[cloud_id] = latency
                print(f'request to cloud cloud_id: {cloud_id}, res: {result}, used {latency} seconds')
        logger.info(f"{tick} requests ended at {time.strftime('%X')}")
        return cloud_latency

    def run(self):
        result = asyncio.run(self.get_latency(np.array([0,0,1,0,1,0]), 1))
        print(result)

ASyncJOB().run()

