import numpy as np
import asyncio
import requests
from requests.exceptions import Timeout, ConnectionError
import sys
import time
import os
from os.path import dirname, join, realpath
from common.config_logging import init_logging

sys.path.append(dirname(realpath(".")))
from common.config_logging import init_logging
logger = init_logging(join(dirname(realpath(__file__)), "common.log"))


async def do_request(cloud_base_url, cloud_id, size, read) -> list:
    # make a request to cloud provider
    result = 'success'
    try:
        if read:
            url = f"{cloud_base_url}/get?size={size}"
            logger.info(f"cloud_id: {cloud_id} make read request url: {url}")
            response = requests.get(url, timeout=10)
            logger.info(f"cloud_id: {cloud_id} got read response for url: {url}")
            if response.status_code != 200:
                result = 'fail'
        else:
            url = f"{cloud_base_url}/put?size={size}"
            logger.info(f"cloud_id: {cloud_id} make write request url: {url}")
            response = requests.put(url, files={"file": os.urandom(size)}, timeout=10)
            logger.info(f"cloud_id: {cloud_id} got write response for url: {url}")
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

async def get_latency(clould_placements, tick, N, k, cloud_providers, data_size, read):
    # make a parallel request to cloud providers which is enabled in clould_placements
    request_tasks = [do_request(cloud_providers[cloud_id], cloud_id, int(data_size / k), read) for cloud_id, enabled in enumerate(clould_placements) if enabled == 1]
    logger.info(f"{tick} requests started at {time.strftime('%X')}")
    start_time = time.time()
    latency_cloud = np.zeros((N, ))
    for task in asyncio.as_completed(request_tasks):
        logger.info(f"{tick} await task")
        cloud_id, result = await task
        logger.info(f"{tick} await task returned")
        logger.info(f'cloud_id: {cloud_id} got response!')
        if result != 'success':
            logger.error(f"request to cloud {cloud_id} failed")
        else:
            latency = time.time() - start_time
            latency_cloud[cloud_id] = latency
            logger.info(f'request to cloud cloud_id: {cloud_id}, res: {result}, used {latency} seconds')
    logger.info(f"{tick} requests ended at {time.strftime('%X')}")
    return latency_cloud
