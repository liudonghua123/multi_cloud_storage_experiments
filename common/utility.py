import numpy as np
import asyncio
import requests
from requests.exceptions import Timeout, ConnectionError
import sys
import time
import os
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os.path import dirname, join, realpath
from common.config_logging import init_logging
from halo import Halo
import yaml
import random

sys.path.append(dirname(realpath(".")))
from common.config_logging import init_logging
logger = init_logging(join(dirname(realpath(__file__)), "common.log"))


# read config-default.yml, config.yml(optional, override the default configurations) using yaml
def get_config(dir_path):
    default_config_file = "config-default.yml"
    config_file = "config.yml"
    logger.info(f"Loading default config file: {config_file}")
    with open(join(dir_path, default_config_file), mode="r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if os.path.exists(join(dir_path, config_file)):
        logger.info(f"Loading override config file: {config_file}")
        with open(join(dir_path, config_file), mode="r", encoding="utf-8") as file:
            # use **kwargs to merge the two dictionaries
            config = {**config, **yaml.safe_load(file)}
    return config

class spinner_context:
    def __init__(self, start_text: str, end_text: str = None, spinner_indicator: str = 'dots'):
        self.start_text = start_text
        self.end_text = end_text or start_text
        self.spinner_indicator = spinner_indicator
        self.spinner = Halo(text=self.start_text, spinner=self.spinner_indicator)
        self.start_time = None
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.spinner.start()
        return self.spinner
    def __exit__(self, exc_type, exc_value, traceback):
        self.spinner.succeed(f'{self.end_text}, took {time.perf_counter() - self.start_time:.2f}s')


async def do_request(cloud_base_url, cloud_id, size, read) -> list:
    # make a request to cloud provider
    result = 'success'
    try:
        start_time = time.time()
        start_datetime = datetime.now()
        latency = 0
        if read:
            url = f"{cloud_base_url}/get?size={size}"
            logger.debug(f"do_request {url} make read request url: {url}")
            response = requests.get(url, timeout=10)
            latency = time.time() - start_time
            logger.debug(f"do_request {url} got read response, status_code: ${response.status_code}")
            if response.status_code != 200:
                result = 'fail'
        else:
            url = f"{cloud_base_url}/put?size={size}"
            logger.debug(f"do_request {url} make write request url: {url}")
            response = requests.put(url, files={"file": os.urandom(size)}, timeout=10)
            latency = time.time() - start_time
            logger.debug(f"do_request {url} got write response, status_code: ${response.status_code}")
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
    return cloud_id, latency, result, start_datetime

async def get_latency(clould_placements, tick, N, k, cloud_providers, data_size, read):
    # make a parallel request to cloud providers which is enabled in clould_placements
    request_tasks = [do_request(cloud_providers[cloud_id], cloud_id, int(data_size / k), read) for cloud_id, enabled in enumerate(clould_placements) if enabled == 1]
    logger.info(f"get_latency of {clould_placements}, {'read' if read else 'write'}, {data_size} started")
    latency_cloud = np.zeros((N, ))
    request_start_datetime = None
    for task in asyncio.as_completed(request_tasks):
        logger.debug(f"{tick} await task")
        cloud_id, latency, result, start_datetime = await task
        logger.debug(f"{tick} await task returned")
        # use the first request start time as the start time of the grouped requests
        if request_start_datetime is None:
            request_start_datetime = start_datetime
        logger.info(f"do_request of {cloud_providers[cloud_id]}, {'read' if read else 'write'} {data_size} finished, used {latency} seconds")
        if result != 'success':
            logger.error(f"request to cloud {cloud_id} failed")
        else:
            latency_cloud[cloud_id] = latency
    logger.info(f"get_latency of {clould_placements}, {'read' if read else 'write'}, {data_size} finished")
    return [request_start_datetime, *latency_cloud]


def do_request_sync(cloud_base_url, cloud_id, size, read) -> list:
    # make a request to cloud provider
    result = 'success'
    try:
        start_time = time.time()
        start_datetime = datetime.now()
        latency = 0
        if read:
            url = f"{cloud_base_url}/get?size={size}"
            logger.debug(f"do_request_sync {url} make read request url: {url}")
            response = requests.get(url, timeout=10)
            latency = time.time() - start_time
            logger.debug(f"do_request_sync {url} got read response, status_code: ${response.status_code}")
            if response.status_code != 200:
                result = 'fail'
        else:
            url = f"{cloud_base_url}/put?size={size}"
            logger.debug(f"do_request_sync {url} make write request url: {url}")
            response = requests.put(url, files={"file": os.urandom(size)}, timeout=10)
            latency = time.time() - start_time
            logger.debug(f"do_request_sync {url} got write response, status_code: ${response.status_code}")
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
    return cloud_id, latency, result, start_datetime

def get_latency_sync(clould_placements, tick, N, k, cloud_providers, data_size, read):
    # make a parallel request to cloud providers which is enabled in clould_placements
    logger.debug(f"get_latency_sync of {clould_placements}, {'read' if read else 'write'}, {data_size} started")
    latency_cloud = np.zeros((N, ))
    request_start_datetime = None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        cloud_ids = [cloud_id for cloud_id, enabled in enumerate(clould_placements) if enabled == 1]
        return_values = executor.map(do_request_sync, [cloud_providers[cloud_id] for cloud_id in cloud_ids], cloud_ids, [int(data_size / k)] * len(cloud_ids), [read] * len(cloud_ids))
        for cloud_id, latency, result, start_datetime in return_values:
            if request_start_datetime is None:
                request_start_datetime = start_datetime
            logger.debug(f"do_request_sync of {cloud_providers[cloud_id]}, {'read' if read else 'write'} {data_size} finished, used {latency} seconds")
            if result != 'success':
                logger.error(f"request to cloud {cloud_id} failed")
            else:
                latency_cloud[cloud_id] = latency
    logger.debug(f"get_latency_sync of {clould_placements}, {'read' if read else 'write'}, {data_size} finished")
    return [request_start_datetime, *latency_cloud]