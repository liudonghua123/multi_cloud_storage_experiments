import asyncio
import logging
import sys
import requests
import random
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

logging.basicConfig(
    format="%(levelname)s %(asctime)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

urls = ["https://www.oschina.com", "https://www.baidu.com", "https://www.bing.com"]


def request(taskId, delay=1):
    start = time.time()
    logger.info(f'task {taskId} start at {time.strftime("%X")}')
    response = requests.get(urls[taskId])
    # add some random delay to mock the post process time
    # await asyncio.sleep(delay)
    time.sleep(delay)
    logger.info(
        f'task {taskId} finished at {time.strftime("%X")} got {len(response.text)}'
    )
    return taskId, time.time() - start

def one_round_test(weight=1):
    logger.info(f"start one_round_test with weight {weight}")
    results = [0, 0, 0]
    with ThreadPoolExecutor(max_workers=3) as executor:
        return_values = executor.map(request, range(3), map(lambda x: x * weight, [10, 3, 5]))
        for taskId, latency in return_values:
            logger.info(f"got task {taskId}, latency: {latency}")
            results[taskId] = latency
    logger.info(f"one_round_test results: {results}")
    return results


def one_test():
    all_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        return_values = executor.map(one_round_test, map(lambda x: x * 0.5, range(1,4)))
        for results in return_values:
            all_results.append(results)
    return all_results


if __name__ == "__main__":
    start = time.perf_counter()
    all_results = one_test()
    logger.info(f"all_results: {all_results}")
    elapsed = time.perf_counter() - start
    logger.info(f"{__file__} executed in {elapsed:0.2f} seconds.")
