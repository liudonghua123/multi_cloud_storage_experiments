import asyncio
import logging
import sys
import requests
import random
import time

logging.basicConfig(
    format="%(levelname)s %(asctime)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

urls = ["https://www.oschina.com", "https://www.baidu.com", "https://www.bing.com"]


async def request(taskId, delay=1):
    start = time.time()
    logger.info(f'task {taskId} start at {time.strftime("%X")}')
    response = requests.get(urls[taskId])
    # add some random delay to mock the post process time
    await asyncio.sleep(delay)
    logger.info(
        f'task {taskId} finished at {time.strftime("%X")} got {len(response.text)}'
    )
    return taskId, time.time() - start


async def one_round_test(weight=1):
    logger.info(f"start one_round_test with weight {weight}")
    results = [0, 0, 0]
    for task in asyncio.as_completed(
        [
            request(index, value)
            for index, value in enumerate(map(lambda x: x * weight, [10, 3, 5]))
        ]
    ):
        taskId, latency = await task
        logger.info(f"got task {taskId}, latency: {latency}")
        results[taskId] = latency
    logger.info(f"one_round_test results: {results}")
    return results


async def one_test():
    all_results = []
    for task in asyncio.as_completed([one_round_test(i * 0.5) for i in range(1, 4)]):
        results = await task
        all_results.append(results)
    return all_results


if __name__ == "__main__":
    start = time.perf_counter()
    all_results = asyncio.run(one_test(), debug=False)
    logger.info(f"all_results: {all_results}")
    elapsed = time.perf_counter() - start
    logger.info(f"{__file__} executed in {elapsed:0.2f} seconds.")
