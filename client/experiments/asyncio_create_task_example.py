import asyncio
import time


async def call_api(message, result=1000, delay=3):
    print(message)
    await asyncio.sleep(delay)
    return result


async def show_message():
    for _ in range(3):
        await asyncio.sleep(1)
    print("API call is in progress...")


async def main():
    start = time.perf_counter()
    message_task = asyncio.create_task(show_message())
    task_1 = asyncio.create_task(call_api("Get stock price of GOOG...", 300, delay=2))
    task_2 = asyncio.create_task(call_api("Get stock price of APPL...", 300, delay=1))
    print(await message_task)
    print(await task_1)
    print(await task_2)
    end = time.perf_counter()
    print(f"It took {round(end-start,0)} second(s) to complete.")


async def main_without_task():
    start = time.perf_counter()
    print(await show_message())
    print(await call_api("Get stock price of GOOG...", 300, delay=2))
    print(await call_api("Get stock price of APPL...", 300, delay=1))
    end = time.perf_counter()
    print(f"It took {round(end-start,0)} second(s) to complete.")


async def main_group():
    start = time.perf_counter()
    task_group = [
        asyncio.create_task(show_message()),
        asyncio.create_task(call_api("Get stock price of GOOG...", 300, delay=2)),
        asyncio.create_task(call_api("Get stock price of APPL...", 300, delay=1)),
    ]
    for task in asyncio.as_completed(task_group):
        result = await task
        print(f"got task result: {result}")
    end = time.perf_counter()
    print(f"It took {round(end-start,0)} second(s) to complete.")


async def main_group_without_task():
    start = time.perf_counter()
    task_group = [
        show_message(),
        call_api("Get stock price of GOOG...", 300, delay=2),
        call_api("Get stock price of APPL...", 300, delay=1),
    ]
    for task in asyncio.as_completed(task_group):
        result = await task
        print(f"got task result: {result}")
    end = time.perf_counter()
    print(f"It took {round(end-start,0)} second(s) to complete.")

async def test():
    await main()
    print('-' * 80)
    await main_without_task()
    print('-' * 80)
    await main_group()
    print('-' * 80)
    await main_group_without_task()

asyncio.run(test())