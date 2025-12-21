import asyncio


async def get_burgers(number: int):
    for i in range(number):
        await asyncio.sleep(1)
    return number


