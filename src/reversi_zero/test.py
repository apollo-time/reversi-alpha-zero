import asyncio
import time
async def func1():
    for i in range(50):
        print('F')
async def func2():
    for i in range(50):
        print('A')

coroutine_list = []
coroutine_list.append(func1())
coroutine_list.append(func2())
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(*coroutine_list))
print('end')
