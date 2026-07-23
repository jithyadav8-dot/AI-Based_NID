import asyncio
from capture.pyshark_capture import PySharkCapture

async def main():
    cap = PySharkCapture("4", "ip")
    print("Starting cap...")
    task = asyncio.create_task(cap.start())
    await asyncio.sleep(5)
    print("Stats:", cap.stats)

if __name__ == "__main__":
    asyncio.run(main())
