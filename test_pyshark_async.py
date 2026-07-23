import asyncio
import pyshark
import config

async def main():
    print("Starting capture...")
    cap = pyshark.LiveCapture(interface=config.CAPTURE_INTERFACE, bpf_filter=config.BPF_FILTER)
    # pyshark async generator
    try:
        async for packet in cap.sniff_continuously():
            print("Packet:", packet)
            break
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
