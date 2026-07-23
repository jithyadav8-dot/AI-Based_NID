import asyncio
import pyshark
import config

async def main():
    print("Starting capture...")
    cap = pyshark.LiveCapture(interface=config.CAPTURE_INTERFACE, bpf_filter=config.BPF_FILTER)
    try:
        def on_packet(pkt):
            print("Packet:", pkt)
        
        await cap.packets_from_tshark(on_packet)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
