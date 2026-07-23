import asyncio
import pyshark
import config

async def main():
    cap = pyshark.LiveCapture(interface=config.CAPTURE_INTERFACE, bpf_filter=config.BPF_FILTER)
    try:
        def on_packet(packet):
            print("Packet received!")
            try:
                print("has_ip:", hasattr(packet, "ip"))
                if hasattr(packet, "ip"):
                    print("ip attributes:", dir(packet.ip))
                    print("ihl:", hasattr(packet.ip, "ihl"))
                    print("hdr_len:", hasattr(packet.ip, "hdr_len"))
            except Exception as e:
                print("Error inspecting packet:", repr(e))
            raise ValueError("stop")

        await cap.packets_from_tshark(on_packet)
    except Exception as e:
        print("Fatal Error:", repr(e))

if __name__ == "__main__":
    asyncio.run(main())
