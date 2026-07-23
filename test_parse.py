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
                    print("proto:", packet.ip.proto)
                    print("length:", packet.length)
                    print("timestamp:", packet.sniff_timestamp)
            except Exception as e:
                print("Error inspecting packet:", repr(e))
            raise StopIteration("Stop after 1")

        await cap.packets_from_tshark(on_packet)
    except StopIteration:
        pass
    except Exception as e:
        print("Fatal Error:", repr(e))

if __name__ == "__main__":
    asyncio.run(main())
