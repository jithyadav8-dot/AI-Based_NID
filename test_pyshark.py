import pyshark
import config

def test():
    print("Starting capture...")
    cap = pyshark.LiveCapture(interface=config.CAPTURE_INTERFACE)
    for packet in cap.sniff_continuously(packet_count=5):
        print(packet)
    print("Done")

if __name__ == "__main__":
    test()
