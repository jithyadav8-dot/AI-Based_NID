import socket
import threading

TARGET = "127.0.0.1"

def scan_port(ip, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect((ip, port))
        s.close()
    except Exception:
        pass

if __name__ == "__main__":
    print(f"Starting quick scan against {TARGET}...")
    for port in range(1, 2001):
        t = threading.Thread(target=scan_port, args=(TARGET, port))
        t.start()
    print("Scan threads launched!")
