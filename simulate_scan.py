import socket
import threading

def scan_port(ip, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect((ip, port))
        s.close()
    except Exception:
        pass

# Rapidly scan 1000 ports to trigger the NIDS
for port in range(1, 1001):
    t = threading.Thread(target=scan_port, args=("127.0.0.1", port))
    t.start()
