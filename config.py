# config.py
# Single place to change runtime settings.
# Every script imports from here — no more hardcoded interface numbers.

CAPTURE_INTERFACE  = "5"       # ← Wi-Fi interface
BPF_FILTER         = "ip"
WINDOW_SIZE        = 10.0      # seconds
STRIDE             = 2.0       # seconds
AE_BYPASS_UDP      = {443, 4500, 51820}   # QUIC, WireGuard, IPSec
AE_BYPASS_TCP      = {53}                 # DNS-over-TCP
API_HOST           = "0.0.0.0"
API_PORT           = 8000
MAX_ALERT_STORE    = 1000      # alerts kept in memory
