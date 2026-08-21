"""
debug_live_score.py
Patches NIDSPredictor.score() to print every window's raw scores,
so we can see what AE errors / XGB classes are actually coming out.
Run while Phase 7 is running in another terminal.
"""
import json
import urllib.request
from collections import Counter

BASE = "http://localhost:8000"

# Pull stats to see current state
with urllib.request.urlopen(f"{BASE}/stats") as r:
    stats = json.loads(r.read())

with urllib.request.urlopen(f"{BASE}/status") as r:
    status = json.loads(r.read())

print(f"\n{'='*60}")
print(f"LIVE SYSTEM STATE")
print(f"{'='*60}")
print(f"  Status       : {status['status']}")
print(f"  Interface    : {status['interface']}")
print(f"  AE threshold : {status['ae_threshold']}")
print(f"  Inf threshold: {status['inf_threshold']}")
print(f"  Features     : {status['feature_count']}")
print(f"  Classes      : {status['model_classes']}")
print()
print(f"  Windows scored : {stats['windows_scored']:,}")
print(f"  Alerts raised  : {stats['alerts_raised']:,}")
print(f"  Latency        : {stats['avg_latency_ms']}ms")
print(f"  Active flows   : {stats['active_flows']}")
print(f"  Alert counts   : {stats['alert_counts']}")
print(f"  Capture stats  : {stats['capture_stats']}")

# ── Now locally re-score with LOWERED threshold ──────────────────
print(f"\n{'='*60}")
print(f"THRESHOLD ANALYSIS")
print(f"{'='*60}")
print(f"""
The AE threshold is {status['ae_threshold']}.
The system is seeing {stats['windows_scored']} windows but 0 alerts.

This means ALL windows have:
  - XGBoost predicting BENIGN
  - AE error <= {status['ae_threshold']}

Possible causes:
  1. quick_scan traffic hits interface {status['interface']} (Wi-Fi)
     but the scan target ({{}}) may route differently
  2. The AE threshold 3.5 is too tight — scan traffic AE error < 3.5
  3. The window assembler isn't grouping scan SYN packets into
     a flow that matches the training data distribution
""")

# ── Pull a raw debug score by calling the predictor directly ─────
print(f"{'='*60}")
print(f"SUGGESTED FIXES")
print(f"{'='*60}")
print("""
Option A: Lower the AE threshold temporarily
  - Edit models/threshold_ae.json: change "threshold": 3.5 to 2.0
  - Restart Phase 7

Option B: Scan localhost (loopback) instead of external IP
  - Edit quick_scan.py: TARGET = "127.0.0.1"
  - The loopback interface is usually tshark interface 9
  - Change config.py: CAPTURE_INTERFACE = "9"
  - Restart Phase 7 then scan

Option C: Run simulate_scan.py (if it injects directly into the pipeline)
  - py simulate_scan.py

Option D: Use the current AE threshold — lower it in threshold_ae.json
""")
