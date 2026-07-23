# run_phase6.py
# Full pipeline: capture → features → inference → alerts

import asyncio
import json
from datetime import datetime
from collections import defaultdict

from capture.pyshark_capture import PySharkCapture
from flow.window_assembler import WindowAssembler
from inference.predictor import NIDSPredictor

import config
from config import CAPTURE_INTERFACE
INTERFACE = CAPTURE_INTERFACE

# ── Init models ───────────────────────────────────────────────────
predictor = NIDSPredictor()

# ── Alert handler ─────────────────────────────────────────────────
alert_counts = defaultdict(int)

async def on_window_ready(window: dict):
    alert = predictor.score(window)

    alert_counts[alert.level] += 1

    # Only print non-NORMAL alerts to keep output readable
    # Change to `if True:` to see all windows
    if alert.level != "NORMAL":
        ts  = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
        src = f"{alert.src_ip}:{alert.src_port}"
        dst = f"{alert.dst_ip}:{alert.dst_port}"

        print(f"\n{'='*60}")
        print(f"[{ts}]  {alert.level:<10}  severity={alert.severity:.1f}")
        print(f"  Flow     : {src} → {dst} ({alert.protocol})")
        print(f"  Label    : {alert.label}")
        print(f"  XGBoost  : {alert.xgb_class} "
              f"(conf={alert.xgb_confidence:.3f})")
        print(f"  AE error : {alert.ae_error:.6f} "
              f"(threshold={alert.ae_threshold:.6f}  "
              f"anomaly={alert.ae_is_anomaly})")

# ── Stats printer ─────────────────────────────────────────────────
async def print_stats():
    while True:
        await asyncio.sleep(10)
        stats = predictor.stats
        print(
            f"\n[Stats] scored={stats['windows_scored']}  "
            f"alerts={stats['alerts_raised']}  "
            f"latency={stats['avg_latency_ms']}ms  "
            f"| {dict(alert_counts)}"
        )

# ── Main ──────────────────────────────────────────────────────────
async def main():
    assembler = WindowAssembler(
        feature_list_path = "processed/feature_list.json",
        window_size       = config.WINDOW_SIZE,
        stride            = config.STRIDE,
        on_window_ready   = on_window_ready,
    )

    capture = PySharkCapture(
        iface           = INTERFACE,
        bpf_filter      = config.BPF_FILTER,
        packet_callback = assembler.update,
    )

    print("=" * 60)
    print("PHASE 6 — Real-Time Inference")
    print("=" * 60)
    print("Showing non-NORMAL alerts only.")
    print("Generate attack traffic to test:")
    print("  nmap -sS <target>           → BruteForce/PortScan-like")
    print("  hping3 -S --flood -p 80 <t> → DoS-like")
    print("Press Ctrl+C to stop.\n")

    try:
        await asyncio.gather(
            capture.start(),
            assembler.run_stride_loop(),
            print_stats(),
        )
    except asyncio.CancelledError:
        pass
    finally:
        await capture.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nFinal stats:", predictor.stats)
        print("Alert distribution:", dict(alert_counts))
