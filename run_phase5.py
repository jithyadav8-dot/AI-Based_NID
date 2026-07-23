# run_phase5.py
# Tests the full capture -> flow assembly -> feature extraction pipeline.
# Run this to verify feature vectors appear for live traffic.

import asyncio
import numpy as np
from capture.pyshark_capture import PySharkCapture
from flow.window_assembler import WindowAssembler

import config
from config import CAPTURE_INTERFACE
INTERFACE = CAPTURE_INTERFACE

# ── Window callback ───────────────────────────────────────────────
window_count = 0

async def on_window_ready(window: dict):
    global window_count
    window_count += 1

    features = window["features"]
    src      = f"{window['src_ip']}:{window['src_port']}"
    dst      = f"{window['dst_ip']}:{window['dst_port']}"
    proto    = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(
        window["protocol"], "?"
    )

    print(f"\n[Window #{window_count}]")
    print(f"  Flow     : {src} -> {dst} ({proto})")
    print(f"  Packets  : {window['n_packets']} in last 10s")
    print(f"  Features : shape={features.shape}  "
          f"min={features.min():.4f}  max={features.max():.6f}")
    print(f"  Vector   : {features[:8]}...")  # first 8 values

    # Sanity checks
    assert features.shape[0] == 42,    f"Expected 42 features, got {features.shape[0]}"
    assert not np.isnan(features).any(), "NaN in feature vector"
    assert not np.isinf(features).any(), "Inf in feature vector"
    print(f"  Checks   : shape [OK]  no NaN [OK]  no Inf [OK]")

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
    print("PHASE 5 - Live Capture + Feature Extraction")
    print("=" * 60)
    print(f"Interface : {INTERFACE}")
    print(f"Window    : {config.WINDOW_SIZE}s  Stride: {config.STRIDE}s")
    print(f"Waiting for traffic... (generate some with a browser or ping)")
    print("=" * 60 + "\n")

    # Run capture and stride loop concurrently
    await asyncio.gather(
        capture.start(),
        assembler.run_stride_loop(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
