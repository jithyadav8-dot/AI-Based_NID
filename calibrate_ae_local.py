# calibrate_ae_local.py
# Runs for N minutes, collects AE reconstruction errors on flows
# that XGBoost classifies as BENIGN with high confidence,
# then resets the threshold to the local 95th percentile.

import asyncio
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from capture.pyshark_capture import PySharkCapture
from flow.window_assembler import WindowAssembler
from inference.predictor import NIDSPredictor

# ── Config ────────────────────────────────────────────────────────
import config
from config import CAPTURE_INTERFACE
INTERFACE = CAPTURE_INTERFACE
COLLECTION_MINUTES  = 1.0       # how long to collect benign samples
MIN_SAMPLES         = 10        # minimum before computing threshold
XGB_CONF_THRESHOLD  = 0.95      # only use flows XGBoost is very sure about

# ── State ─────────────────────────────────────────────────────────
predictor      = NIDSPredictor()
benign_errors  = []
collection_done = asyncio.Event()

async def on_window_ready(window: dict):
    xgb_class, xgb_conf, _ = predictor._xgb_predict(window["features"])
    ae_error, _             = predictor._ae_predict(window["features"])

    # Only collect errors where XGBoost is very confident it's BENIGN
    if xgb_class == "BENIGN" and xgb_conf >= XGB_CONF_THRESHOLD:
        benign_errors.append(ae_error)
        if len(benign_errors) % 50 == 0:
            print(f"  Collected {len(benign_errors)} benign samples  "
                  f"(current 95th pct: "
                  f"{np.percentile(benign_errors, 95):.6f})")

async def stop_after(minutes: float):
    print(f"Collecting for {minutes} minutes...")
    await asyncio.sleep(minutes * 60)
    collection_done.set()
    raise KeyboardInterrupt()

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
    print("AE Local Threshold Calibration")
    print("=" * 60)
    print(f"Current threshold : {predictor.ae_threshold:.6f}")
    print(f"Collection time   : {COLLECTION_MINUTES} minutes")
    print(f"XGBoost confidence: >= {XGB_CONF_THRESHOLD}")
    print("\nBrowse normally — generating representative benign traffic.")
    print("Ctrl+C to stop early (will compute from collected samples).\n")

    try:
        await asyncio.gather(
            capture.start(),
            assembler.run_stride_loop(),
            stop_after(COLLECTION_MINUTES),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    # ── Compute new threshold ─────────────────────────────────────
    if len(benign_errors) < MIN_SAMPLES:
        print(f"\nOnly {len(benign_errors)} samples collected "
              f"(minimum: {MIN_SAMPLES}).")
        print("Browse more actively and re-run, or lower MIN_SAMPLES.")
        return

    errors_arr    = np.array(benign_errors)

    # Cap extreme outliers before computing threshold.
    # Flows with AE error > 99th percentile are likely edge cases
    # (encrypted traffic, large transfers) that would push the
    # threshold too high for attack detection to work.
    p99           = float(np.percentile(errors_arr, 99))
    errors_capped = errors_arr[errors_arr <= p99]

    print(f"\n  Raw samples   : {len(errors_arr)}")
    print(f"  After cap (<=p99={p99:.4f}): {len(errors_capped)}")

    # Use 80th percentile on capped distribution.
    # 80th vs 95th tradeoff:
    #   95th → fewer false positives, may miss attacks
    #   80th → slightly more false positives, better attack detection
    # For a NIDS, missing attacks is the bigger risk.
    PERCENTILE    = 80
    old_threshold = predictor.ae_threshold
    new_threshold = float(np.percentile(errors_capped, PERCENTILE))

    print(f"\n{'='*60}")
    print(f"Calibration results ({len(errors_capped)} capped samples)")
    print(f"{'='*60}")
    print(f"  mean  : {errors_capped.mean():.6f}")
    print(f"  std   : {errors_capped.std():.6f}")
    print(f"  p80   : {new_threshold:.6f}  ← new threshold")
    print(f"  p95   : {np.percentile(errors_capped, 95):.6f}")
    print(f"  p99   : {p99:.6f}  (outlier cap)")
    print(f"\n  Old threshold : {old_threshold:.6f}")
    print(f"  New threshold : {new_threshold:.6f}")
    print(f"  Change        : {new_threshold - old_threshold:+.6f}")

    # ── Update threshold_ae.json ──────────────────────────────────
    with open("models/threshold_ae.json") as f:
        cfg = json.load(f)

    cfg["threshold_original_2018"]  = old_threshold
    cfg["threshold"]                = new_threshold
    cfg["threshold_source"]         = "local_calibration"
    cfg["local_calibration"] = {
        "n_samples":        len(errors_arr),
        "collection_minutes": COLLECTION_MINUTES,
        "xgb_conf_filter":  XGB_CONF_THRESHOLD,
        "percentile":       PERCENTILE,
        "calibrated_at":    datetime.now().isoformat(),
        "error_mean":       float(errors_arr.mean()),
        "error_std":        float(errors_arr.std()),
        "note": (
            "Threshold recalibrated on local network benign traffic. "
            "Only flows where XGBoost P(BENIGN) >= 0.95 were used "
            "to ensure high-confidence benign samples only."
        ),
    }

    with open("models/threshold_ae.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved: models/threshold_ae.json")
    print(f"\nRe-run run_phase6.py — MEDIUM false positives should drop "
          f"significantly.")


if __name__ == "__main__":
    asyncio.run(main())
