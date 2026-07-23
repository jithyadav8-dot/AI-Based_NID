import asyncio
import numpy as np
from capture.pyshark_capture import PySharkCapture
from flow.window_assembler import WindowAssembler
from inference.predictor import NIDSPredictor

predictor = NIDSPredictor()
errors = []

async def on_window_ready(window: dict):
    xgb_class, xgb_conf, _ = predictor._xgb_predict(window["features"])
    ae_error, _             = predictor._ae_predict(
        window["features"],
        protocol = window["protocol"],
        dst_port = window["dst_port"],
    )
    errors.append((xgb_class, xgb_conf, ae_error,
                   window["src_ip"], window["dst_ip"]))

    # Print every window so we see what's happening
    print(f"  {xgb_class:<12} conf={xgb_conf:.3f}  "
          f"ae_err={ae_error:.4f}  "
          f"{window['src_ip']} → {window['dst_ip']}")

async def stop():
    await asyncio.sleep(30)   # run for 30 seconds

import config

async def main():
    assembler = WindowAssembler(
        feature_list_path = "processed/feature_list.json",
        window_size       = config.WINDOW_SIZE,
        stride            = config.STRIDE,
        on_window_ready   = on_window_ready,
    )
    capture = PySharkCapture(
        iface           = config.CAPTURE_INTERFACE,
        bpf_filter      = config.BPF_FILTER,
        packet_callback = assembler.update,
    )

    print("Watching all windows for 30s — run nmap now\n")
    print(f"{'XGBoost':<12} {'Conf':>7}  {'AE Error':>10}  Flow")
    print("-" * 60)

    try:
        await asyncio.wait_for(
            asyncio.gather(capture.start(), assembler.run_stride_loop()),
            timeout=30
        )
    except (asyncio.TimeoutError, KeyboardInterrupt):
        pass
    finally:
        await capture.stop()

    if errors:
        all_errors = [e[2] for e in errors]
        attack_errors = [e[2] for e in errors if e[0] != "BENIGN"]
        benign_errors = [e[2] for e in errors if e[0] == "BENIGN"]

        print(f"\n{'='*50}")
        print(f"All windows   — p95 error: {np.percentile(all_errors, 95):.4f}")
        if benign_errors:
            print(f"BENIGN windows— p95 error: "
                  f"{np.percentile(benign_errors, 95):.4f}")
        if attack_errors:
            print(f"Attack windows— mean error: "
                  f"{np.mean(attack_errors):.4f}")
        print(f"\nCurrent threshold: {predictor.ae_threshold:.6f}")
        
        if benign_errors:
            print(f"Suggested threshold: "
                  f"{np.percentile(benign_errors, 90):.4f} "
                  f"(90th pct of benign)")

if __name__ == "__main__":
    asyncio.run(main())
