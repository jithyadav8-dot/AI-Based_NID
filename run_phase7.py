# run_phase7.py
# Starts the full NIDS system:
#   capture → flow assembly → inference → FastAPI API + WebSocket stream

import asyncio
import uvicorn

from config import (
    CAPTURE_INTERFACE, BPF_FILTER,
    WINDOW_SIZE, STRIDE,
    API_HOST, API_PORT,
)

from capture.pyshark_capture import PySharkCapture
from flow.window_assembler   import WindowAssembler
from inference.predictor     import NIDSPredictor
from api.main                import app, on_alert, pipeline_state


# ── Pipeline setup ────────────────────────────────────────────────

def build_pipeline():
    """Creates all pipeline components and wires them together."""
    predictor = NIDSPredictor()

    async def on_window_ready(window: dict):
        alert = predictor.score(window)
        await on_alert(alert)    # store + broadcast to WebSocket clients

    assembler = WindowAssembler(
        feature_list_path = "processed/feature_list.json",
        window_size       = WINDOW_SIZE,
        stride            = STRIDE,
        on_window_ready   = on_window_ready,
    )

    capture = PySharkCapture(
        iface           = CAPTURE_INTERFACE,
        bpf_filter      = BPF_FILTER,
        packet_callback = assembler.update,
    )

    return predictor, assembler, capture


# ── FastAPI startup / shutdown ────────────────────────────────────

@app.on_event("startup")
async def startup():
    import time
    print(f"[API] Starting pipeline on interface {CAPTURE_INTERFACE}...")

    predictor, assembler, capture = build_pipeline()

    # Inject into shared state so endpoints can read stats
    pipeline_state["predictor"]  = predictor
    pipeline_state["assembler"]  = assembler
    pipeline_state["capture"]    = capture
    pipeline_state["start_time"] = time.time()
    pipeline_state["status"]     = "running"

    # Run capture and stride loop as background tasks
    asyncio.create_task(capture.start(),           name="capture")
    asyncio.create_task(assembler.run_stride_loop(), name="stride_loop")

    print(f"[API] Pipeline running.")
    print(f"[API] Docs at http://{API_HOST}:{API_PORT}/docs")
    print(f"[API] Alerts at http://{API_HOST}:{API_PORT}/alerts")
    print(f"[API] WebSocket at ws://{API_HOST}:{API_PORT}/ws/alerts")


@app.on_event("shutdown")
async def shutdown():
    pipeline_state["status"] = "stopped"
    print("[API] Shutting down.")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 7 — AI-NIDS Full System")
    print("=" * 60)
    print(f"Interface : {CAPTURE_INTERFACE}")
    print(f"Window    : {WINDOW_SIZE}s  Stride: {STRIDE}s")
    print(f"API       : http://{API_HOST}:{API_PORT}")
    print(f"WebSocket : ws://{API_HOST}:{API_PORT}/ws/alerts")
    print("=" * 60 + "\n")

    uvicorn.run(
        "run_phase7:app",
        host        = API_HOST,
        port        = API_PORT,
        reload      = False,
        loop        = "asyncio",
        log_level   = "warning",   # suppress uvicorn noise
    )
