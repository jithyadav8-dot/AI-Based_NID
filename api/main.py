# api/main.py

import asyncio
import json
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from api.models import AlertSchema, StatsSchema, SystemStatusSchema
from inference.predictor import Alert

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title       = "AI-NIDS API",
    description = "Real-time Network Intrusion Detection System",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Shared state (injected by run_phase7.py at startup) ───────────
# These are populated once the pipeline initialises.
# Endpoints read from these; the pipeline writes to them.
pipeline_state = {
    "predictor":   None,
    "assembler":   None,
    "capture":     None,
    "start_time":  None,
    "status":      "starting",
}

# ── Alert store ───────────────────────────────────────────────────
from config import MAX_ALERT_STORE

alert_store:    deque  = deque(maxlen=MAX_ALERT_STORE)
alert_id_ctr:   int    = 0
alert_counts:   dict   = defaultdict(int)


# ── WebSocket connection manager ──────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] Client connected  (active: {len(self.active)})")

        # Send the last 20 alerts immediately on connect
        # so the dashboard doesn't start blank
        recent = list(alert_store)[:20]
        for a in reversed(recent):
            try:
                await ws.send_json({**a, "_type": "history"})
            except Exception:
                break

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        print(f"[WS] Client disconnected (active: {len(self.active)})")

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)


manager = ConnectionManager()


# ── Called by run_phase7.py for every scored window ───────────────
async def on_alert(alert: Alert):
    """
    Entry point from the inference pipeline into the API layer.
    Called for every window regardless of level so stats stay accurate.
    Only non-NORMAL alerts are stored and broadcast to save memory.
    """
    global alert_id_ctr

    alert_counts[alert.level] += 1
    alert_counts["total"]     += 1

    if alert.level == "NORMAL":
        return

    alert_id_ctr += 1

    record = {
        "id":             alert_id_ctr,
        "timestamp":      alert.timestamp,
        "src_ip":         alert.src_ip,
        "dst_ip":         alert.dst_ip,
        "src_port":       alert.src_port,
        "dst_port":       alert.dst_port,
        "protocol":       alert.protocol,
        "n_packets":      alert.n_packets,
        "xgb_class":      alert.xgb_class,
        "xgb_confidence": round(alert.xgb_confidence, 4),
        "xgb_is_attack":  alert.xgb_is_attack,
        "ae_error":       round(alert.ae_error, 6),
        "ae_threshold":   round(alert.ae_threshold, 6),
        "ae_is_anomaly":  alert.ae_is_anomaly,
        "severity":       round(alert.severity, 2),
        "level":          alert.level,
        "label":          alert.label,
        "_type":          "alert",
    }

    alert_store.appendleft(record)
    await manager.broadcast(record)


# ── REST: GET /alerts ─────────────────────────────────────────────
@app.get("/alerts", response_model=List[AlertSchema])
def get_alerts(
    limit: int           = Query(default=100, le=1000),
    level: Optional[str] = Query(default=None,
                                 description="Filter: CRITICAL/HIGH/MEDIUM/LOW"),
    since: Optional[float] = Query(default=None,
                                   description="Unix timestamp — only alerts after this"),
):
    """
    Returns recent alerts, newest first.
    Supports filtering by level and timestamp.
    """
    alerts = list(alert_store)

    if level:
        alerts = [a for a in alerts if a["level"] == level.upper()]

    if since is not None:
        alerts = [a for a in alerts if a["timestamp"] > since]

    return alerts[:limit]


# ── REST: GET /alerts/{id} ────────────────────────────────────────
@app.get("/alerts/{alert_id}", response_model=AlertSchema)
def get_alert(alert_id: int):
    from fastapi import HTTPException
    for a in alert_store:
        if a["id"] == alert_id:
            return a
    raise HTTPException(status_code=404, detail="Alert not found")


# ── REST: GET /stats ──────────────────────────────────────────────
@app.get("/stats", response_model=StatsSchema)
def get_stats():
    predictor = pipeline_state["predictor"]
    assembler  = pipeline_state["assembler"]
    capture    = pipeline_state["capture"]
    start_time = pipeline_state["start_time"]

    pred_stats = predictor.stats if predictor else {}
    asm_stats  = assembler.stats  if assembler  else {}
    cap_stats  = capture.stats    if capture    else {}

    uptime = (time.time() - start_time) if start_time else 0

    return {
        "windows_scored":  pred_stats.get("windows_scored", 0),
        "alerts_raised":   pred_stats.get("alerts_raised",  0),
        "avg_latency_ms":  pred_stats.get("avg_latency_ms", 0),
        "active_flows":    asm_stats.get("active_flows",    0),
        "total_flows":     asm_stats.get("total_flows",     0),
        "alert_counts":    dict(alert_counts),
        "uptime_seconds":  round(uptime, 1),
        "capture_stats":   cap_stats,
    }


# ── REST: GET /status ─────────────────────────────────────────────
@app.get("/status", response_model=SystemStatusSchema)
def get_status():
    predictor = pipeline_state["predictor"]
    from config import (CAPTURE_INTERFACE, WINDOW_SIZE, STRIDE)

    return {
        "status":        pipeline_state["status"],
        "interface":     CAPTURE_INTERFACE,
        "window_size":   WINDOW_SIZE,
        "stride":        STRIDE,
        "ae_threshold":  predictor.ae_threshold    if predictor else 0,
        "inf_threshold": predictor.inf_threshold   if predictor else 0,
        "feature_count": predictor.scaler.n_features_in_ if predictor else 42,
        "model_classes": list(predictor.le.classes_) if predictor else [],
    }


# ── REST: DELETE /alerts ──────────────────────────────────────────
@app.delete("/alerts")
def clear_alerts():
    """Clears the in-memory alert store. Useful for demo resets."""
    global alert_id_ctr
    alert_store.clear()
    alert_counts.clear()
    alert_id_ctr = 0
    return {"cleared": True}


# ── WebSocket: /ws/alerts ─────────────────────────────────────────
@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """
    Live alert stream. On connect, sends the last 20 alerts
    as history events, then streams new alerts in real time.

    Message format:
      { ...AlertSchema fields..., "_type": "alert" | "history" | "ping" }
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive — send ping every 30s
            # Also receive any client messages (e.g. filter commands)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0
                )
                # Client can send {"action": "clear"} to reset their view
                msg = json.loads(data)
                if msg.get("action") == "ping":
                    await websocket.send_json({"_type": "pong",
                                               "ts": time.time()})
            except asyncio.TimeoutError:
                # No message from client — send keepalive ping
                await websocket.send_json({"_type": "ping",
                                           "ts": time.time()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] Error: {e}")
        manager.disconnect(websocket)


# ── Health check ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "time":   datetime.utcnow().isoformat(),
        "alerts": len(alert_store),
    }
