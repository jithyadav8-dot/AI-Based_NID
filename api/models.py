# api/models.py

from pydantic import BaseModel
from typing import Optional


class AlertSchema(BaseModel):
    id:               int
    timestamp:        float
    src_ip:           str
    dst_ip:           str
    src_port:         int
    dst_port:         int
    protocol:         str
    n_packets:        int
    xgb_class:        str
    xgb_confidence:   float
    xgb_is_attack:    bool
    ae_error:         float
    ae_threshold:     float
    ae_is_anomaly:    bool
    severity:         float
    level:            str
    label:            str
    # ── Deduplication / event-collapsing fields ──────────────────
    count:            int   = 1      # how many times this event repeated
    first_seen:       float = 0.0   # unix ts of first occurrence
    last_seen:        float = 0.0   # unix ts of most-recent repeat


class AlertsListSchema(BaseModel):
    """Paginated alert list."""
    total:    int
    offset:   int
    limit:    int
    alerts:   list[AlertSchema]


class StatsSchema(BaseModel):
    windows_scored:   int
    alerts_raised:    int
    avg_latency_ms:   float
    active_flows:     int
    total_flows:      int
    alert_counts:     dict
    uptime_seconds:   float
    capture_stats:    dict


class SystemStatusSchema(BaseModel):
    status:           str       # "running" | "starting" | "error"
    interface:        str
    window_size:      float
    stride:           float
    ae_threshold:     float
    inf_threshold:    float
    feature_count:    int
    model_classes:    list


class ConfigUpdate(BaseModel):
    """Mutable config fields."""
    ae_threshold:      Optional[float] = None
    interface:         Optional[str]   = None
