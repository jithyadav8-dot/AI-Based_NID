import json
import numpy as np
from typing import Optional
from capture.packet_info import PacketInfo
from flow.flow_record import FlowRecord

# Gap threshold for active/idle period detection (microseconds → seconds)
# CICFlowMeter default is 5 seconds
ACTIVE_TIMEOUT = 5.0


class FeatureExtractor:
    """
    Computes CICFlowMeter-compatible features from a sliding window
    of packets belonging to one flow.

    Computes all possible features, then selects and orders them
    according to feature_list.json — the exact contract from Phase 1.
    """

    def __init__(self, feature_list_path: str = "processed/feature_list.json"):
        with open(feature_list_path) as f:
            self.feature_list = json.load(f)
        print(f"[FeatureExtractor] Loaded {len(self.feature_list)} features")

    def compute(
        self,
        flow:    FlowRecord,
        window:  list,          # [(PacketInfo, is_forward), ...]
        now:     float,
    ) -> Optional[np.ndarray]:
        """
        Computes the feature vector for one flow window.
        Returns a float32 numpy array of shape (n_features,)
        or None if the window is invalid.
        """
        if len(window) < 2:
            return None

        # ── Split into forward and backward packets ────────────────
        fwd = [(p, ts) for p, fwd in window
               for ts in [p.timestamp] if fwd]
        bwd = [(p, ts) for p, fwd in window
               for ts in [p.timestamp] if not fwd]

        fwd_pkts  = [p for p, _ in [(pkt, fwd) for pkt, fwd in window if fwd]]
        bwd_pkts  = [p for p, _ in [(pkt, fwd) for pkt, fwd in window if not fwd]]
        all_pkts  = [p for p, _ in window]

        # Timestamps
        all_ts  = sorted([p.timestamp for p in all_pkts])
        fwd_ts  = sorted([p.timestamp for p in fwd_pkts])
        bwd_ts  = sorted([p.timestamp for p in bwd_pkts])

        # Lengths
        fwd_lens = [p.length for p in fwd_pkts]
        bwd_lens = [p.length for p in bwd_pkts]
        all_lens = [p.length for p in all_pkts]

        # Duration (microseconds in CICFlowMeter — use seconds here,
        # consistent with training data which used CICFlowMeter microseconds)
        duration = (all_ts[-1] - all_ts[0]) * 1_000_000  # → microseconds
        duration = max(duration, 1e-6)   # prevent division by zero

        duration_s = duration / 1_000_000   # seconds for rate features

        # ── IAT computation ────────────────────────────────────────
        def iats(ts_list):
            if len(ts_list) < 2:
                return [0.0]
            return [(ts_list[i] - ts_list[i-1]) * 1_000_000
                    for i in range(1, len(ts_list))]

        flow_iats = iats(all_ts)
        fwd_iats  = iats(fwd_ts)
        bwd_iats  = iats(bwd_ts)

        # ── Active / Idle periods ──────────────────────────────────
        def active_idle(ts_list):
            """
            Splits a sorted timestamp list into active periods and
            idle gaps. Returns (active_times, idle_times) in microseconds.
            """
            if len(ts_list) < 2:
                return [0.0], [0.0]

            active_times = []
            idle_times   = []
            period_start = ts_list[0]
            prev_ts      = ts_list[0]

            for ts in ts_list[1:]:
                gap = ts - prev_ts
                if gap > ACTIVE_TIMEOUT:
                    # End of active period
                    active_times.append((prev_ts - period_start) * 1e6)
                    idle_times.append(gap * 1e6)
                    period_start = ts
                prev_ts = ts

            # Last active period
            active_times.append((prev_ts - period_start) * 1e6)

            return (active_times if active_times else [0.0],
                    idle_times   if idle_times   else [0.0])

        active_times, idle_times = active_idle(all_ts)

        # ── Helper stats functions ────────────────────────────────
        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        def safe_std(lst):  return float(np.std(lst))  if lst else 0.0
        def safe_max(lst):  return float(np.max(lst))  if lst else 0.0
        def safe_min(lst):  return float(np.min(lst))  if lst else 0.0
        def safe_sum(lst):  return float(np.sum(lst))  if lst else 0.0
        def safe_var(lst):  return float(np.var(lst))  if lst else 0.0

        n_fwd  = len(fwd_pkts)
        n_bwd  = len(bwd_pkts)
        n_all  = len(all_pkts)

        fwd_bytes = safe_sum(fwd_lens)
        bwd_bytes = safe_sum(bwd_lens)
        all_bytes = safe_sum(all_lens)

        # ── Build full feature dict ────────────────────────────────
        # Keys match the canonical 2017 column names used in training
        f: dict = {}

        # Core counts
        f["Destination Port"]           = flow.dst_port
        f["Protocol"]                   = flow.protocol
        f["Flow Duration"]              = duration

        f["Total Fwd Packets"]          = n_fwd
        f["Total Backward Packets"]     = n_bwd
        f["Total Length of Fwd Packets"] = fwd_bytes
        f["Total Length of Bwd Packets"] = bwd_bytes

        # Fwd packet length stats
        f["Fwd Packet Length Max"]      = safe_max(fwd_lens)
        f["Fwd Packet Length Min"]      = safe_min(fwd_lens)
        f["Fwd Packet Length Mean"]     = safe_mean(fwd_lens)
        f["Fwd Packet Length Std"]      = safe_std(fwd_lens)

        # Bwd packet length stats
        f["Bwd Packet Length Max"]      = safe_max(bwd_lens)
        f["Bwd Packet Length Min"]      = safe_min(bwd_lens)
        f["Bwd Packet Length Mean"]     = safe_mean(bwd_lens)
        f["Bwd Packet Length Std"]      = safe_std(bwd_lens)

        # Rate features
        f["Flow Bytes/s"]               = all_bytes / duration_s
        f["Flow Packets/s"]             = n_all    / duration_s

        # Flow IAT
        f["Flow IAT Mean"]              = safe_mean(flow_iats)
        f["Flow IAT Std"]               = safe_std(flow_iats)
        f["Flow IAT Max"]               = safe_max(flow_iats)
        f["Flow IAT Min"]               = safe_min(flow_iats)

        # Fwd IAT
        f["Fwd IAT Total"]              = safe_sum(fwd_iats)
        f["Fwd IAT Mean"]               = safe_mean(fwd_iats)
        f["Fwd IAT Std"]                = safe_std(fwd_iats)
        f["Fwd IAT Max"]                = safe_max(fwd_iats)
        f["Fwd IAT Min"]                = safe_min(fwd_iats)

        # Bwd IAT
        f["Bwd IAT Total"]              = safe_sum(bwd_iats)
        f["Bwd IAT Mean"]               = safe_mean(bwd_iats)
        f["Bwd IAT Std"]                = safe_std(bwd_iats)
        f["Bwd IAT Max"]                = safe_max(bwd_iats)
        f["Bwd IAT Min"]                = safe_min(bwd_iats)

        # TCP flags
        f["Fwd PSH Flags"]  = sum(1 for p in fwd_pkts if p.flag_psh)
        f["Bwd PSH Flags"]  = sum(1 for p in bwd_pkts if p.flag_psh)
        f["Fwd URG Flags"]  = sum(1 for p in fwd_pkts if p.flag_urg)
        f["Bwd URG Flags"]  = sum(1 for p in bwd_pkts if p.flag_urg)

        # Header lengths
        f["Fwd Header Length"] = safe_sum([p.header_len for p in fwd_pkts])
        f["Bwd Header Length"] = safe_sum([p.header_len for p in bwd_pkts])

        # Packet rates
        f["Fwd Packets/s"]  = n_fwd / duration_s
        f["Bwd Packets/s"]  = n_bwd / duration_s

        # All-packet length stats
        f["Min Packet Length"]       = safe_min(all_lens)
        f["Max Packet Length"]       = safe_max(all_lens)
        f["Packet Length Mean"]      = safe_mean(all_lens)
        f["Packet Length Std"]       = safe_std(all_lens)
        f["Packet Length Variance"]  = safe_var(all_lens)

        # Flag counts (all directions)
        all_tcp = fwd_pkts + bwd_pkts
        f["FIN Flag Count"] = sum(1 for p in all_tcp if p.flag_fin)
        f["SYN Flag Count"] = sum(1 for p in all_tcp if p.flag_syn)
        f["RST Flag Count"] = sum(1 for p in all_tcp if p.flag_rst)
        f["PSH Flag Count"] = sum(1 for p in all_tcp if p.flag_psh)
        f["ACK Flag Count"] = sum(1 for p in all_tcp if p.flag_ack)
        f["URG Flag Count"] = sum(1 for p in all_tcp if p.flag_urg)
        f["CWE Flag Count"] = sum(1 for p in all_tcp if p.flag_cwr)
        f["ECE Flag Count"] = sum(1 for p in all_tcp if p.flag_ece)

        # Ratio / size features
        f["Down/Up Ratio"]       = bwd_bytes / fwd_bytes if fwd_bytes > 0 else 0.0
        f["Average Packet Size"] = all_bytes / n_all     if n_all    > 0 else 0.0
        f["Avg Fwd Segment Size"] = fwd_bytes / n_fwd   if n_fwd    > 0 else 0.0
        f["Avg Bwd Segment Size"] = bwd_bytes / n_bwd   if n_bwd    > 0 else 0.0

        # Bulk features (CICFlowMeter computes these; 0 for non-bulk)
        f["Fwd Avg Bytes/Bulk"]    = 0.0
        f["Fwd Avg Packets/Bulk"]  = 0.0
        f["Fwd Avg Bulk Rate"]     = 0.0
        f["Bwd Avg Bytes/Bulk"]    = 0.0
        f["Bwd Avg Packets/Bulk"]  = 0.0
        f["Bwd Avg Bulk Rate"]     = 0.0

        # Subflow (equals full flow for single-flow windows)
        f["Subflow Fwd Packets"] = n_fwd
        f["Subflow Fwd Bytes"]   = fwd_bytes
        f["Subflow Bwd Packets"] = n_bwd
        f["Subflow Bwd Bytes"]   = bwd_bytes

        # Initial TCP window sizes
        f["Init_Win_bytes_forward"]  = flow.init_win_fwd  or 0
        f["Init_Win_bytes_backward"] = flow.init_win_bwd  or 0

        # Active data packets (fwd packets with payload)
        f["act_data_pkt_fwd"] = sum(1 for p in fwd_pkts if p.has_payload)

        # Min segment size (min fwd header length — proxy for TCP MSS)
        fwd_headers = [p.header_len for p in fwd_pkts if p.header_len > 0]
        f["min_seg_size_forward"] = safe_min(fwd_headers)

        # Active / Idle time stats
        f["Active Mean"]  = safe_mean(active_times)
        f["Active Std"]   = safe_std(active_times)
        f["Active Max"]   = safe_max(active_times)
        f["Active Min"]   = safe_min(active_times)
        f["Idle Mean"]    = safe_mean(idle_times)
        f["Idle Std"]     = safe_std(idle_times)
        f["Idle Max"]     = safe_max(idle_times)
        f["Idle Min"]     = safe_min(idle_times)

        # ── Select and order by feature_list.json ─────────────────
        try:
            vector = np.array(
                [f.get(feat, 0.0) for feat in self.feature_list],
                dtype=np.float32
            )
        except Exception as e:
            print(f"[FeatureExtractor] Error building vector: {e}")
            return None

        # Cap physically implausible values caused by near-zero durations.
        # Flow Bytes/s and Flow Packets/s can explode when duration → 0.
        # 1e9 bytes/s = 1 Gbps — a reasonable hard ceiling.
        FEATURE_CAPS = {
            "Flow Bytes/s":   1e9,
            "Flow Packets/s": 1e6,
            "Fwd Packets/s":  1e6,
            "Bwd Packets/s":  1e6,
            "Flow IAT Mean":  1e10,
            "Flow IAT Max":   1e10,
            "Fwd IAT Total":  1e10,
            "Bwd IAT Total":  1e10,
        }

        for feat, cap in FEATURE_CAPS.items():
            if feat in self.feature_list:
                idx = self.feature_list.index(feat)
                vector[idx] = min(vector[idx], cap)

        # Final safety net — clip anything still extreme
        vector = np.clip(vector, 0, 1e10)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        return vector
