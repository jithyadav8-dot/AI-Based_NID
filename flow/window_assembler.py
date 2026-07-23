import asyncio
import time
from typing import Callable, Awaitable, Optional

from capture.packet_info import PacketInfo
from flow.flow_record import FlowRecord
from flow.feature_extractor import FeatureExtractor


class WindowAssembler:
    """
    Groups PacketInfo objects into bidirectional flows and
    fires a callback every stride with the computed feature vector.

    Design:
      - window_size=10s: feature vector computed over last 10s of traffic
      - stride=2s:       feature vector emitted every 2s per active flow
      - Flows identified by 5-tuple (bidirectional canonical key)
      - Stale flows (5min inactive) pruned automatically
    """

    def __init__(
        self,
        feature_list_path: str = "processed/feature_list.json",
        window_size:       float = 10.0,
        stride:            float = 2.0,
        on_window_ready: Optional[
            Callable[[dict], Awaitable[None]]
        ] = None,
    ):
        self.window_size    = window_size
        self.stride         = stride
        self.on_window_ready = on_window_ready

        self._flows: dict[tuple, FlowRecord] = {}
        self._extractor = FeatureExtractor(feature_list_path)

        # Stats
        self._n_packets  = 0
        self._n_windows  = 0
        self._n_flows    = 0
        self._n_pruned   = 0

    # ── Public API ────────────────────────────────────────────────

    async def update(self, pkt: PacketInfo):
        """Called for every captured packet."""
        key = self._flow_key(pkt)
        if key not in self._flows:
            self._flows[key] = FlowRecord(pkt)
            self._n_flows += 1
        else:
            self._flows[key].add_packet(pkt)
        self._n_packets += 1

    async def run_stride_loop(self):
        """
        Background task: fires every stride seconds.
        For each active flow, extracts features from the last
        window_size seconds and fires on_window_ready.
        """
        print(f"[WindowAssembler] Running — "
              f"window={self.window_size}s  stride={self.stride}s")
        while True:
            await asyncio.sleep(self.stride)
            await self._process_windows()
            self._prune_stale_flows()

    # ── Internal ──────────────────────────────────────────────────

    def _flow_key(self, pkt: PacketInfo) -> tuple:
        """
        Canonical bidirectional 5-tuple key.
        The same flow is identified regardless of packet direction.
        """
        fwd = (pkt.src_ip, pkt.dst_ip, pkt.src_port,
               pkt.dst_port, pkt.protocol)
        rev = (pkt.dst_ip, pkt.src_ip, pkt.dst_port,
               pkt.src_port, pkt.protocol)

        # If flow exists under either direction, return that key
        if fwd in self._flows:
            return fwd
        if rev in self._flows:
            return rev
        # New flow — canonical key is forward direction
        return fwd

    async def _process_windows(self):
        now     = time.time()
        t_start = now - self.window_size

        for key, flow in list(self._flows.items()):
            window_packets = flow.get_window(t_start, now)

            # Skip flows with fewer than 2 packets — no IAT computable
            if len(window_packets) < 2:
                continue

            features = self._extractor.compute(flow, window_packets, now)
            if features is None:
                continue

            self._n_windows += 1

            if self.on_window_ready:
                await self.on_window_ready({
                    "flow_key":  key,
                    "timestamp": now,
                    "features":  features,    # numpy array, 42 values
                    "n_packets": len(window_packets),
                    "src_ip":    flow.fwd_src_ip,
                    "dst_ip":    flow.fwd_dst_ip,
                    "src_port":  flow.fwd_src_port,
                    "dst_port":  flow.fwd_dst_port,
                    "protocol":  flow.protocol,
                })

    def _prune_stale_flows(self):
        stale = [k for k, v in self._flows.items() if v.is_stale]
        for k in stale:
            del self._flows[k]
        self._n_pruned += len(stale)

    @property
    def stats(self) -> dict:
        return {
            "active_flows":   len(self._flows),
            "total_flows":    self._n_flows,
            "total_packets":  self._n_packets,
            "windows_emitted": self._n_windows,
            "flows_pruned":   self._n_pruned,
        }
