import time
from collections import deque
from typing import Optional
from capture.packet_info import PacketInfo


class FlowRecord:
    """
    Stores all packets for one bidirectional flow.
    Forward direction = direction of the first packet.
    Maintains a rolling deque for memory efficiency.
    """

    # Max packets stored per flow — prevents unbounded memory on long flows
    MAX_PACKETS = 5000

    def __init__(self, first_packet: PacketInfo):
        # Forward direction defined by first packet
        self.fwd_src_ip   = first_packet.src_ip
        self.fwd_src_port = first_packet.src_port
        self.fwd_dst_ip   = first_packet.dst_ip
        self.fwd_dst_port = first_packet.dst_port
        self.protocol     = first_packet.protocol
        self.dst_port     = first_packet.dst_port  # for Destination Port feature

        self.start_time = first_packet.timestamp
        self.last_seen  = first_packet.timestamp

        # (PacketInfo, is_forward) tuples — oldest first
        self.packets: deque = deque(maxlen=self.MAX_PACKETS)

        # Initial TCP window sizes — set once from first packet each direction
        self.init_win_fwd: Optional[int] = first_packet.tcp_window
        self.init_win_bwd: Optional[int] = None

        self.add_packet(first_packet)

    def is_forward(self, pkt: PacketInfo) -> bool:
        """True if this packet is in the same direction as the first packet."""
        return (
            pkt.src_ip   == self.fwd_src_ip and
            pkt.src_port == self.fwd_src_port
        )

    def add_packet(self, pkt: PacketInfo):
        fwd = self.is_forward(pkt)
        self.packets.append((pkt, fwd))
        self.last_seen = pkt.timestamp

        # Capture initial backward window size once
        if not fwd and self.init_win_bwd is None:
            self.init_win_bwd = pkt.tcp_window

    def get_window(self, t_start: float, t_end: float) -> list:
        """
        Returns (PacketInfo, is_forward) pairs within [t_start, t_end].
        Used by the feature extractor every stride.
        """
        return [
            (pkt, fwd)
            for pkt, fwd in self.packets
            if t_start <= pkt.timestamp <= t_end
        ]

    @property
    def is_stale(self) -> bool:
        """True if no packet has arrived in the last 5 minutes."""
        return (time.time() - self.last_seen) > 300
