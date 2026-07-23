from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PacketInfo:
    """
    Parsed representation of a single captured packet.
    All downstream components consume this object —
    never raw PyShark packets.
    """
    timestamp:   float          # Unix timestamp (seconds)
    src_ip:      str
    dst_ip:      str
    src_port:    int            # 0 for ICMP
    dst_port:    int            # 0 for ICMP
    protocol:    int            # 6=TCP, 17=UDP, 1=ICMP
    length:      int            # total packet length (bytes)
    payload_len: int            # bytes above transport header
    header_len:  int            # IP + transport header bytes
    tcp_flags:   Optional[str]  # hex string e.g. "0x012"
    tcp_window:  Optional[int]  # TCP window size (bytes)

    @property
    def has_payload(self) -> bool:
        return self.payload_len > 0

    @property
    def flag_syn(self) -> bool:
        return self._flag(0x002)

    @property
    def flag_fin(self) -> bool:
        return self._flag(0x001)

    @property
    def flag_rst(self) -> bool:
        return self._flag(0x004)

    @property
    def flag_psh(self) -> bool:
        return self._flag(0x008)

    @property
    def flag_ack(self) -> bool:
        return self._flag(0x010)

    @property
    def flag_urg(self) -> bool:
        return self._flag(0x020)

    @property
    def flag_ece(self) -> bool:
        return self._flag(0x040)

    @property
    def flag_cwr(self) -> bool:
        return self._flag(0x080)

    def _flag(self, mask: int) -> bool:
        if not self.tcp_flags:
            return False
        try:
            return bool(int(self.tcp_flags, 16) & mask)
        except (ValueError, TypeError):
            return False
