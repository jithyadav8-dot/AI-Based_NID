import asyncio
from typing import Callable, Awaitable, Optional
from collections import defaultdict
import pyshark

from capture.packet_info import PacketInfo


class PySharkCapture:
    """
    Async live packet capture using PyShark (tshark backend).
    Fires a callback for every parsed PacketInfo.

    Windows usage:
        Run tshark -D to find your interface index.
        Pass the index as iface e.g. iface="1"
    """

    def __init__(
        self,
        iface:           str,
        bpf_filter:      str = "ip",
        packet_callback: Optional[Callable[[PacketInfo], Awaitable[None]]] = None,
    ):
        self.iface           = iface
        self.bpf_filter      = bpf_filter
        self.callback        = packet_callback
        self._stats          = defaultdict(int)

    def _parse(self, packet) -> Optional[PacketInfo]:
        try:
            if not hasattr(packet, "ip"):
                return None

            proto       = int(packet.ip.proto)
            src_port    = 0
            dst_port    = 0
            tcp_flags   = None
            tcp_window  = None
            payload_len = 0
            header_len  = 0

            total_len = int(packet.length)
            ip_header = int(packet.ip.hdr_len)  # IP header length in bytes

            if hasattr(packet, "tcp"):
                tcp         = packet.tcp
                src_port    = int(tcp.srcport)
                dst_port    = int(tcp.dstport)
                tcp_flags   = tcp.flags
                tcp_window  = int(tcp.window_size_value)
                tcp_header  = int(tcp.hdr_len)
                header_len  = ip_header + tcp_header
                payload_len = max(0, total_len - header_len)

            elif hasattr(packet, "udp"):
                udp         = packet.udp
                src_port    = int(udp.srcport)
                dst_port    = int(udp.dstport)
                header_len  = ip_header + 8     # UDP header always 8 bytes
                payload_len = max(0, int(udp.length) - 8)

            elif hasattr(packet, "icmp"):
                header_len  = ip_header + 8     # ICMP header ~8 bytes
                payload_len = max(0, total_len - header_len)

            return PacketInfo(
                timestamp   = float(packet.sniff_timestamp),
                src_ip      = packet.ip.src,
                dst_ip      = packet.ip.dst,
                src_port    = src_port,
                dst_port    = dst_port,
                protocol    = proto,
                length      = total_len,
                payload_len = payload_len,
                header_len  = header_len,
                tcp_flags   = tcp_flags,
                tcp_window  = tcp_window,
            )

        except (AttributeError, ValueError) as e:
            print(f"Parse error on packet: {e}")
            self._stats["parse_errors"] += 1
            return None

    def _handle(self, packet):
        info = self._parse(packet)
        if not info:
            return
        self._stats["captured"] += 1
        proto_name = {6: "tcp", 17: "udp", 1: "icmp"}.get(
            info.protocol, "other"
        )
        self._stats[f"proto_{proto_name}"] += 1
        print(f"[Capture] Received {proto_name} packet ({self._stats['captured']} total)")
        if self.callback:
            def on_done(t):
                if t.exception():
                    print("[Capture] Error in callback:", t.exception())
            task = asyncio.create_task(self.callback(info))
            task.add_done_callback(on_done)

    async def start(self):
        """
        Start live capture asynchronously on the main event loop.
        """
        print(f"[Capture] Interface: {self.iface}  "
              f"Filter: '{self.bpf_filter}'")
        
        while True:
            try:
                self.cap = pyshark.LiveCapture(
                    interface  = self.iface,
                    bpf_filter = self.bpf_filter,
                )
                await self.cap.packets_from_tshark(self._handle)
            except Exception as e:
                import time
                print(f"[Capture] tshark error: {e} — restarting in 2s")
                await asyncio.sleep(2)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    async def stop(self):
        if hasattr(self, 'cap') and self.cap:
            await self.cap.close_async()
