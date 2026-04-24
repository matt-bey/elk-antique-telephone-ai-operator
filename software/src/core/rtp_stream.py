"""
Custom RTP Audio Stream - Antique Telephone AI Operator

Replaces pyVoIP's RTP layer with direct int16 → µ-law encoding for
higher audio quality.  pyVoIP's codec path goes int16 → 8-bit linear
→ µ-law, losing 8 bits of precision before the companding step.  This
module encodes directly from int16, preserving the full dynamic range
that G.711 µ-law is designed to handle (~72 dB SNR).

The G711Codec uses precomputed numpy lookup tables for vectorized
encode/decode — no ``audioop`` dependency.

The RTPStream manages a UDP socket with send/receive threads and
exposes a simple read/write API that the existing AudioProcessor
bridge callbacks can use without changes.
"""

import collections
import logging
import random
import socket
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# G.711 µ-law codec (ITU-T recommendation)
# ---------------------------------------------------------------------------

class G711Codec:
    """ITU-T G.711 µ-law codec using numpy lookup tables.

    Encodes int16 PCM directly to 8-bit µ-law and vice versa.  All
    operations are vectorized via precomputed tables so a full 160-sample
    RTP frame encodes in a single numpy indexing operation.

    The algorithm matches the reference C implementation used by
    ``audioop`` (and most telephony stacks):
    - CLIP = 32635 (not 32767)
    - BIAS = 0x84 = 132
    - Exponent found via standard lookup table
    """

    _MULAW_BIAS = 0x84   # 132
    _MULAW_CLIP = 32635   # max magnitude before bias

    # Exponent lookup table (standard, indexed by (biased_magnitude >> 7))
    _EXP_LUT = [
        0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    ]

    # Tables built lazily on first use
    _encode_table: Optional[np.ndarray] = None
    _decode_table: Optional[np.ndarray] = None

    @classmethod
    def _build_tables(cls) -> None:
        """Build the encode and decode lookup tables once."""
        if cls._encode_table is not None:
            return

        # --- Decode table (µ-law byte → int16) ---
        decode = np.zeros(256, dtype=np.int16)
        for i in range(256):
            val = ~i & 0xFF
            sign = val & 0x80
            exponent = (val >> 4) & 0x07
            mantissa = val & 0x0F
            magnitude = ((mantissa << 1) + 33) << (exponent + 2)
            magnitude -= cls._MULAW_BIAS
            if sign:
                decode[i] = np.int16(-magnitude)
            else:
                decode[i] = np.int16(magnitude)
        cls._decode_table = decode

        # --- Encode table (int16 → µ-law byte) ---
        encode = np.zeros(65536, dtype=np.uint8)
        for s in range(-32768, 32768):
            sign = 0x80 if s < 0 else 0x00
            magnitude = min(abs(s), cls._MULAW_CLIP)
            magnitude += cls._MULAW_BIAS

            exponent = cls._EXP_LUT[(magnitude >> 7) & 0xFF]
            mantissa = (magnitude >> (exponent + 3)) & 0x0F
            ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
            encode[s + 32768] = ulaw_byte
        cls._encode_table = encode

    @classmethod
    def encode(cls, pcm_int16: np.ndarray) -> np.ndarray:
        """Encode int16 PCM samples to µ-law bytes (vectorized)."""
        cls._build_tables()
        # Offset int16 to uint16 range [0, 65535] for table indexing
        indices = pcm_int16.astype(np.int32) + 32768
        return cls._encode_table[indices]

    @classmethod
    def decode(cls, ulaw: np.ndarray) -> np.ndarray:
        """Decode µ-law bytes to int16 PCM samples (vectorized)."""
        cls._build_tables()
        return cls._decode_table[ulaw.astype(np.uint8)]


# ---------------------------------------------------------------------------
# RTP stream
# ---------------------------------------------------------------------------

# RTP header constants
_RTP_VERSION = 2
_RTP_PAYLOAD_PCMU = 0
_RTP_HEADER_SIZE = 12
_SAMPLES_PER_FRAME = 160  # 20 ms at 8 kHz
_FRAME_DURATION_NS = 20_000_000  # 20 ms in nanoseconds


class RTPStream:
    """Bidirectional RTP audio stream over UDP.

    Sends and receives G.711 µ-law (PCMU) RTP packets with proper
    headers, sequence numbers, and timestamps.  The read/write API
    works with int16 numpy arrays — encoding/decoding is handled
    internally.
    """

    def __init__(
        self,
        local_ip: str,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        ssrc: Optional[int] = None,
    ):
        self.local_ip = local_ip
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port

        self._ssrc = ssrc or random.randint(1000, 65530)
        self._sequence = random.randint(1, 100)
        self._timestamp = random.randint(1, 10000)

        self._sock: Optional[socket.socket] = None
        self._running = False

        self._send_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None

        # Thread-safe audio buffers (deque is safe for single-producer
        # single-consumer append/popleft under CPython GIL).
        # Max ~1 second of buffered audio in each direction.
        self._out_buf: collections.deque = collections.deque(maxlen=50)
        self._in_buf: collections.deque = collections.deque(maxlen=50)

    def start(self) -> None:
        """Bind the UDP socket and start send/receive threads."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Retry bind in case the OS hasn't fully released the port
        # from pyVoIP's recently-closed RTPClient socket.
        for attempt in range(10):
            try:
                self._sock.bind((self.local_ip, self.local_port))
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(0.05)

        self._sock.setblocking(False)
        self._running = True

        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="rtp-send"
        )
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="rtp-recv"
        )
        self._send_thread.start()
        self._recv_thread.start()
        logger.info(
            f"RTP stream started: {self.local_ip}:{self.local_port} "
            f"→ {self.remote_ip}:{self.remote_port}"
        )

    def stop(self) -> None:
        """Stop threads and close the socket."""
        self._running = False
        if self._send_thread:
            self._send_thread.join(timeout=2.0)
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        logger.info("RTP stream stopped")

    # -- Public audio API --------------------------------------------------

    def write(self, pcm_int16: np.ndarray) -> None:
        """Queue int16 PCM for transmission.

        Accepts any length; segments into 160-sample frames internally
        so the send thread always sends properly sized RTP packets.
        """
        pos = 0
        while pos + _SAMPLES_PER_FRAME <= len(pcm_int16):
            self._out_buf.append(pcm_int16[pos:pos + _SAMPLES_PER_FRAME].copy())
            pos += _SAMPLES_PER_FRAME
        # Queue any remaining samples (padded with silence in send loop)
        if pos < len(pcm_int16):
            self._out_buf.append(pcm_int16[pos:].copy())

    def read(self) -> Optional[np.ndarray]:
        """Return the next decoded int16 frame, or None if empty."""
        try:
            return self._in_buf.popleft()
        except IndexError:
            return None

    # -- Send thread -------------------------------------------------------

    def _send_loop(self) -> None:
        """Send one RTP packet every 20 ms."""
        # Pre-encode silence for keepalive packets
        silence_payload = bytes([0xFF] * _SAMPLES_PER_FRAME)

        while self._running:
            send_start = time.monotonic_ns()

            # Dequeue audio or send silence
            try:
                pcm = self._out_buf.popleft()
                payload = G711Codec.encode(pcm).tobytes()
            except IndexError:
                payload = silence_payload

            # Ensure payload is exactly 160 bytes
            if len(payload) < _SAMPLES_PER_FRAME:
                payload += bytes([0xFF] * (_SAMPLES_PER_FRAME - len(payload)))
            elif len(payload) > _SAMPLES_PER_FRAME:
                payload = payload[:_SAMPLES_PER_FRAME]

            header = self._build_rtp_header(
                self._sequence, self._timestamp, self._ssrc
            )
            try:
                self._sock.sendto(
                    header + payload, (self.remote_ip, self.remote_port)
                )
            except (OSError, AttributeError):
                pass  # socket closed during shutdown

            self._sequence = (self._sequence + 1) & 0xFFFF
            self._timestamp = (self._timestamp + _SAMPLES_PER_FRAME) & 0xFFFFFFFF

            # Pace to 20 ms
            elapsed = time.monotonic_ns() - send_start
            remaining = _FRAME_DURATION_NS - elapsed
            if remaining > 0:
                time.sleep(remaining / 1_000_000_000)

    # -- Receive thread ----------------------------------------------------

    def _recv_loop(self) -> None:
        """Receive RTP packets and decode to int16."""
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
                if len(data) < _RTP_HEADER_SIZE:
                    continue
                pt, _, _, _, payload = self._parse_rtp_header(data)
                if pt != _RTP_PAYLOAD_PCMU:
                    continue  # skip telephone-event, etc.
                if payload:
                    pcm = G711Codec.decode(
                        np.frombuffer(payload, dtype=np.uint8)
                    )
                    self._in_buf.append(pcm)
            except BlockingIOError:
                time.sleep(0.005)
            except (OSError, AttributeError):
                break  # socket closed

    # -- RTP packet helpers ------------------------------------------------

    @staticmethod
    def _build_rtp_header(
        sequence: int, timestamp: int, ssrc: int,
        payload_type: int = _RTP_PAYLOAD_PCMU,
    ) -> bytes:
        """Build a 12-byte RTP header (RFC 3550)."""
        byte0 = (_RTP_VERSION << 6)  # V=2, P=0, X=0, CC=0
        byte1 = payload_type & 0x7F  # M=0
        return struct.pack(
            "!BBHII", byte0, byte1, sequence, timestamp, ssrc
        )

    @staticmethod
    def _parse_rtp_header(packet: bytes) -> Tuple[int, int, int, int, bytes]:
        """Parse an RTP packet into (payload_type, seq, ts, ssrc, payload)."""
        byte0, byte1, seq, ts, ssrc = struct.unpack_from("!BBHII", packet)
        cc = byte0 & 0x0F
        pt = byte1 & 0x7F
        header_len = _RTP_HEADER_SIZE + cc * 4
        payload = packet[header_len:]
        return pt, seq, ts, ssrc, payload
