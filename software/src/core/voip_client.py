"""
VoIP Client Module - Antique Telephone AI Operator

Handles SIP/VoIP communication via pyVoIP (pure Python, pip-installable).
Designed for Callcentric SIP service: G.711 µ-law/A-law, RFC 2833 DTMF,
DNS SRV enabled, no STUN required.
"""

import logging
import asyncio
import socket
import threading
import time
import uuid
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
from dataclasses import dataclass, field

try:
    from pyVoIP.VoIP import VoIPPhone, VoIPCall as PyVoIPCall, CallState as PyVoIPCallState
    from pyVoIP.VoIP import PhoneStatus
    HAS_PYVOIP = True
except ImportError:
    HAS_PYVOIP = False
    logging.warning("pyVoIP not available - VoIP functionality disabled")

import math
import numpy as np
from core.rtp_stream import RTPStream
from utils.config_manager import ConfigManager


class CallState(Enum):
    """Local call state (maps from pyVoIP's CallState)."""
    IDLE = "idle"
    DIALING = "dialing"
    RINGING = "ringing"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class SIPAccount:
    """SIP account configuration."""
    username: str
    password: str
    domain: str
    port: int = 5060


@dataclass
class CallInfo:
    """Read-only call snapshot returned to callers."""
    call_id: str
    remote_uri: str
    state: CallState
    duration: float
    start_time: float


@dataclass
class _ActiveCall:
    """Internal call tracking entry."""
    call_id: str
    remote_uri: str
    state: CallState
    start_time: float = field(default_factory=time.time)
    pyvoip_call: Optional[Any] = None  # PyVoIPCall instance
    audio_in_callback: Optional[Callable] = None
    audio_out_callback: Optional[Callable] = None
    bridge_thread: Optional[threading.Thread] = None
    bridge_active: bool = False
    # Custom RTP stream (replaces pyVoIP's RTPClient for audio)
    rtp_stream: Optional[Any] = None
    # Call state monitor
    monitor_thread: Optional[threading.Thread] = None
    monitor_active: bool = False
    on_state_change: Optional[Callable] = None

    def get_duration(self) -> float:
        return time.time() - self.start_time


class VoIPClient:
    """
    VoIP client backed by pyVoIP.

    Callcentric configuration:
      - Registrar: sip.callcentric.net
      - Port: 5060 UDP
      - Codecs: G.711 µ-law (PCMU) / A-law (PCMA)
      - DTMF: RFC 2833 (payload type 101)
      - STUN: disabled (Callcentric recommends off)
      - Registration expiry: 1800 s

    Public API preserved from the original PJSIP implementation so that
    main.py needs no changes.
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)

        self.sip_account = self._load_sip_config()
        self._phone: Optional[Any] = None  # VoIPPhone
        self.active_calls: Dict[str, _ActiveCall] = {}
        self.registered = False
        self.running = False
        self._incoming_call_callback: Optional[Callable] = None

        # Silence gate: suppress ambient noise on outbound RTP.
        # - Lookback buffer preserves speech onsets (soft word beginnings)
        # - Hangover keeps transmitting after speech ends (word tails)
        self._rtp_silence_threshold = int(
            self.config.get('audio.rtp_silence_threshold', 300)
        )
        self._silence_hangover_frames = 15   # ~300ms tail after speech
        self._silence_hangover = 0
        self._silence_lookback_size = 5      # ~100ms of pre-speech audio
        self._silence_lookback: list = []    # ring buffer of recent "silent" frames

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_sip_config(self) -> SIPAccount:
        return SIPAccount(
            username=str(self.config.get('sip.username', '')),
            password=str(self.config.get('sip.password', '')),
            domain=str(self.config.get('sip.domain', 'sip.callcentric.net')),
            port=int(self.config.get('sip.port', 5060)),
        )

    def _normalize_number(self, destination: str) -> str:
        """Normalize a dialed number to E.164-style for Callcentric.

        Callcentric requires:
          - US calls: 1 + area code + number (11 digits)
          - International: 011 + country code + number

        Handles:
          - 7 digits  → prepend local area code + country code 1
          - 10 digits → prepend country code 1
          - 11 digits starting with 1 → pass through
          - Longer / international → pass through
        """
        digits = ''.join(c for c in destination if c.isdigit())
        local_area_code = str(self.config.get('sip.local_area_code', ''))

        if len(digits) == 7 and local_area_code:
            digits = '1' + local_area_code + digits
        elif len(digits) == 10:
            digits = '1' + digits

        return digits

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    def _detect_local_ip(self) -> str:
        """Detect the local IP address routable to the SIP server.

        Opens a UDP socket to the SIP server (no data sent) and reads
        back the local address the OS chose.  This gives the correct
        interface IP even on multi-homed machines.  Falls back to
        "0.0.0.0" if detection fails.
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((self.sip_account.domain, self.sip_account.port))
            local_ip = s.getsockname()[0]
            s.close()
            self.logger.debug(f"Detected local IP: {local_ip}")
            return local_ip
        except Exception as e:
            self.logger.warning(f"Could not detect local IP: {e}")
            return "0.0.0.0"

    # ------------------------------------------------------------------
    # Incoming call handler (called from pyVoIP's internal thread)
    # ------------------------------------------------------------------

    def _on_incoming_call(self, pyvoip_call: Any) -> None:
        call_id = str(uuid.uuid4())
        try:
            remote = str(pyvoip_call.request.headers.get('From', 'unknown'))
        except Exception:
            remote = 'unknown'

        entry = _ActiveCall(
            call_id=call_id,
            remote_uri=remote,
            state=CallState.RINGING,
            pyvoip_call=pyvoip_call,
        )
        self.active_calls[call_id] = entry
        self.logger.info(f"Incoming call {call_id} from {remote}")

        if self._incoming_call_callback:
            self._incoming_call_callback(call_id, remote)

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    async def start_service(self) -> bool:
        """Start VoIP service and register with SIP server."""
        if not HAS_PYVOIP:
            self.logger.error("pyVoIP not available - VoIP service disabled")
            return False

        if not self.sip_account.username:
            self.logger.error("SIP username not configured")
            return False

        try:
            # Patch pyVoIP for 407 Proxy-Auth (Callcentric) — safe to
            # call multiple times; only patches once.
            from core.pyvoip_patch import apply as _apply_pyvoip_patch
            _apply_pyvoip_patch()

            rtp_low = int(self.config.get('sip.rtp_port_min', 10000))
            rtp_high = int(self.config.get('sip.rtp_port_max', 20000))
            # Local SIP port — defaults to sip.port (5060) but can be
            # overridden with sip.local_port if 5060 is already in use.
            sip_port = int(
                self.config.get('sip.local_port', 0)
                or self.config.get('sip.port', 5060)
            )

            # Detect our local IP that's routable to the SIP server.
            # pyVoIP defaults myIP to "0.0.0.0" which makes the SDP
            # c= line unroutable — the SBC can't send RTP back to us
            # and responds with a=recvonly (one-way audio).
            my_ip = self._detect_local_ip()

            self.logger.info(
                f"SIP connecting: {self.sip_account.username}@"
                f"{self.sip_account.domain}:{self.sip_account.port} "
                f"(sipPort={sip_port}, rtp={rtp_low}-{rtp_high}, "
                f"myIP={my_ip})"
            )

            # Enable pyVoIP packet-level debug when our logger is at DEBUG
            import pyVoIP as _pyvoip_mod
            if self.logger.isEnabledFor(logging.DEBUG):
                _pyvoip_mod.DEBUG = True
                self.logger.debug("pyVoIP packet debug enabled")

            self._phone = VoIPPhone(
                server=self.sip_account.domain,
                port=self.sip_account.port,
                username=self.sip_account.username,
                password=self.sip_account.password,
                callCallback=self._on_incoming_call,
                myIP=my_ip,
                sipPort=sip_port,
                rtpPortLow=rtp_low,
                rtpPortHigh=rtp_high,
            )
            self._phone.start()
            self.running = True

            # Wait up to 10 s for registration, logging status each second
            last_status = None
            for i in range(20):
                await asyncio.sleep(0.5)
                status = self._phone.get_status()
                if status != last_status:
                    self.logger.info(f"SIP status: {status} ({i * 0.5:.1f}s)")
                    last_status = status
                if status == PhoneStatus.REGISTERED:
                    self.registered = True
                    self.logger.info(
                        f"SIP registered: {self.sip_account.username}@{self.sip_account.domain}"
                    )
                    return True
                if status == PhoneStatus.FAILED:
                    self.logger.error(
                        f"SIP registration failed for "
                        f"{self.sip_account.username}@{self.sip_account.domain}"
                    )
                    self._phone.stop()
                    self._phone = None
                    return False

            self.logger.error(
                f"SIP registration timeout (10s) — last status: {last_status}"
            )
            self._phone.stop()
            self._phone = None
            return False

        except Exception as e:
            self.logger.error(f"Failed to start VoIP service: {e}", exc_info=True)
            self._phone = None
            return False

    # ------------------------------------------------------------------
    # Call management
    # ------------------------------------------------------------------

    async def make_call(self, destination: str) -> Optional[str]:
        """Place an outgoing call. Returns a call_id on success."""
        if not self._phone or not self.registered:
            self.logger.error(
                f"Cannot make call - not registered "
                f"(phone={'set' if self._phone else 'None'}, "
                f"registered={self.registered}, "
                f"user={self.sip_account.username}@{self.sip_account.domain})"
            )
            return None

        try:
            dial_number = self._normalize_number(destination)
            self.logger.info(f"Dialing {dial_number}...")

            pyvoip_call = self._phone.call(dial_number)
            call_id = str(uuid.uuid4())
            self.active_calls[call_id] = _ActiveCall(
                call_id=call_id,
                remote_uri=destination,
                state=CallState.DIALING,
                pyvoip_call=pyvoip_call,
            )
            self.logger.info(f"Call {call_id} initiated to {destination}")
            return call_id

        except Exception as e:
            self.logger.error(f"Failed to call {destination}: {e}")
            return None

    async def answer_call(self, call_id: str) -> bool:
        """Answer an incoming call."""
        entry = self.active_calls.get(call_id)
        if not entry:
            self.logger.error(f"answer_call: unknown call_id {call_id}")
            return False

        try:
            if entry.pyvoip_call:
                entry.pyvoip_call.answer()
            entry.state = CallState.CONNECTED
            entry.start_time = time.time()
            self.logger.info(f"Answered call {call_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error answering {call_id}: {e}")
            return False

    async def hangup_call(self, call_id: str) -> bool:
        """Hang up an active call.

        Handles all pyVoIP call states: ANSWERED (normal hangup), DIALING
        (cancel before remote answers), and already-ENDED (no-op cleanup).
        """
        entry = self.active_calls.get(call_id)
        if not entry:
            self.logger.warning(f"hangup_call: unknown call_id {call_id}")
            return False

        try:
            self.stop_audio_bridge(call_id)
            self.stop_call_monitor(call_id)
            if entry.rtp_stream:
                entry.rtp_stream.stop()
                entry.rtp_stream = None
            if entry.pyvoip_call:
                try:
                    entry.pyvoip_call.hangup()
                except Exception:
                    # hangup() fails if not ANSWERED — force-stop RTP
                    try:
                        for rtp in getattr(entry.pyvoip_call, 'RTPClients', []):
                            rtp.stop()
                    except Exception:
                        pass
            entry.state = CallState.DISCONNECTED
            self.active_calls.pop(call_id, None)
            self.logger.info(f"Hung up call {call_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error hanging up {call_id}: {e}")
            self.active_calls.pop(call_id, None)
            return False

    # ------------------------------------------------------------------
    # Audio bridge
    # ------------------------------------------------------------------

    def start_audio_bridge(self, call_id: str,
                           audio_in_callback: Callable[[np.ndarray], None],
                           audio_out_callback: Callable[[], Optional[np.ndarray]]) -> bool:
        """Start the audio bridge thread for a call.

        audio_in_callback: receives PCM int16 array from the remote party
        audio_out_callback: returns PCM int16 array from the handset mic
        """
        entry = self.active_calls.get(call_id)
        if not entry:
            self.logger.error(f"start_audio_bridge: unknown call_id {call_id}")
            return False

        entry.audio_in_callback = audio_in_callback
        entry.audio_out_callback = audio_out_callback
        entry.bridge_active = True

        entry.bridge_thread = threading.Thread(
            target=self._audio_bridge_worker,
            args=(entry,),
            daemon=True,
            name=f"audio-bridge-{call_id[:8]}",
        )
        entry.bridge_thread.start()
        self.logger.info(f"Audio bridge started for {call_id}")
        return True

    def stop_audio_bridge(self, call_id: str) -> None:
        """Stop the audio bridge thread for a call."""
        entry = self.active_calls.get(call_id)
        if not entry:
            return

        entry.bridge_active = False
        if entry.bridge_thread and entry.bridge_thread.is_alive():
            entry.bridge_thread.join(timeout=2.0)
        self.logger.info(f"Audio bridge stopped for {call_id}")

    def _audio_bridge_worker(self, entry: _ActiveCall) -> None:
        """Shuttle audio between pyVoIP call and handset callbacks.

        G.711 at 8 kHz: 20 ms frame = 160 bytes (8-bit PCM, 1 byte/sample).
        """
        CHUNK = 160  # bytes = samples (1 byte/sample at 8-bit)
        while entry.bridge_active and entry.pyvoip_call:
            try:
                raw = entry.pyvoip_call.read_audio(CHUNK, blocking=False)
                if raw and entry.audio_in_callback:
                    pcm_u8 = np.frombuffer(raw, dtype=np.uint8)
                    pcm16 = (pcm_u8.astype(np.int16) - 128) * 256
                    entry.audio_in_callback(pcm16)

                if entry.audio_out_callback:
                    pcm_out = entry.audio_out_callback()
                    if pcm_out is not None:
                        pcm_u8_out = ((pcm_out.astype(np.int16) >> 8) + 128).astype(np.uint8)
                        entry.pyvoip_call.write_audio(pcm_u8_out.tobytes())
            except Exception as e:
                self.logger.debug(f"Audio bridge error (call {entry.call_id[:8]}): {e}")
            time.sleep(0.02)

    # ------------------------------------------------------------------
    # Call state monitor
    # ------------------------------------------------------------------

    def start_call_monitor(self, call_id: str,
                           on_state_change: Callable[[str, CallState], None]) -> None:
        """Poll pyVoIP call state and fire *on_state_change* on transitions."""
        entry = self.active_calls.get(call_id)
        if not entry:
            self.logger.error(f"start_call_monitor: unknown call_id {call_id}")
            return

        entry.on_state_change = on_state_change
        entry.monitor_active = True
        entry.monitor_thread = threading.Thread(
            target=self._call_monitor_worker,
            args=(entry,),
            daemon=True,
            name=f"call-monitor-{call_id[:8]}",
        )
        entry.monitor_thread.start()
        self.logger.info(f"Call state monitor started for {call_id}")

    def stop_call_monitor(self, call_id: str) -> None:
        """Stop the call state monitor thread."""
        entry = self.active_calls.get(call_id)
        if not entry:
            return
        entry.monitor_active = False
        if entry.monitor_thread and entry.monitor_thread.is_alive():
            entry.monitor_thread.join(timeout=2.0)

    def _call_monitor_worker(self, entry: _ActiveCall) -> None:
        """Poll pyVoIP call state every 200 ms and map to local CallState."""
        _PYVOIP_MAP = {}
        if HAS_PYVOIP:
            _PYVOIP_MAP = {
                PyVoIPCallState.DIALING: CallState.DIALING,
                PyVoIPCallState.RINGING: CallState.RINGING,
                PyVoIPCallState.ANSWERED: CallState.CONNECTED,
                PyVoIPCallState.ENDED: CallState.DISCONNECTED,
            }

        prev_state = entry.state
        while entry.monitor_active and entry.pyvoip_call:
            try:
                pv_state = entry.pyvoip_call.state
                local = _PYVOIP_MAP.get(pv_state, entry.state)
                if local != prev_state:
                    self.logger.info(
                        f"Call {entry.call_id[:8]} state: {prev_state.value} -> {local.value}"
                    )
                    entry.state = local
                    prev_state = local
                    # Start our custom RTP stream when call connects
                    if local == CallState.CONNECTED:
                        self._start_rtp_stream(entry)
                    if entry.on_state_change:
                        try:
                            entry.on_state_change(entry.call_id, local)
                        except Exception as cb_err:
                            self.logger.error(f"State change callback error: {cb_err}")
                    if local == CallState.DISCONNECTED:
                        entry.monitor_active = False
                        break
            except Exception as e:
                self.logger.debug(f"Monitor error (call {entry.call_id[:8]}): {e}")
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # Custom RTP stream lifecycle
    # ------------------------------------------------------------------

    def _start_rtp_stream(self, entry: _ActiveCall) -> bool:
        """Start our custom RTP stream using params extracted by patch 7.

        pyVoIP's answered() parsed the SDP and created RTPClients, but
        patch 7 stopped them and stashed the connection params on
        ``pyvoip_call._rtp_params``.  We bind our RTPStream to the same
        local port and send to the same remote endpoint.
        """
        rtp_params_list = getattr(entry.pyvoip_call, '_rtp_params', None)
        if not rtp_params_list:
            self.logger.warning("No RTP params from pyVoIP — using legacy audio path")
            return False

        params = rtp_params_list[0]  # first audio stream
        try:
            stream = RTPStream(
                local_ip=params['local_ip'],
                local_port=params['local_port'],
                remote_ip=params['remote_ip'],
                remote_port=params['remote_port'],
                ssrc=params['ssrc'],
            )
            stream.start()
            entry.rtp_stream = stream
            self.logger.info(
                f"Custom RTP stream: {params['local_ip']}:{params['local_port']} "
                f"→ {params['remote_ip']}:{params['remote_port']}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to start RTP stream: {e}")
            return False

    # ------------------------------------------------------------------
    # Direct RTP read/write (used by AudioProcessor bridge callbacks)
    # ------------------------------------------------------------------

    def read_call_audio(self, call_id: str, length: int = 160) -> Optional[np.ndarray]:
        """Read int16 audio from the RTP stream.

        Uses our custom RTPStream (direct µ-law decode) when available,
        falling back to pyVoIP's legacy uint8 path otherwise.
        Returns *None* when no data is available.
        """
        entry = self.active_calls.get(call_id)
        if not entry:
            return None

        # Custom RTP stream — already returns int16
        if entry.rtp_stream:
            return entry.rtp_stream.read()

        # Legacy fallback: pyVoIP uint8 path
        if not entry.pyvoip_call:
            return None
        try:
            raw = entry.pyvoip_call.read_audio(length, blocking=False)
            if raw:
                pcm_u8 = np.frombuffer(raw, dtype=np.uint8)
                return (pcm_u8.astype(np.int16) - 128) * 256
        except Exception:
            pass
        return None

    def write_call_audio(self, call_id: str, pcm: np.ndarray) -> bool:
        """Write int16 PCM to the RTP stream.

        Applies a frame-level silence gate with lookback + hangover to
        suppress ambient noise while preserving word onsets and tails.

        Uses our custom RTPStream (direct int16→µ-law) when available,
        falling back to pyVoIP's legacy uint8 path otherwise.
        """
        entry = self.active_calls.get(call_id)
        if not entry:
            return False
        try:
            # Frame-level silence gate with lookback + hangover
            rms = math.sqrt(float(np.mean(pcm.astype(np.float32) ** 2)))
            if rms >= self._rtp_silence_threshold:
                if self._silence_hangover == 0 and self._silence_lookback:
                    for buffered in self._silence_lookback:
                        self._write_rtp(entry, buffered)
                    self._silence_lookback.clear()
                self._silence_hangover = self._silence_hangover_frames
            elif self._silence_hangover > 0:
                self._silence_hangover -= 1
            else:
                # Hangover expired — buffer for lookback, send silence
                self._silence_lookback.append(pcm.copy())
                if len(self._silence_lookback) > self._silence_lookback_size:
                    self._silence_lookback.pop(0)
                self._write_rtp(entry, np.zeros(len(pcm), dtype=np.int16))
                return True

            self._write_rtp(entry, pcm)
            return True
        except Exception:
            return False

    def _write_rtp(self, entry: _ActiveCall, pcm: np.ndarray) -> None:
        """Write int16 PCM to the active RTP path (custom or legacy)."""
        if entry.rtp_stream:
            entry.rtp_stream.write(pcm)
        elif entry.pyvoip_call:
            pcm_u8 = ((pcm.astype(np.int16) >> 8) + 128).astype(np.uint8)
            entry.pyvoip_call.write_audio(pcm_u8.tobytes())

    # ------------------------------------------------------------------
    # Callbacks and status
    # ------------------------------------------------------------------

    def register_call_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for call lifecycle events (e.g. 'incoming')."""
        if event == "incoming":
            self._incoming_call_callback = callback

    def get_active_calls(self) -> List[CallInfo]:
        return [
            CallInfo(
                call_id=e.call_id,
                remote_uri=e.remote_uri,
                state=e.state,
                duration=e.get_duration(),
                start_time=e.start_time,
            )
            for e in self.active_calls.values()
        ]

    def get_registration_status(self) -> Dict[str, Any]:
        status = "not_started"
        if self._phone:
            try:
                status = self._phone.get_status().value
            except Exception:
                status = "unknown"
        return {
            "registered": self.registered,
            "username": self.sip_account.username,
            "domain": self.sip_account.domain,
            "status": status,
        }

    def cleanup(self) -> None:
        """Stop all calls and shut down pyVoIP."""
        self.running = False

        for call_id in list(self.active_calls.keys()):
            self.stop_audio_bridge(call_id)
            entry = self.active_calls.get(call_id)
            if entry:
                if entry.rtp_stream:
                    entry.rtp_stream.stop()
                if entry.pyvoip_call:
                    try:
                        entry.pyvoip_call.hangup()
                    except Exception:
                        pass
        self.active_calls.clear()

        if self._phone:
            try:
                self._phone.stop()
            except Exception as e:
                self.logger.error(f"Error stopping VoIP phone: {e}")
            self._phone = None

        self.registered = False
        self.logger.info("VoIP client cleanup complete")

    def test_connection(self) -> Dict[str, Any]:
        if not HAS_PYVOIP:
            return {"status": "disabled", "error": "pyVoIP library not available"}

        return {
            "status": "available" if self._phone else "idle",
            "registered": self.registered,
            "account_configured": bool(self.sip_account.username),
            "active_calls": len(self.active_calls),
            "registration_info": self.get_registration_status(),
        }


async def main():
    """Standalone connectivity test."""
    logging.basicConfig(level=logging.INFO)
    client = VoIPClient()
    result = client.test_connection()
    print("Connection test:", result)
    if result["account_configured"]:
        print("Starting service...")
        if await client.start_service():
            print("Registered successfully")
            await asyncio.sleep(3)
        client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
