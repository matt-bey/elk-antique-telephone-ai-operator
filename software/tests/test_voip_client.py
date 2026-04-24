"""
VoIP Client Tests - Antique Telephone AI Operator

Tests for VoIPClient backed by pyVoIP.  All pyVoIP objects are mocked so
these tests run without a live SIP server.
"""

import asyncio
import threading
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.voip_client import VoIPClient, CallState, CallInfo, _ActiveCall
from utils.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(username="17775551234", password="secret", domain="sip.callcentric.net"):
    """Return a VoIPClient with minimal SIP config, no real pyVoIP phone."""
    config = ConfigManager()
    config.set("sip.username", username)
    config.set("sip.password", password)
    config.set("sip.domain", domain)
    return VoIPClient(config)


def _mock_pyvoip_call():
    """Return a mock pyVoIP VoIPCall-like object."""
    c = MagicMock()
    c.answer = MagicMock()
    c.hangup = MagicMock()
    # pyVoIP returns 8-bit signed PCM (1 byte/sample)
    c.read_audio = MagicMock(return_value=bytes(160))  # 160 samples at 8-bit
    c.write_audio = MagicMock()
    # Keep deprecated names for backward compat tests
    c.readAudio = c.read_audio
    c.writeAudio = c.write_audio
    c.request = MagicMock()
    c.request.headers = {"From": "sip:caller@example.com", "Call-ID": "test-call-id"}
    return c


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------

class TestTestConnection:
    def test_disabled_when_pyvoip_unavailable(self):
        client = _make_client()
        with patch("core.voip_client.HAS_PYVOIP", False):
            result = client.test_connection()
        assert result["status"] == "disabled"
        assert "pyVoIP" in result["error"]

    def test_idle_before_start(self):
        client = _make_client()
        result = client.test_connection()
        assert result["status"] == "idle"
        assert result["registered"] is False
        assert result["account_configured"] is True

    def test_no_account_configured(self):
        client = _make_client(username="")
        result = client.test_connection()
        assert result["account_configured"] is False


# ---------------------------------------------------------------------------
# start_service
# ---------------------------------------------------------------------------

class TestStartService:
    @pytest.mark.asyncio
    async def test_returns_false_when_pyvoip_unavailable(self):
        client = _make_client()
        with patch("core.voip_client.HAS_PYVOIP", False):
            result = await client.start_service()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_username(self):
        client = _make_client(username="")
        result = await client.start_service()
        assert result is False

    @pytest.mark.asyncio
    async def test_successful_registration(self):
        client = _make_client()
        mock_phone = MagicMock()

        from pyVoIP.VoIP import PhoneStatus
        mock_phone.get_status.return_value = PhoneStatus.REGISTERED

        with patch("core.voip_client.VoIPPhone", return_value=mock_phone):
            result = await client.start_service()

        assert result is True
        assert client.registered is True
        mock_phone.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_registration_failure(self):
        client = _make_client()
        mock_phone = MagicMock()

        from pyVoIP.VoIP import PhoneStatus
        mock_phone.get_status.return_value = PhoneStatus.FAILED

        with patch("core.voip_client.VoIPPhone", return_value=mock_phone):
            result = await client.start_service()

        assert result is False
        assert client.registered is False

    @pytest.mark.asyncio
    async def test_exception_during_start(self):
        client = _make_client()
        with patch("core.voip_client.VoIPPhone", side_effect=RuntimeError("port in use")):
            result = await client.start_service()
        assert result is False


# ---------------------------------------------------------------------------
# make_call
# ---------------------------------------------------------------------------

class TestMakeCall:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_registered(self):
        client = _make_client()
        assert client.registered is False
        result = await client.make_call("5551234567")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_call_id_on_success(self):
        client = _make_client()
        client.registered = True

        mock_phone = MagicMock()
        mock_phone.call.return_value = _mock_pyvoip_call()
        client._phone = mock_phone

        call_id = await client.make_call("5551234567")

        assert call_id is not None
        assert call_id in client.active_calls
        assert client.active_calls[call_id].state == CallState.DIALING

    @pytest.mark.asyncio
    async def test_strips_formatting_and_prepends_1_for_10_digit(self):
        client = _make_client()
        client.registered = True

        mock_phone = MagicMock()
        mock_phone.call.return_value = _mock_pyvoip_call()
        client._phone = mock_phone

        await client.make_call("(555) 123-4567")
        # 10-digit US numbers get "1" prepended for Callcentric
        mock_phone.call.assert_called_once_with("15551234567")

    @pytest.mark.asyncio
    async def test_prepends_area_code_and_1_for_7_digit(self):
        client = _make_client()
        client.config.set("sip.local_area_code", "614")
        client.registered = True

        mock_phone = MagicMock()
        mock_phone.call.return_value = _mock_pyvoip_call()
        client._phone = mock_phone

        await client.make_call("555-1234")
        mock_phone.call.assert_called_once_with("16145551234")

    @pytest.mark.asyncio
    async def test_7_digit_without_area_code_passes_through(self):
        client = _make_client()
        # No local_area_code configured
        client.registered = True

        mock_phone = MagicMock()
        mock_phone.call.return_value = _mock_pyvoip_call()
        client._phone = mock_phone

        await client.make_call("555-1234")
        # Without area code config, 7 digits pass through as-is
        mock_phone.call.assert_called_once_with("5551234")

    @pytest.mark.asyncio
    async def test_11_digit_passes_through(self):
        client = _make_client()
        client.registered = True

        mock_phone = MagicMock()
        mock_phone.call.return_value = _mock_pyvoip_call()
        client._phone = mock_phone

        await client.make_call("1-614-555-1234")
        mock_phone.call.assert_called_once_with("16145551234")

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        client = _make_client()
        client.registered = True
        mock_phone = MagicMock()
        mock_phone.call.side_effect = RuntimeError("SIP error")
        client._phone = mock_phone

        result = await client.make_call("5551234567")
        assert result is None


# ---------------------------------------------------------------------------
# hangup_call
# ---------------------------------------------------------------------------

class TestHangupCall:
    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_call(self):
        client = _make_client()
        result = await client.hangup_call("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_hangs_up_and_removes_call(self):
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        call_id = "test-call-123"
        client.active_calls[call_id] = _ActiveCall(
            call_id=call_id,
            remote_uri="5551234567",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )

        result = await client.hangup_call(call_id)

        assert result is True
        pyvoip_call.hangup.assert_called_once()
        assert call_id not in client.active_calls


# ---------------------------------------------------------------------------
# answer_call
# ---------------------------------------------------------------------------

class TestAnswerCall:
    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_call(self):
        client = _make_client()
        result = await client.answer_call("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_answers_and_sets_connected(self):
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        call_id = "incoming-001"
        client.active_calls[call_id] = _ActiveCall(
            call_id=call_id,
            remote_uri="sip:caller@example.com",
            state=CallState.RINGING,
            pyvoip_call=pyvoip_call,
        )

        result = await client.answer_call(call_id)

        assert result is True
        pyvoip_call.answer.assert_called_once()
        assert client.active_calls[call_id].state == CallState.CONNECTED


# ---------------------------------------------------------------------------
# start_audio_bridge / stop_audio_bridge
# ---------------------------------------------------------------------------

class TestAudioBridge:
    def test_returns_false_for_unknown_call(self):
        client = _make_client()
        result = client.start_audio_bridge("nonexistent", lambda x: None, lambda: None)
        assert result is False

    def test_starts_bridge_thread(self):
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        call_id = "bridge-test"
        client.active_calls[call_id] = _ActiveCall(
            call_id=call_id,
            remote_uri="5550000",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )

        received = []
        result = client.start_audio_bridge(
            call_id,
            audio_in_callback=received.append,
            audio_out_callback=lambda: None,
        )

        assert result is True
        entry = client.active_calls[call_id]
        assert entry.bridge_active is True
        assert entry.bridge_thread is not None
        assert entry.bridge_thread.is_alive()

        # Clean up
        client.stop_audio_bridge(call_id)

    def test_stop_bridge_terminates_thread(self):
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        call_id = "bridge-stop-test"
        client.active_calls[call_id] = _ActiveCall(
            call_id=call_id,
            remote_uri="5550001",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )

        client.start_audio_bridge(call_id, lambda x: None, lambda: None)
        thread = client.active_calls[call_id].bridge_thread

        client.stop_audio_bridge(call_id)

        thread.join(timeout=3.0)
        assert not thread.is_alive()

    def test_audio_in_callback_receives_pcm(self):
        """read_audio bytes are passed to audio_in_callback as numpy int16 array."""
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        # 160 samples of unsigned 8-bit PCM (170 = 42 above silence center of 128)
        expected_bytes = np.full(160, 170, dtype=np.uint8).tobytes()
        pyvoip_call.read_audio.return_value = expected_bytes

        call_id = "audio-in-test"
        client.active_calls[call_id] = _ActiveCall(
            call_id=call_id,
            remote_uri="5550002",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )

        received = []
        client.start_audio_bridge(call_id, audio_in_callback=received.append, audio_out_callback=lambda: None)

        # Wait a moment for the bridge thread to process at least one chunk
        time.sleep(0.1)
        client.stop_audio_bridge(call_id)

        assert len(received) > 0
        assert isinstance(received[0], np.ndarray)
        assert received[0].dtype == np.int16


# ---------------------------------------------------------------------------
# incoming call callback
# ---------------------------------------------------------------------------

class TestIncomingCall:
    def test_incoming_call_stored_and_callback_fired(self):
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()

        fired = []
        client.register_call_callback("incoming", lambda cid, uri: fired.append((cid, uri)))

        client._on_incoming_call(pyvoip_call)

        assert len(client.active_calls) == 1
        entry = next(iter(client.active_calls.values()))
        assert entry.state == CallState.RINGING
        assert len(fired) == 1
        assert fired[0][0] == entry.call_id


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_stops_phone_and_clears_calls(self):
        client = _make_client()
        mock_phone = MagicMock()
        client._phone = mock_phone
        client.registered = True

        # Add a fake active call
        call_id = "cleanup-call"
        pyvoip_call = _mock_pyvoip_call()
        client.active_calls[call_id] = _ActiveCall(
            call_id=call_id,
            remote_uri="5550003",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )

        client.cleanup()

        mock_phone.stop.assert_called_once()
        assert len(client.active_calls) == 0
        assert client.registered is False
        assert client._phone is None

    def test_cleanup_handles_no_phone(self):
        client = _make_client()
        assert client._phone is None
        client.cleanup()  # Should not raise


# ---------------------------------------------------------------------------
# get_active_calls
# ---------------------------------------------------------------------------

class TestGetActiveCalls:
    def test_returns_call_info_list(self):
        client = _make_client()
        call_id = "info-test"
        client.active_calls[call_id] = _ActiveCall(
            call_id=call_id,
            remote_uri="sip:test@example.com",
            state=CallState.CONNECTED,
        )

        calls = client.get_active_calls()
        assert len(calls) == 1
        assert isinstance(calls[0], CallInfo)
        assert calls[0].call_id == call_id
        assert calls[0].state == CallState.CONNECTED


# ---------------------------------------------------------------------------
# read_call_audio / write_call_audio
# ---------------------------------------------------------------------------

class TestDirectRTP:
    def test_read_returns_none_for_unknown_call(self):
        client = _make_client()
        assert client.read_call_audio("nonexistent") is None

    def test_read_returns_numpy_int16_scaled_from_uint8(self):
        """read_call_audio should decode unsigned 8-bit PCM (center=128) to int16."""
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        # Simulate pyVoIP returning 160 bytes of unsigned 8-bit PCM
        # Value 170 (42 above center of 128) should become int16 42*256 = 10752
        pcm_u8 = np.full(160, 170, dtype=np.uint8)
        pyvoip_call.read_audio.return_value = pcm_u8.tobytes()
        entry = _ActiveCall(
            call_id="rtp-read",
            remote_uri="test",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )
        client.active_calls["rtp-read"] = entry

        result = client.read_call_audio("rtp-read")
        assert result is not None
        assert result.dtype == np.int16
        assert len(result) == 160
        # uint8 170 → (170 - 128) * 256 = 42 * 256 = 10752
        assert result[0] == 10752

    def test_read_silence_is_zero(self):
        """Silence (uint8 128) should decode to int16 0."""
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        pcm_u8 = np.full(160, 128, dtype=np.uint8)
        pyvoip_call.read_audio.return_value = pcm_u8.tobytes()
        entry = _ActiveCall(
            call_id="rtp-silence",
            remote_uri="test",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )
        client.active_calls["rtp-silence"] = entry

        result = client.read_call_audio("rtp-silence")
        assert result is not None
        np.testing.assert_array_equal(result, np.zeros(160, dtype=np.int16))

    def test_read_returns_none_when_no_data(self):
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        pyvoip_call.read_audio.return_value = b""
        entry = _ActiveCall(
            call_id="rtp-empty",
            remote_uri="test",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )
        client.active_calls["rtp-empty"] = entry
        assert client.read_call_audio("rtp-empty") is None

    def test_write_returns_false_for_unknown_call(self):
        client = _make_client()
        assert client.write_call_audio("nonexistent", np.zeros(160, dtype=np.int16)) is False

    def test_write_scales_int16_to_uint8_centered_128(self):
        """write_call_audio should convert int16 to unsigned 8-bit centered at 128."""
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        entry = _ActiveCall(
            call_id="rtp-write",
            remote_uri="test",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )
        client.active_calls["rtp-write"] = entry

        # int16 10752 → (10752 >> 8) + 128 = 42 + 128 = 170 uint8
        pcm = np.full(160, 10752, dtype=np.int16)
        assert client.write_call_audio("rtp-write", pcm) is True
        pyvoip_call.write_audio.assert_called_once()
        written_bytes = pyvoip_call.write_audio.call_args[0][0]
        assert len(written_bytes) == 160  # 160 samples at 8-bit = 160 bytes
        # Verify first byte is 170 (42 above center)
        assert written_bytes[0] == 170

    def test_write_silence_is_128(self):
        """int16 silence (0) should encode to uint8 128."""
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        entry = _ActiveCall(
            call_id="rtp-silence-w",
            remote_uri="test",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )
        client.active_calls["rtp-silence-w"] = entry

        pcm = np.zeros(160, dtype=np.int16)
        assert client.write_call_audio("rtp-silence-w", pcm) is True
        written_bytes = pyvoip_call.write_audio.call_args[0][0]
        assert written_bytes[0] == 128


# ---------------------------------------------------------------------------
# call state monitor
# ---------------------------------------------------------------------------

class TestCallStateMonitor:
    def test_monitor_detects_answered(self):
        """Monitor should fire callback when pyVoIP state changes to ANSWERED."""
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()

        # Simulate state progression: DIALING → ANSWERED
        states = iter([MagicMock(name="DIALING"), MagicMock(name="ANSWERED")])
        # We need real pyVoIP enums for the mapping, so patch HAS_PYVOIP
        from unittest.mock import PropertyMock

        entry = _ActiveCall(
            call_id="monitor-test",
            remote_uri="test",
            state=CallState.DIALING,
            pyvoip_call=pyvoip_call,
        )
        client.active_calls["monitor-test"] = entry

        transitions = []

        with patch('core.voip_client.HAS_PYVOIP', True), \
             patch('core.voip_client.PyVoIPCallState') as MockPyState:
            MockPyState.DIALING = "DIALING"
            MockPyState.ANSWERED = "ANSWERED"
            MockPyState.ENDED = "ENDED"
            MockPyState.RINGING = "RINGING"

            # pyvoip_call.state cycles through DIALING then ANSWERED
            pyvoip_call.state = "DIALING"

            def on_change(cid, state):
                transitions.append((cid, state))
                # Stop monitoring after first transition
                entry.monitor_active = False

            client.start_call_monitor("monitor-test", on_change)
            time.sleep(0.1)

            # Simulate the remote answering
            pyvoip_call.state = "ANSWERED"
            time.sleep(0.5)

            client.stop_call_monitor("monitor-test")

        assert len(transitions) >= 1
        assert transitions[0][1] == CallState.CONNECTED

    def test_monitor_detects_disconnect(self):
        """Monitor should fire callback and stop when pyVoIP state becomes ENDED."""
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()

        entry = _ActiveCall(
            call_id="disconnect-test",
            remote_uri="test",
            state=CallState.CONNECTED,
            pyvoip_call=pyvoip_call,
        )
        client.active_calls["disconnect-test"] = entry

        transitions = []

        with patch('core.voip_client.HAS_PYVOIP', True), \
             patch('core.voip_client.PyVoIPCallState') as MockPyState:
            MockPyState.DIALING = "DIALING"
            MockPyState.ANSWERED = "ANSWERED"
            MockPyState.ENDED = "ENDED"
            MockPyState.RINGING = "RINGING"

            pyvoip_call.state = "ANSWERED"

            def on_change(cid, state):
                transitions.append((cid, state))

            client.start_call_monitor("disconnect-test", on_change)
            time.sleep(0.1)

            # Remote hangs up
            pyvoip_call.state = "ENDED"
            time.sleep(0.5)

            client.stop_call_monitor("disconnect-test")

        assert any(s == CallState.DISCONNECTED for _, s in transitions)
        assert entry.monitor_active is False


# ---------------------------------------------------------------------------
# hangup non-ANSWERED states
# ---------------------------------------------------------------------------

class TestHangupDialing:
    @pytest.mark.asyncio
    async def test_hangup_dialing_call_does_not_raise(self):
        """Hanging up a DIALING call should succeed even if pyVoIP hangup fails."""
        client = _make_client()
        pyvoip_call = _mock_pyvoip_call()
        pyvoip_call.hangup.side_effect = Exception("Not in ANSWERED state")
        pyvoip_call.RTPClients = [MagicMock()]

        entry = _ActiveCall(
            call_id="dialing-hangup",
            remote_uri="test",
            state=CallState.DIALING,
            pyvoip_call=pyvoip_call,
        )
        client.active_calls["dialing-hangup"] = entry

        result = await client.hangup_call("dialing-hangup")
        assert result is True
        assert "dialing-hangup" not in client.active_calls
        # Verify RTP clients were stopped as fallback
        pyvoip_call.RTPClients[0].stop.assert_called_once()
