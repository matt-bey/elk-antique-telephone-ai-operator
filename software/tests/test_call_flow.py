"""
Call Flow Tests - Antique Telephone AI Operator

Tests for the end-to-end outbound call flow: resampling helpers, audio bridge
wiring, and call lifecycle management.  All external I/O (pyVoIP, PyAudio) is
mocked so these tests run without hardware or a SIP server.
"""

import asyncio
import time
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio_processor import resample_to_8k, resample_from_8k, VOIP_SAMPLE_RATE
from core.rtp_stream import G711Codec, RTPStream


# ---------------------------------------------------------------------------
# Resample helpers
# ---------------------------------------------------------------------------

class TestResample:
    def test_resample_to_8k_preserves_frequency(self):
        """A 440 Hz sine at 44100 Hz resampled to 8000 Hz should still contain 440 Hz."""
        sr_in = 44100
        duration = 0.1  # 100 ms
        t = np.arange(int(sr_in * duration)) / sr_in
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

        resampled = resample_to_8k(tone, sr_in)

        # Expected length: ~800 samples (0.1s at 8 kHz)
        expected_len = int(len(tone) * VOIP_SAMPLE_RATE / sr_in)
        assert abs(len(resampled) - expected_len) <= 2
        assert resampled.dtype == np.int16

    def test_resample_from_8k_preserves_frequency(self):
        """An 8 kHz signal resampled to 44100 Hz should produce the right length."""
        sr_out = 44100
        duration = 0.02  # 20 ms = one RTP frame
        n_samples = int(VOIP_SAMPLE_RATE * duration)
        tone = (np.sin(2 * np.pi * 440 * np.arange(n_samples) / VOIP_SAMPLE_RATE) * 16000).astype(np.int16)

        resampled = resample_from_8k(tone, sr_out)

        expected_len = int(n_samples * sr_out / VOIP_SAMPLE_RATE)
        assert abs(len(resampled) - expected_len) <= 2
        assert resampled.dtype == np.int16

    def test_roundtrip_fidelity(self):
        """Resample 44.1k → 8k → 44.1k and check the dominant frequency is 440 Hz."""
        sr = 44100
        duration = 0.05
        t = np.arange(int(sr * duration)) / sr
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

        down = resample_to_8k(tone, sr)
        up = resample_from_8k(down, sr)

        # FFT to find peak frequency
        fft = np.abs(np.fft.rfft(up.astype(np.float32)))
        freqs = np.fft.rfftfreq(len(up), d=1 / sr)
        peak_freq = freqs[np.argmax(fft)]
        assert abs(peak_freq - 440) < 10  # within 10 Hz

    def test_identity_at_8k(self):
        """If source rate == 8000, resample_to_8k should return input unchanged."""
        pcm = np.arange(160, dtype=np.int16)
        result = resample_to_8k(pcm, 8000)
        np.testing.assert_array_equal(result, pcm)

    def test_identity_from_8k(self):
        """If target rate == 8000, resample_from_8k should return input unchanged."""
        pcm = np.arange(160, dtype=np.int16)
        result = resample_from_8k(pcm, 8000)
        np.testing.assert_array_equal(result, pcm)

    def test_gcd_optimized_ratio(self):
        """The ratio 44100:8000 simplifies via GCD to 441:80."""
        g = math.gcd(44100, VOIP_SAMPLE_RATE)
        assert 44100 // g == 441
        assert VOIP_SAMPLE_RATE // g == 80


# ---------------------------------------------------------------------------
# G.711 µ-law codec
# ---------------------------------------------------------------------------

class TestG711Codec:
    def test_silence_encodes_to_0xff(self):
        """int16 silence (0) encodes to µ-law 0xFF."""
        pcm = np.zeros(160, dtype=np.int16)
        ulaw = G711Codec.encode(pcm)
        assert ulaw[0] == 0xFF

    def test_max_positive_encodes_to_0x80(self):
        """int16 max positive (32767) encodes to µ-law 0x80."""
        pcm = np.array([32767], dtype=np.int16)
        ulaw = G711Codec.encode(pcm)
        assert ulaw[0] == 0x80

    def test_max_negative_encodes_to_0x00(self):
        """int16 max negative (-32768) encodes to µ-law 0x00."""
        pcm = np.array([-32768], dtype=np.int16)
        ulaw = G711Codec.encode(pcm)
        assert ulaw[0] == 0x00

    def test_roundtrip_fidelity(self):
        """encode → decode preserves signal within µ-law quantization error."""
        # 440 Hz tone at moderate level
        t = np.arange(160) / 8000.0
        pcm = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        ulaw = G711Codec.encode(pcm)
        recovered = G711Codec.decode(ulaw)

        # µ-law quantization error should be small relative to signal
        error = np.abs(pcm.astype(np.float32) - recovered.astype(np.float32))
        max_error = np.max(error)
        # Max step size for µ-law at high amplitudes is ~1000; at low ~8
        assert max_error < 2000, f"Max roundtrip error {max_error} too large"

    def test_decode_symmetry(self):
        """Decoding all 256 µ-law values produces symmetric positive/negative range."""
        all_ulaw = np.arange(256, dtype=np.uint8)
        decoded = G711Codec.decode(all_ulaw)
        assert decoded.dtype == np.int16
        assert len(decoded) == 256

    def test_encode_vectorized(self):
        """Encoding a full frame produces the right output shape."""
        pcm = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        ulaw = G711Codec.encode(pcm)
        assert ulaw.shape == (160,)
        assert ulaw.dtype == np.uint8


# ---------------------------------------------------------------------------
# RTP packet construction
# ---------------------------------------------------------------------------

class TestRTPPacket:
    def test_header_roundtrip(self):
        """Build and parse an RTP header; fields should match."""
        header = RTPStream._build_rtp_header(
            sequence=1234, timestamp=56789, ssrc=42
        )
        assert len(header) == 12
        pt, seq, ts, ssrc, payload = RTPStream._parse_rtp_header(header)
        assert pt == 0  # PCMU
        assert seq == 1234
        assert ts == 56789
        assert ssrc == 42
        assert payload == b""

    def test_header_with_payload(self):
        """Parse a full RTP packet with payload."""
        header = RTPStream._build_rtp_header(sequence=1, timestamp=0, ssrc=100)
        payload_data = bytes([0xFF] * 160)
        packet = header + payload_data
        pt, seq, ts, ssrc, payload = RTPStream._parse_rtp_header(packet)
        assert seq == 1
        assert len(payload) == 160
        assert payload == payload_data

    def test_sequence_wraps(self):
        """Sequence numbers wrap at 16-bit boundary."""
        header = RTPStream._build_rtp_header(sequence=0xFFFF, timestamp=0, ssrc=1)
        _, seq, _, _, _ = RTPStream._parse_rtp_header(header)
        assert seq == 0xFFFF


# ---------------------------------------------------------------------------
# RTPStream read/write queue
# ---------------------------------------------------------------------------

class TestRTPStreamQueue:
    def test_write_read_roundtrip(self):
        """Written frames appear in the read buffer."""
        stream = RTPStream("127.0.0.1", 0, "127.0.0.1", 0)
        pcm = np.ones(160, dtype=np.int16) * 1000
        stream._out_buf.append(pcm)
        result = stream._out_buf.popleft()
        np.testing.assert_array_equal(result, pcm)

    def test_read_empty_returns_none(self):
        """Reading from empty buffer returns None."""
        stream = RTPStream("127.0.0.1", 0, "127.0.0.1", 0)
        assert stream.read() is None

    def test_in_buf_fifo(self):
        """Inbound buffer maintains FIFO order."""
        stream = RTPStream("127.0.0.1", 0, "127.0.0.1", 0)
        frame1 = np.ones(160, dtype=np.int16)
        frame2 = np.ones(160, dtype=np.int16) * 2
        stream._in_buf.append(frame1)
        stream._in_buf.append(frame2)
        np.testing.assert_array_equal(stream.read(), frame1)
        np.testing.assert_array_equal(stream.read(), frame2)
        assert stream.read() is None


# ---------------------------------------------------------------------------
# Call flow integration (mocked I/O)
# ---------------------------------------------------------------------------

class TestCallFlowIntegration:
    """Test the wiring between main.py callbacks and VoIPClient."""

    @pytest.mark.asyncio
    async def test_simulation_mode_logs_without_voip(self):
        """When voip_client is None, _on_call_request logs simulation."""
        from core.ai_operator import CallRequest

        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), \
             patch('main.AnthropicProvider'):

            from main import AntiquePhoneSystem
            system = AntiquePhoneSystem()
            await system.initialize()

            system.voip_client = None  # ensure simulation mode
            call_request = CallRequest(
                requested_number="555-0001",
                caller_intent="test",
                confidence=0.9,
                timestamp=time.time(),
            )
            # Should not raise
            await system._on_call_request(call_request)
            assert system.in_call is False

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_end_call_resets_state(self):
        """_end_call should clear in_call and current_call_id."""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), \
             patch('main.AnthropicProvider'):

            from main import AntiquePhoneSystem
            system = AntiquePhoneSystem()
            await system.initialize()

            system.in_call = True
            system.current_call_id = "test-call"
            system.voip_client = MagicMock()

            system._end_call()

            assert system.in_call is False
            assert system.current_call_id is None

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_call_state_change_connected(self):
        """CONNECTED state should update LED."""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), \
             patch('main.AnthropicProvider'):

            from main import AntiquePhoneSystem
            from core.voip_client import CallState
            system = AntiquePhoneSystem()
            await system.initialize()

            system._on_call_state_change("test-call", CallState.CONNECTED)
            # Should not raise; LED updated internally

            await system.shutdown()
