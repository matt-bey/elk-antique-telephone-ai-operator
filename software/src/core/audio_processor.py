"""
Audio Processor Module - Antique Telephone AI Operator

Handles audio input/output processing, including microphone capture,
speaker output, and audio bridge management for VoIP calls.
Supports both modern USB audio (Phase 1) and antique hardware (Phase 4).
"""

import io
import logging
import math
import wave
import numpy as np
import threading
import time
import queue
from typing import Optional, Callable, Dict, Any, List
from enum import Enum

from scipy.signal import resample_poly  # legacy — used by tests and non-streaming callers

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logging.warning("PyAudio not available - audio functionality disabled")

from utils.config_manager import ConfigManager

# VoIP audio runs at 8 kHz (G.711); local audio at the device sample rate
# (typically 44100 Hz).  These helpers convert between the two using
# rational resampling so the integer ratio is exact and no drift accumulates.
VOIP_SAMPLE_RATE = 8000


def resample_to_8k(pcm: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample PCM int16 from *source_rate* down to 8 kHz for RTP.

    Uses stateless ``resample_poly`` — suitable for one-shot conversions
    (tests, file processing) but NOT for streaming audio.  For the live
    call bridge, use ``soxr.ResampleStream`` which carries filter state
    between chunks and eliminates edge artifacts.
    """
    if source_rate == VOIP_SAMPLE_RATE:
        return pcm
    g = math.gcd(VOIP_SAMPLE_RATE, source_rate)
    return resample_poly(pcm.astype(np.float32), VOIP_SAMPLE_RATE // g, source_rate // g).astype(np.int16)


def resample_from_8k(pcm: np.ndarray, target_rate: int) -> np.ndarray:
    """Resample 8 kHz PCM int16 up to *target_rate* for the local speaker.

    Stateless — see ``resample_to_8k`` docstring for streaming vs. one-shot.
    """
    if target_rate == VOIP_SAMPLE_RATE:
        return pcm
    g = math.gcd(target_rate, VOIP_SAMPLE_RATE)
    return resample_poly(pcm.astype(np.float32), target_rate // g, VOIP_SAMPLE_RATE // g).astype(np.int16)


class AudioState(Enum):
    """Audio system states"""
    IDLE = "idle"
    RECORDING = "recording"
    PLAYING = "playing"
    BRIDGE_ACTIVE = "bridge_active"


class AudioProcessor:
    """
    Audio processing service for antique telephone system
    
    Manages audio capture, playback, and real-time bridging for VoIP calls.
    Designed to work with both modern USB audio and antique telephone hardware.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize audio processor with configuration"""
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Audio configuration
        self.sample_rate = self.config.get('audio.sample_rate', 44100)
        self.channels = self.config.get('audio.channels', 1)
        self.chunk_size = self.config.get('audio.chunk_size', 1024)
        self.format = pyaudio.paInt16 if HAS_PYAUDIO else None

        # Device configuration
        self.input_device = self.config.get('audio.input_device', 'default')
        self.output_device = self.config.get('audio.output_device', 'default')
        self.volume = self.config.get('audio.volume', 0.8)

        # Processing configuration
        self.input_gain = self.config.get('audio.input_gain', 1.0)
        self.noise_gate_threshold = self.config.get('audio.noise_gate', 100)
        self.noise_gate_enabled = self.config.get('audio.enable_noise_gate', True)
        
        # PyAudio instance
        self.audio: Optional[pyaudio.PyAudio] = None
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        # State management
        self.state = AudioState.IDLE
        self.recording_callback: Optional[Callable] = None
        self.audio_buffer = queue.Queue()
        
        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        self.bridge_thread: Optional[threading.Thread] = None
        self.stop_processing = False
        
        # Initialize audio system
        self._initialize_audio()
    
    def _initialize_audio(self) -> None:
        """Initialize PyAudio system"""
        if not HAS_PYAUDIO:
            self.logger.warning("PyAudio not available - audio system disabled")
            return
        
        try:
            self.audio = pyaudio.PyAudio()
            self.logger.info("PyAudio initialized successfully")
            
            # Log available devices
            self._log_audio_devices()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PyAudio: {e}")
            self.audio = None
    
    def _log_audio_devices(self) -> None:
        """Log available audio devices for debugging"""
        if not self.audio:
            return
        
        self.logger.info("Available audio devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            self.logger.info(
                f"  Device {i}: {device_info['name']} "
                f"(In: {device_info['maxInputChannels']}, "
                f"Out: {device_info['maxOutputChannels']})"
            )
    
    def _get_device_index(self, device_name: str, is_input: bool) -> Optional[int]:
        """Get device index by name or use default"""
        if not self.audio or device_name == 'default':
            return None
        
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_name.lower() in device_info['name'].lower():
                channels = device_info['maxInputChannels' if is_input else 'maxOutputChannels']
                if channels > 0:
                    return i
        
        self.logger.warning(f"Device '{device_name}' not found, using default")
        return None
    
    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """Start audio recording"""
        if not self.audio:
            self.logger.error("Audio system not available")
            return False
        
        if self.state != AudioState.IDLE:
            self.logger.warning(f"Cannot start recording in state: {self.state}")
            return False
        
        try:
            input_device_index = self._get_device_index(self.input_device, True)
            
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._input_callback
            )
            
            self.recording_callback = callback
            self.state = AudioState.RECORDING
            self.input_stream.start_stream()
            
            self.logger.info("Audio recording started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self) -> None:
        """Stop audio recording"""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        
        self.recording_callback = None
        if self.state == AudioState.RECORDING:
            self.state = AudioState.IDLE
        
        self.logger.info("Audio recording stopped")
    
    def _input_callback(self, in_data, frame_count, time_info, status):
        """PyAudio input stream callback"""
        if status:
            self.logger.warning(f"Audio input status: {status}")
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Apply volume control and basic processing
        processed_data = self._process_input_audio(audio_data)
        
        # Queue for further processing
        if not self.audio_buffer.full():
            self.audio_buffer.put(processed_data)
        
        # Call registered callback if available
        if self.recording_callback:
            try:
                self.recording_callback(processed_data)
            except Exception as e:
                self.logger.error(f"Error in recording callback: {e}")
        
        return (None, pyaudio.paContinue)
    
    def _process_input_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Process input audio data: gain, noise gate, clip prevention."""
        processed = audio_data.astype(np.float32)

        # Apply input gain
        if self.input_gain != 1.0:
            processed = processed * self.input_gain

        # Noise gate: zero out samples below threshold to suppress background hiss
        if self.noise_gate_enabled and self.noise_gate_threshold > 0:
            processed[np.abs(processed) < self.noise_gate_threshold] = 0

        # Clip prevention
        max_val = np.max(np.abs(processed))
        if max_val > 32767:
            processed = processed * (32767 / max_val)

        return processed.astype(np.int16)
    
    def play_audio(self, audio_data: np.ndarray) -> bool:
        """Play audio data through speakers"""
        if not self.audio:
            self.logger.error("Audio system not available")
            return False
        
        try:
            output_device_index = self._get_device_index(self.output_device, False)
            
            # Apply volume control
            volume_adjusted = (audio_data * self.volume).astype(np.int16)
            
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=output_device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.state = AudioState.PLAYING
            stream.write(volume_adjusted.tobytes())
            stream.stop_stream()
            stream.close()
            
            if self.state == AudioState.PLAYING:
                self.state = AudioState.IDLE
            
            self.logger.debug("Audio playback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to play audio: {e}")
            return False
    
    def play_wav_bytes(self, data: bytes) -> bool:
        """Play WAV-formatted bytes through the output device.

        Parses the WAV header for sample rate / format so Piper models
        (22050 Hz) play correctly regardless of the processor's default rate.
        """
        if not self.audio:
            self.logger.error("Audio system not available")
            return False

        try:
            buf = io.BytesIO(data)
            with wave.open(buf, "rb") as wf:
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())

            pa_format = self.audio.get_format_from_width(sample_width)
            output_device_index = self._get_device_index(self.output_device, False)

            stream = self.audio.open(
                format=pa_format,
                channels=channels,
                rate=sample_rate,
                output=True,
                output_device_index=output_device_index,
                frames_per_buffer=self.chunk_size,
            )
            # Apply volume and write in chunks to avoid PortAudio buffer deadlocks
            audio_array = np.frombuffer(frames, dtype=np.int16)
            adjusted = (audio_array * self.volume).astype(np.int16).tobytes()
            chunk_bytes = self.chunk_size * channels * sample_width
            for offset in range(0, len(adjusted), chunk_bytes):
                stream.write(adjusted[offset:offset + chunk_bytes])
            stream.stop_stream()
            stream.close()
            self.logger.debug(f"WAV playback complete ({len(data)} bytes, {sample_rate} Hz)")
            return True

        except Exception as e:
            self.logger.error(f"WAV playback failed: {e}")
            return False

    def play_audio_file(self, file_path: str) -> bool:
        """Load a WAV file and play it through the output device."""
        try:
            with wave.open(file_path, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, 'wb') as wf:
                with wave.open(file_path, 'rb') as src:
                    wf.setparams(src.getparams())
                    wf.writeframes(raw)
            self.logger.info(f"Playing audio file: {file_path}")
            return self.play_wav_bytes(wav_bytes.getvalue())
        except Exception as e:
            self.logger.error(f"Failed to play audio file {file_path}: {e}")
            return False

    def record_to_file(self, file_path: str, duration: float) -> bool:
        """Record microphone input for `duration` seconds and save as WAV."""
        if not self.audio:
            self.logger.error("Audio system not available")
            return False

        try:
            frames: list = []

            def collect(chunk: np.ndarray) -> None:
                frames.append(chunk.tobytes())

            if not self.start_recording(collect):
                return False

            time.sleep(duration)
            self.stop_recording()

            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))

            self.logger.info(f"Audio recorded to {file_path} ({duration}s)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to record to file {file_path}: {e}")
            return False
    
    def start_audio_bridge(self, voip_audio_in: Callable, voip_audio_out: Callable) -> bool:
        """Start audio bridge for VoIP calls"""
        if self.state != AudioState.IDLE:
            self.logger.warning(f"Cannot start bridge in state: {self.state}")
            return False
        
        self.state = AudioState.BRIDGE_ACTIVE
        self.stop_processing = False
        
        # Start bridge processing thread
        self.bridge_thread = threading.Thread(
            target=self._audio_bridge_worker,
            args=(voip_audio_in, voip_audio_out),
            daemon=True
        )
        self.bridge_thread.start()
        
        self.logger.info("Audio bridge started")
        return True
    
    def stop_audio_bridge(self) -> None:
        """Stop audio bridge"""
        self.stop_processing = True
        
        if self.bridge_thread:
            self.bridge_thread.join(timeout=2.0)
            self.bridge_thread = None
        
        if self.state == AudioState.BRIDGE_ACTIVE:
            self.state = AudioState.IDLE
        
        self.logger.info("Audio bridge stopped")
    
    def _audio_bridge_worker(self, voip_in: Callable, voip_out: Callable) -> None:
        """Audio bridge worker thread.

        Shuttles audio between the local mic/speaker (via PyAudio) and
        the VoIP call (via voip_in/voip_out closures).

        **Outbound (mic → RTP):** Mic produces ``chunk_size`` samples at
        ``sample_rate`` (e.g. 1024 @ 44.1 kHz).  We accumulate in
        ``out_buf`` and drain in ``rtp_frame_device`` chunks (882 @ 44.1k)
        so the closure's ``soxr.ResampleStream`` produces ~160 samples at
        8 kHz — one clean RTP frame per call with no remainder padding.
        The frame alignment keeps RTP packets evenly sized; the stateful
        resampler eliminates FIR filter edge artifacts between chunks.

        **Inbound (RTP → speaker):** ``voip_in`` returns variable-length
        resampled audio.  Accumulate in ``in_buf`` and write full
        ``chunk_size`` blocks to the speaker to prevent underruns.
        """
        if not self.audio:
            self.logger.error("Audio bridge failed: PyAudio not initialized")
            self.state = AudioState.IDLE
            return

        try:
            input_device_index = self._get_device_index(self.input_device, True)
            output_device_index = self._get_device_index(self.output_device, False)

            input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk_size
            )

            output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=output_device_index,
                frames_per_buffer=self.chunk_size
            )

            input_stream.start_stream()
            output_stream.start_stream()

            # Frame size for outbound RTP alignment: feed exactly
            # rtp_frame_device samples to voip_out so the soxr resampler
            # produces ~160 samples at 8 kHz (one clean RTP frame).
            #   44100 Hz: 160 * 44100/8000 = 882 samples
            #   48000 Hz: 160 * 48000/8000 = 960 samples
            rtp_frame_device = int(160 * self.sample_rate / VOIP_SAMPLE_RATE)

            out_buf = np.array([], dtype=np.int16)  # mic → RTP (device rate)
            in_buf = np.array([], dtype=np.int16)   # RTP → speaker (device rate)

            while not self.stop_processing:
                # --- Outbound: mic → noise gate → accumulate → RTP ---
                # Read all available mic data to prevent buffer buildup
                # (output_stream.write blocks ~23ms per chunk).
                while input_stream.get_read_available() >= self.chunk_size:
                    mic_data = input_stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_array = np.frombuffer(mic_data, dtype=np.int16)
                    processed = self._process_input_audio(audio_array)
                    out_buf = np.concatenate([out_buf, processed])

                # Drain in rtp_frame_device chunks (882 @ 44.1k) so each
                # voip_out call produces ~160 samples at 8 kHz after
                # resampling — one clean RTP frame with no remainder.
                while len(out_buf) >= rtp_frame_device:
                    frame = out_buf[:rtp_frame_device]
                    out_buf = out_buf[rtp_frame_device:]
                    voip_out(frame)

                # --- Inbound: RTP → accumulate → speaker ---
                # Read up to 5 frames per iteration to prevent latency
                # buildup without risking a spin loop (the custom
                # RTPStream returns None when empty, but pyVoIP's legacy
                # path may return silence-filled bytes).
                for _ in range(5):
                    voip_data = voip_in()
                    if voip_data is None or len(voip_data) == 0:
                        break
                    in_buf = np.concatenate([in_buf, voip_data])

                # Write full chunk_size blocks to the speaker to avoid
                # underruns from sub-buffer writes.
                while len(in_buf) >= self.chunk_size:
                    frame = in_buf[:self.chunk_size]
                    in_buf = in_buf[self.chunk_size:]
                    volume_adjusted = (frame * self.volume).astype(np.int16)
                    output_stream.write(volume_adjusted.tobytes(), exception_on_underflow=False)

                time.sleep(0.001)

            input_stream.stop_stream()
            output_stream.stop_stream()
            input_stream.close()
            output_stream.close()

        except Exception as e:
            self.logger.error(f"Audio bridge error: {e}")

        self.logger.info("Audio bridge worker stopped")
    
    def get_audio_levels(self) -> Dict[str, float]:
        """Return current audio configuration and live levels.

        input_level/output_level are populated in Phase 3 when a continuous
        stream is active; they report 0.0 outside of an audio bridge session.
        """
        return {
            "input_level": 0.0,
            "output_level": 0.0,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }
    
    def test_audio_system(self) -> Dict[str, Any]:
        """Test audio system and return status"""
        if not HAS_PYAUDIO or not self.audio:
            return {
                "status": "disabled",
                "error": "PyAudio not available",
                "devices": {}
            }
        
        devices = {}
        try:
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                devices[i] = {
                    "name": device_info['name'],
                    "inputs": device_info['maxInputChannels'],
                    "outputs": device_info['maxOutputChannels'],
                    "sample_rate": device_info['defaultSampleRate']
                }
        except Exception as e:
            self.logger.error(f"Error enumerating devices: {e}")
        
        return {
            "status": "available",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "devices": devices,
            "current_state": self.state.value
        }
    
    def cleanup(self) -> None:
        """Clean up audio resources"""
        self.stop_processing = True
        
        if self.input_stream:
            self.stop_recording()
        
        if self.bridge_thread:
            self.stop_audio_bridge()
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        self.logger.info("Audio processor cleanup completed")


def main():
    """Test audio processor functionality"""
    logging.basicConfig(level=logging.INFO)
    
    audio_processor = AudioProcessor()
    
    try:
        print("Audio processor test running...")
        test_result = audio_processor.test_audio_system()
        print("Test result:", test_result)
        
        if test_result["status"] == "available":
            print("Testing 3-second recording...")
            if audio_processor.record_to_file("test_recording.wav", 3.0):
                print("Recording test completed")
            
            print("Testing audio playback...")
            # Generate test tone
            duration = 1.0
            sample_rate = audio_processor.sample_rate
            frequency = 440  # A note
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_tone = (np.sin(2 * np.pi * frequency * t) * 16384).astype(np.int16)
            
            if audio_processor.play_audio(test_tone):
                print("Playback test completed")
        
    except KeyboardInterrupt:
        print("\nShutting down audio processor...")
    finally:
        audio_processor.cleanup()


if __name__ == "__main__":
    main()