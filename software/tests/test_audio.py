"""
Audio Processor Tests - Antique Telephone AI Operator

Tests for audio processing functionality including recording,
playback, and audio bridge operations.
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio_processor import AudioProcessor, AudioState
from utils.config_manager import ConfigManager


class TestAudioProcessor:
    """Test cases for audio processor functionality"""
    
    @pytest.fixture
    def config_manager(self):
        """Create test configuration manager"""
        config = ConfigManager()
        config.set('audio.sample_rate', 44100)
        config.set('audio.channels', 1)
        config.set('audio.chunk_size', 1024)
        config.set('audio.input_device', 'default')
        config.set('audio.output_device', 'default')
        config.set('audio.volume', 0.8)
        return config
    
    @pytest.fixture
    def audio_processor(self, config_manager):
        """Create audio processor instance for testing"""
        return AudioProcessor(config_manager)
    
    def test_audio_processor_initialization(self, audio_processor):
        """Test audio processor initializes correctly"""
        assert audio_processor.sample_rate == 44100
        assert audio_processor.channels == 1
        assert audio_processor.chunk_size == 1024
        assert audio_processor.volume == 0.8
        assert audio_processor.state == AudioState.IDLE
    
    def test_audio_system_test(self, audio_processor):
        """Test audio system testing functionality"""
        test_result = audio_processor.test_audio_system()
        
        assert 'status' in test_result
        assert test_result['status'] in ['available', 'disabled']
        
        if test_result['status'] == 'available':
            assert 'sample_rate' in test_result
            assert 'channels' in test_result
            assert 'chunk_size' in test_result
            assert 'devices' in test_result
        
        audio_processor.cleanup()
    
    def test_audio_levels(self, audio_processor):
        """Test audio level monitoring"""
        levels = audio_processor.get_audio_levels()
        
        assert 'input_level' in levels
        assert 'output_level' in levels
        assert 'sample_rate' in levels
        assert 'channels' in levels
        
        assert levels['sample_rate'] == 44100
        assert levels['channels'] == 1
    
    def test_process_input_audio(self, audio_processor):
        """Test audio input processing"""
        # Create test audio data
        test_audio = np.random.randint(-1000, 1000, 1024, dtype=np.int16)
        
        # Process the audio
        processed = audio_processor._process_input_audio(test_audio)
        
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.int16
        assert len(processed) == len(test_audio)
        
        # Check that very quiet signals are gated
        quiet_audio = np.full(1024, 50, dtype=np.int16)  # Below noise threshold
        processed_quiet = audio_processor._process_input_audio(quiet_audio)
        
        # Should be mostly zeros due to noise gate
        assert np.sum(np.abs(processed_quiet)) < np.sum(np.abs(quiet_audio))
    
    @patch('core.audio_processor.HAS_PYAUDIO', False)
    def test_audio_disabled_mode(self):
        """Test audio processor when PyAudio is not available"""
        config = ConfigManager()
        processor = AudioProcessor(config)
        
        test_result = processor.test_audio_system()
        assert test_result['status'] == 'disabled'
        assert 'error' in test_result
        
        # Recording should fail gracefully
        assert not processor.start_recording()
        
        # Playback should fail gracefully
        test_audio = np.zeros(1024, dtype=np.int16)
        assert not processor.play_audio(test_audio)
        
        processor.cleanup()
    
    @patch('core.audio_processor.pyaudio.PyAudio')
    def test_start_stop_recording(self, mock_pyaudio, audio_processor):
        """Test recording start/stop functionality"""
        # Mock PyAudio components
        mock_audio_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.return_value = mock_audio_instance
        mock_audio_instance.open.return_value = mock_stream
        
        # Test starting recording
        callback_data = []
        def test_callback(data):
            callback_data.append(data)
        
        if audio_processor.audio:  # Only test if audio is available
            success = audio_processor.start_recording(test_callback)
            
            if success:
                assert audio_processor.state == AudioState.RECORDING
                assert audio_processor.recording_callback == test_callback
                
                # Test stopping recording
                audio_processor.stop_recording()
                assert audio_processor.state == AudioState.IDLE
                assert audio_processor.recording_callback is None
    
    def test_play_audio_with_volume(self, audio_processor):
        """Test audio playback with volume control"""
        # Create test tone
        duration = 0.1  # Short duration for testing
        t = np.linspace(0, duration, int(audio_processor.sample_rate * duration))
        test_tone = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
        
        # Test playback (will use simulation if no hardware)
        result = audio_processor.play_audio(test_tone)
        
        # Should return True even in simulation mode
        assert isinstance(result, bool)
    
    def test_noise_gate_applied_from_config(self):
        """_process_input_audio respects noise_gate config threshold."""
        config = ConfigManager()
        config.set('audio.noise_gate', 500)
        config.set('audio.enable_noise_gate', True)
        processor = AudioProcessor(config)

        # Samples below 500 should be zeroed
        audio = np.array([100, 200, 499, 500, 1000], dtype=np.int16)
        result = processor._process_input_audio(audio)
        assert result[0] == 0   # 100 < 500 → zeroed
        assert result[1] == 0   # 200 < 500 → zeroed
        assert result[2] == 0   # 499 < 500 → zeroed
        assert result[3] != 0   # 500 == threshold → passes (condition is strict <)
        assert result[4] != 0   # 1000 > 500 → passes

    def test_noise_gate_disabled(self):
        """Noise gate does not modify audio when disabled."""
        config = ConfigManager()
        config.set('audio.noise_gate', 500)
        config.set('audio.enable_noise_gate', False)
        processor = AudioProcessor(config)

        audio = np.array([100, 200, 300], dtype=np.int16)
        result = processor._process_input_audio(audio)
        assert result[0] != 0  # should pass through unchanged

    def test_input_gain_applied(self):
        """_process_input_audio applies input_gain multiplier."""
        config = ConfigManager()
        config.set('audio.input_gain', 2.0)
        config.set('audio.enable_noise_gate', False)
        processor = AudioProcessor(config)

        audio = np.array([100, 200], dtype=np.int16)
        result = processor._process_input_audio(audio)
        assert result[0] == 200
        assert result[1] == 400

    def test_record_to_file_no_audio(self, audio_processor):
        """record_to_file returns False when PyAudio is not initialised."""
        audio_processor.audio = None  # simulate no hardware
        result = audio_processor.record_to_file("test_recording.wav", 0.1)
        assert result is False

    def test_record_to_file_writes_wav(self, tmp_path):
        """record_to_file produces a valid WAV when audio hardware is available."""
        import wave as wave_mod

        mock_audio = MagicMock()
        mock_stream = MagicMock()
        mock_stream.read.return_value = b'\x00' * 2048  # 1024 int16 samples of silence
        mock_audio.open.return_value = mock_stream

        processor = AudioProcessor()
        processor.audio = mock_audio  # inject mock PyAudio instance
        processor.sample_rate = 44100
        processor.channels = 1
        processor.chunk_size = 1024

        output = tmp_path / "recording.wav"

        with patch.object(processor, 'start_recording', wraps=lambda cb: (cb(np.zeros(1024, dtype=np.int16)), True)[1]), \
             patch.object(processor, 'stop_recording'):
            result = processor.record_to_file(str(output), 0.05)

        assert result is True
        assert output.exists()
        with wave_mod.open(str(output), 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    def test_play_audio_file_missing_file(self, audio_processor):
        """play_audio_file returns False for a non-existent file."""
        result = audio_processor.play_audio_file("/nonexistent/path/audio.wav")
        assert result is False

    def test_play_audio_file_valid(self, tmp_path):
        """play_audio_file plays a valid WAV file without error."""
        import wave as wave_mod

        # Write a minimal silent WAV
        wav_path = tmp_path / "test.wav"
        with wave_mod.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b'\x00' * 88200)  # 1s silence

        processor = AudioProcessor()
        with patch.object(processor, 'play_wav_bytes', return_value=True) as mock_play:
            result = processor.play_audio_file(str(wav_path))

        assert result is True
        mock_play.assert_called_once()
    
    def test_audio_bridge_lifecycle(self, audio_processor):
        """Test audio bridge start/stop lifecycle"""
        bridge_in_data = []
        bridge_out_data = []
        
        def mock_voip_in():
            return np.zeros(512, dtype=np.int16) if bridge_out_data else None
        
        def mock_voip_out(data):
            bridge_in_data.append(data)
        
        # Test starting bridge
        result = audio_processor.start_audio_bridge(mock_voip_in, mock_voip_out)
        
        if result:
            assert audio_processor.state == AudioState.BRIDGE_ACTIVE
            
            # Stop bridge
            audio_processor.stop_audio_bridge()
            assert audio_processor.state == AudioState.IDLE
    
    def test_configuration_override(self):
        """Test audio processor with custom configuration"""
        config = ConfigManager()
        config.set('audio.sample_rate', 22050)
        config.set('audio.channels', 2)
        config.set('audio.chunk_size', 512)
        config.set('audio.volume', 0.5)
        
        processor = AudioProcessor(config)
        
        assert processor.sample_rate == 22050
        assert processor.channels == 2
        assert processor.chunk_size == 512
        assert processor.volume == 0.5
        
        processor.cleanup()
    
    def test_cleanup(self, audio_processor):
        """Test audio processor cleanup"""
        # Start some operations
        audio_processor.start_recording()
        
        # Cleanup should not raise exceptions
        audio_processor.cleanup()
        
        # State should be reset
        assert audio_processor.state == AudioState.IDLE
    
    @pytest.mark.asyncio
    async def test_async_audio_operations(self, audio_processor):
        """Test audio operations work with async code"""
        # Test that audio operations don't block async execution
        start_time = time.time()
        
        # Create test audio
        test_audio = np.zeros(1024, dtype=np.int16)
        
        # These operations should complete quickly
        audio_processor.play_audio(test_audio)
        levels = audio_processor.get_audio_levels()
        
        elapsed = time.time() - start_time
        
        # Should complete very quickly (not blocking)
        assert elapsed < 1.0
        assert 'input_level' in levels


@pytest.mark.hardware
class TestAudioHardware:
    """Hardware-specific audio tests"""
    
    @pytest.fixture
    def audio_processor(self):
        """Create audio processor for hardware testing"""
        config = ConfigManager()
        return AudioProcessor(config)
    
    def test_real_audio_devices(self, audio_processor):
        """Test real audio device detection"""
        test_result = audio_processor.test_audio_system()
        
        if test_result['status'] == 'available':
            devices = test_result['devices']
            assert isinstance(devices, dict)
            
            # Should have at least one audio device
            assert len(devices) > 0
            
            # Check device structure
            for device_id, device_info in devices.items():
                assert 'name' in device_info
                assert 'inputs' in device_info
                assert 'outputs' in device_info
        
        audio_processor.cleanup()
    
    @pytest.mark.skipif(
        not AudioProcessor(ConfigManager()).test_audio_system()['status'] == 'available',
        reason="Requires working audio system"
    )
    def test_real_audio_recording(self):
        """Test real audio recording on hardware"""
        config = ConfigManager()
        processor = AudioProcessor(config)
        
        recorded_data = []
        
        def collect_audio(data):
            recorded_data.append(data)
        
        # Start recording
        if processor.start_recording(collect_audio):
            # Record for a short time
            time.sleep(0.5)
            
            # Stop recording
            processor.stop_recording()
            
            # Should have collected some data
            assert len(recorded_data) > 0
            
            # Data should be numpy arrays
            for data in recorded_data:
                assert isinstance(data, np.ndarray)
                assert data.dtype == np.int16
        
        processor.cleanup()
    
    @pytest.mark.skipif(
        not AudioProcessor(ConfigManager()).test_audio_system()['status'] == 'available',
        reason="Requires working audio system"
    )
    def test_real_audio_playback(self):
        """Test real audio playback on hardware"""
        config = ConfigManager()
        processor = AudioProcessor(config)
        
        # Generate test tone
        duration = 0.5
        sample_rate = processor.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A note
        test_tone = (np.sin(2 * np.pi * frequency * t) * 16384).astype(np.int16)
        
        # Play the tone
        result = processor.play_audio(test_tone)
        assert result is True
        
        processor.cleanup()


def test_main_function():
    """Test the main function runs without errors"""
    # Import and run main
    from core.audio_processor import main
    
    # Should complete without errors
    try:
        main()
    except KeyboardInterrupt:
        # Expected when running interactively
        pass


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])