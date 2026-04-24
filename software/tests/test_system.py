"""
System Integration Tests - Antique Telephone AI Operator

Tests for overall system integration and end-to-end functionality.
"""

import pytest
import asyncio
import time
import signal
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import AntiquePhoneSystem
from utils.config_manager import ConfigManager
from core.gpio_monitor import GPIOEvent
from core.ai_operator import AIOperator, CallRequest, OperatorState


class TestAntiquePhoneSystem:
    """Test cases for complete system integration"""
    
    @pytest.fixture
    def config_manager(self):
        """Create test configuration manager"""
        config = ConfigManager()
        # Set test configuration
        config.set('audio.sample_rate', 44100)
        config.set('gpio.crank_pin', 18)
        config.set('anthropic.api_key', 'test_key')
        config.set('system.log_level', 'DEBUG')
        return config
    
    @pytest.fixture
    def phone_system(self, config_manager, tmp_path):
        """Create phone system instance for testing"""
        # Use temporary config path
        config_path = tmp_path / "config"
        config_path.mkdir()
        
        system = AntiquePhoneSystem(str(config_path))
        system.config = config_manager  # Override with test config
        return system
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, phone_system):
        """Test system initialization"""
        # Mock hardware components to avoid dependencies
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            result = await phone_system.initialize()
            
            # Should initialize successfully in simulation mode
            assert result is True
            assert phone_system.gpio_monitor is not None
            assert phone_system.audio_processor is not None
            assert phone_system.ai_operator is not None
            # VoIP client may be None if no SIP config
            
            await phone_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_with_audio_available(self, phone_system):
        """Test system with audio system available"""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', True), \
             patch('core.audio_processor.AudioProcessor.test_audio_system') as mock_audio_test, \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            # Mock audio system as available
            mock_audio_test.return_value = {'status': 'available'}
            
            result = await phone_system.initialize()
            assert result is True
            
            await phone_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_with_audio_unavailable(self, phone_system):
        """Test system when audio system unavailable"""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', True), \
             patch('core.audio_processor.AudioProcessor.test_audio_system') as mock_audio_test, \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            # Mock audio system as unavailable
            mock_audio_test.return_value = {'status': 'disabled'}
            
            result = await phone_system.initialize()
            assert result is True  # Should succeed in degraded mode without audio
    
    @pytest.mark.asyncio
    async def test_crank_turn_event_flow(self, phone_system):
        """Test complete crank turn event flow"""
        # Initialize system in test mode
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            await phone_system.initialize()
            
            # Mock AI operator methods
            phone_system.ai_operator.reset_conversation = Mock()
            phone_system.ai_operator.handle_operator_session = AsyncMock()

            # Earpiece must be off-hook for crank to work
            phone_system.gpio_monitor.hook_state = True

            # Simulate crank turn
            await phone_system._on_crank_turn()
            
            # Verify AI operator was called
            phone_system.ai_operator.reset_conversation.assert_called_once()
            phone_system.ai_operator.handle_operator_session.assert_called_once()
            
            await phone_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_hook_state_changes(self, phone_system):
        """Test hook state change handling"""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            await phone_system.initialize()
            
            # Test hook off (phone picked up)
            await phone_system._on_hook_off()
            # Should not raise errors
            
            # Test hook on (phone hung up)
            phone_system.audio_processor.stop_recording = Mock()
            phone_system.ai_operator.reset_conversation = Mock()
            
            await phone_system._on_hook_on()
            
            # Verify cleanup methods were called
            phone_system.audio_processor.stop_recording.assert_called_once()
            phone_system.ai_operator.reset_conversation.assert_called_once()
            
            await phone_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_call_request_handling(self, phone_system):
        """Test call request processing"""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            await phone_system.initialize()
            
            # Test without VoIP client
            call_request = CallRequest(
                requested_number="555-1234",
                caller_intent="Call 555-1234",
                confidence=0.8,
                timestamp=time.time()
            )
            
            await phone_system._on_call_request(call_request)
            # Should log warning but not crash
            
            await phone_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_call_request_with_voip(self, phone_system):
        """Test call request with VoIP client available"""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            # Add SIP configuration to enable VoIP
            phone_system.config.set('sip.username', 'test_user')
            phone_system.config.set('sip.password', 'test_pass')
            phone_system.config.set('sip.domain', 'test.com')
            
            await phone_system.initialize()
            
            # Mock VoIP client
            phone_system.voip_client = Mock()
            phone_system.voip_client.make_call = AsyncMock(return_value='call_123')
            phone_system.voip_client.start_call_monitor = Mock()
            phone_system.voip_client.hangup_call = AsyncMock()

            call_request = CallRequest(
                requested_number="555-1234",
                caller_intent="Call 555-1234",
                confidence=0.8,
                timestamp=time.time()
            )

            await phone_system._on_call_request(call_request)

            # Verify call was placed and monitor started
            phone_system.voip_client.make_call.assert_called_once_with("555-1234")
            phone_system.voip_client.start_call_monitor.assert_called_once()

            assert phone_system.in_call is True
            assert phone_system.current_call_id == 'call_123'
            
            await phone_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_signal_handling(self, phone_system):
        """Test signal handling for graceful shutdown"""
        # Test signal handler
        phone_system._signal_handler(signal.SIGTERM, None)
        assert phone_system.running is False
    
    @pytest.mark.asyncio
    async def test_system_startup_and_shutdown(self, phone_system):
        """Test complete system startup and shutdown cycle"""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            # Mock the main loop to exit quickly
            original_start = phone_system.start
            
            async def quick_start():
                await phone_system.initialize()
                phone_system.running = True
                phone_system.gpio_monitor.start_monitoring()
                # Exit immediately instead of running main loop
                phone_system.running = False
                await phone_system.shutdown()
            
            phone_system.start = quick_start
            
            # Should complete without errors
            await phone_system.start()
    
    def test_configuration_validation(self, phone_system):
        """Test configuration validation"""
        # Test with invalid configuration
        phone_system.config.set('audio.volume', 1.5)  # Invalid volume > 1.0
        phone_system.config.set('gpio.crank_pin', 18)
        phone_system.config.set('gpio.hook_pin', 18)  # Duplicate pin
        
        validation = phone_system.config.validate_config()
        
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_operator_response_callbacks(self, phone_system):
        """Test operator response callback functions"""
        with patch('core.gpio_monitor.HAS_GPIO', False), \
             patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'):
            
            await phone_system.initialize()
            
            # Test text response callback
            phone_system._on_operator_response("Test response")
            # Should not raise errors
            
            # Test speech callback  
            test_audio = b"fake audio data"
            await phone_system._on_operator_speech(test_audio)
            # Should not raise errors
            
            await phone_system.shutdown()


class TestSTTFailureRecovery:
    """Tests for consecutive STT failure handling in _process_user_audio."""

    def _make_system(self):
        """Return a minimal AntiquePhoneSystem with a mocked AI operator."""
        system = AntiquePhoneSystem()
        op = AIOperator()
        op.state = OperatorState.LISTENING
        # Stub out synthesize_speech so it never calls TTS
        op.synthesize_speech = AsyncMock(return_value=None)
        system.ai_operator = op
        system.audio_processor = MagicMock()
        system.audio_processor.sample_rate = 44100
        return system

    @pytest.mark.asyncio
    async def test_single_stt_failure_sends_nudge(self):
        """First STT failure → operator sends a stt_trouble phrase and stays in LISTENING."""
        system = self._make_system()
        system.ai_operator.stt_provider = MagicMock()
        system.ai_operator.stt_provider.is_available = True
        system.ai_operator.process_speech = AsyncMock(return_value=None)

        responses = []
        system._on_operator_response = responses.append
        system._on_operator_speech = AsyncMock()

        await system._process_user_audio(np.zeros(1024, dtype=np.int16))

        assert system._stt_failures == 1
        assert system.ai_operator.state == OperatorState.LISTENING
        assert len(responses) == 1
        assert responses[0] in AIOperator._PHRASES["stt_trouble"]

    @pytest.mark.asyncio
    async def test_three_stt_failures_ends_session(self):
        """Three consecutive STT failures → stt_give_up phrase, state reset to IDLE."""
        system = self._make_system()
        system.ai_operator.process_speech = AsyncMock(return_value=None)
        system._stt_failures = 2  # already at 2 consecutive failures

        responses = []
        system._on_operator_response = responses.append
        system._on_operator_speech = AsyncMock()

        await system._process_user_audio(np.zeros(1024, dtype=np.int16))

        assert system._stt_failures == 3
        assert system.ai_operator.state == OperatorState.IDLE
        assert len(responses) == 1
        assert responses[0] in AIOperator._PHRASES["stt_give_up"]

    @pytest.mark.asyncio
    async def test_successful_transcription_resets_failure_counter(self):
        """Good transcription after a failure resets _stt_failures to 0."""
        system = self._make_system()
        system._stt_failures = 1
        system.ai_operator.process_speech = AsyncMock(return_value="call 555-1234")
        system.ai_operator.process_user_request = AsyncMock()

        system._on_operator_response = lambda _: None
        system._on_operator_speech = AsyncMock()

        await system._process_user_audio(np.zeros(1024, dtype=np.int16))

        assert system._stt_failures == 0

    @pytest.mark.asyncio
    async def test_stt_failures_reset_on_new_crank_session(self):
        """Crank turn resets _stt_failures to 0 at the start of each session."""
        system = self._make_system()
        system._stt_failures = 2

        # Stub out everything so _on_crank_turn returns quickly
        system.in_call = False
        system.running = True
        system.gpio_monitor = MagicMock()
        system.ai_operator.handle_operator_session = AsyncMock(
            side_effect=lambda **kw: setattr(system.ai_operator, "state", OperatorState.IDLE)
        )

        await system._on_crank_turn()

        assert system._stt_failures == 0


@pytest.mark.integration
class TestSystemIntegration:
    """Integration tests with real components"""
    
    @pytest.mark.asyncio
    async def test_real_audio_integration(self):
        """Test system with real audio components"""
        config = ConfigManager()
        system = AntiquePhoneSystem()
        system.config = config
        
        # Only test if audio is available
        from core.audio_processor import AudioProcessor
        audio_test = AudioProcessor(config).test_audio_system()
        
        if audio_test['status'] == 'available':
            with patch('core.gpio_monitor.HAS_GPIO', False), \
                 patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
                 patch('core.voip_client.HAS_PYVOIP', False):
                
                result = await system.initialize()
                assert result is True
                
                # Test audio functionality
                assert system.audio_processor is not None
                levels = system.audio_processor.get_audio_levels()
                assert 'input_level' in levels
                
                await system.shutdown()
        else:
            pytest.skip("Audio system not available")


@pytest.mark.hardware
class TestSystemHardware:
    """Hardware-specific system tests"""
    
    @pytest.mark.skipif(
        not Path('/proc/cpuinfo').exists() or 
        'Raspberry Pi' not in open('/proc/cpuinfo', 'r').read(),
        reason="Requires Raspberry Pi hardware"
    )
    @pytest.mark.asyncio
    async def test_raspberry_pi_integration(self):
        """Test system integration on Raspberry Pi"""
        config = ConfigManager()
        system = AntiquePhoneSystem()
        system.config = config
        
        # Test GPIO functionality
        with patch('core.audio_processor.HAS_PYAUDIO', False), \
             patch('main.WhisperProvider'), patch('main.PiperProvider'), patch('main.AnthropicProvider'), \
             patch('core.voip_client.HAS_PYVOIP', False):
            
            result = await system.initialize()
            assert result is True
            
            # Test GPIO monitor
            assert system.gpio_monitor is not None
            gpio_status = system.gpio_monitor.test_inputs()
            assert gpio_status['mode'] == 'hardware'
            
            await system.shutdown()


def test_main_function_test_mode():
    """Test main function in test mode"""
    import sys
    from main import main
    
    # Mock command line arguments for test mode
    test_args = ['main.py', '--test']
    with patch.object(sys, 'argv', test_args):
        # Should complete without errors in test mode
        try:
            asyncio.run(main())
        except SystemExit as e:
            # Test mode may exit with 0 or 1
            assert e.code in [0, 1]


def test_main_function_debug_mode():
    """Test main function with debug logging"""
    import sys
    from main import main
    
    # Mock command line arguments for debug mode
    test_args = ['main.py', '--debug', '--test']
    with patch.object(sys, 'argv', test_args):
        # Should complete without errors
        try:
            asyncio.run(main())
        except SystemExit as e:
            assert e.code in [0, 1]


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])