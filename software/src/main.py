#!/usr/bin/env python3
"""
Antique Telephone AI Operator - Main Application

Entry point for the antique telephone AI operator system.
Orchestrates all components and manages the main application loop.
"""

import asyncio
import signal
import sys
import time
import argparse
import numpy as np
import math
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.gpio_monitor import GPIOMonitor, GPIOEvent, HAS_GPIO
from core.audio_processor import AudioProcessor
from core.ai_operator import AIOperator, CallRequest, OperatorState
from core.voip_client import VoIPClient, CallState
from core.audio_processor import VOIP_SAMPLE_RATE
import soxr
from core.keyboard_simulator import KeyboardSimulator
from providers.stt.whisper_provider import WhisperProvider
from providers.tts.piper_provider import PiperProvider
from providers.conversation.ollama_provider import OllamaProvider
from providers.conversation.anthropic_provider import AnthropicProvider
from providers.lookup.google_places_provider import GooglePlacesProvider
from utils.config_manager import ConfigManager
from utils.logger import setup_logging, get_component_logger


class AntiquePhoneSystem:
    """
    Main system orchestrator for the antique telephone AI operator
    
    Coordinates all subsystems and manages the overall application state.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the antique phone system"""
        # Set up configuration
        self.config = ConfigManager(config_path)
        
        # Set up logging
        self.logger_instance = setup_logging(self.config)
        self.logger = get_component_logger('main')
        
        # Initialize subsystems
        self.gpio_monitor: Optional[GPIOMonitor] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.ai_operator: Optional[AIOperator] = None
        self.voip_client: Optional[VoIPClient] = None
        self.keyboard_simulator: Optional[KeyboardSimulator] = None
        
        # Application state
        self.running = False
        self.in_call = False
        self.current_call_id: Optional[str] = None
        self._stt_failures: int = 0  # consecutive STT non-results in current session
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # main event loop for thread callbacks
        # Saved bridge closures — created in _on_call_request, started on CONNECTED
        self._bridge_voip_in: Optional[callable] = None
        self._bridge_voip_out: Optional[callable] = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def initialize(self) -> bool:
        """Initialize all subsystems"""
        try:
            self.logger_instance.log_startup_banner()
            self.logger.info("Initializing Antique Telephone AI Operator...")
            
            # Validate configuration
            validation = self.config.validate_config()
            if not validation["valid"]:
                self.logger.error("Configuration validation failed:")
                for error in validation["errors"]:
                    self.logger.error(f"  - {error}")
                return False
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    self.logger.warning(warning)
            
            # Initialize GPIO monitor
            self.logger.info("Initializing GPIO monitor...")
            self.gpio_monitor = GPIOMonitor(self.config)
            # Wrap async handlers so GPIO callbacks (sync) can schedule them safely
            self.gpio_monitor.register_event_handler(
                GPIOEvent.CRANK_TURN,
                lambda: asyncio.create_task(self._on_crank_turn())
            )
            self.gpio_monitor.register_event_handler(
                GPIOEvent.HOOK_OFF,
                lambda: asyncio.create_task(self._on_hook_off())
            )
            self.gpio_monitor.register_event_handler(
                GPIOEvent.HOOK_ON,
                lambda: asyncio.create_task(self._on_hook_on())
            )
            
            # Initialize audio processor
            self.logger.info("Initializing audio processor...")
            self.audio_processor = AudioProcessor(self.config)
            audio_status = self.audio_processor.test_audio_system()
            if audio_status["status"] != "available":
                self.logger.warning("Audio system not available - running in simulation mode")
            
            # Initialize providers
            self.logger.info("Initializing STT provider (Whisper)...")
            stt = WhisperProvider(
                model_name=self.config.get('whisper.model', 'base'),
                language=self.config.get('whisper.language', 'en'),
            )

            self.logger.info("Initializing TTS provider (Piper)...")
            tts = PiperProvider(
                voice_name=self.config.get('tts.voice', 'en_US-lessac-high')
            )

            conv_provider = self.config.get('conversation.provider', 'anthropic')
            self.logger.info(f"Initializing conversation provider ({conv_provider})...")
            if conv_provider == 'ollama':
                conv = OllamaProvider(
                    model=self.config.get('conversation.model', 'llama3.2:1b'),
                    host=self.config.get('conversation.host', 'http://localhost:11434'),
                )
            else:  # default: anthropic
                conv = AnthropicProvider(
                    api_key=self.config.get('anthropic.api_key', ''),
                    model=self.config.get('conversation.model', 'claude-haiku-4-5-20251001'),
                )

            # Initialize lookup providers (optional — requires GOOGLE_PLACES_API_KEY)
            lookup_providers = []
            if self.config.get('lookup.google_api_key', ''):
                self.logger.info("Initializing Google Places lookup provider...")
                lookup_providers.append(GooglePlacesProvider(self.config))
            else:
                self.logger.info("No GOOGLE_PLACES_API_KEY — business name lookup disabled")

            # Initialize AI operator
            self.logger.info("Initializing AI operator...")
            self.ai_operator = AIOperator(
                self.config,
                stt_provider=stt,
                tts_provider=tts,
                conversation_provider=conv,
                lookup_providers=lookup_providers,
            )
            ai_status = self.ai_operator.get_status()
            self.logger.info(f"AI providers available: {ai_status['providers']}")
            
            # Initialize VoIP client (optional)
            if self.config.get('sip.username'):
                self.logger.info("Initializing VoIP client...")
                voip = VoIPClient(self.config)
                if await voip.start_service():
                    self.voip_client = voip
                else:
                    self.logger.warning(
                        "VoIP registration failed - falling back to simulation mode"
                    )
            else:
                self.logger.info("No SIP configuration - VoIP calling disabled")
            
            self.logger.info("System initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def start(self) -> None:
        """Start the main application loop"""
        if not await self.initialize():
            self.logger.error("Failed to initialize system")
            return

        self.running = True
        self._loop = asyncio.get_running_loop()
        self.gpio_monitor.start_monitoring()

        # Set status LED to indicate ready state
        self.gpio_monitor.set_status_led(True)

        # Enable keyboard simulation when no GPIO hardware is present
        if not HAS_GPIO:
            self.keyboard_simulator = KeyboardSimulator(
                self.gpio_monitor,
                on_quit=lambda: setattr(self, 'running', False)
            )
            self.keyboard_simulator.start()

        self.logger.info("Antique Telephone AI Operator is ready")
        self.logger.info("Waiting for crank turn to summon operator...")
        
        try:
            # Main application loop
            while self.running:
                await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all subsystems"""
        self.logger.info("Shutting down Antique Telephone AI Operator...")
        
        self.running = False

        # Stop keyboard simulator first so terminal is restored before other log output
        if self.keyboard_simulator:
            self.keyboard_simulator.stop()
            self.keyboard_simulator = None

        # Hang up any active calls and stop audio bridge
        if self.in_call and self.voip_client and self.current_call_id:
            await self.voip_client.hangup_call(self.current_call_id)
            self._end_call()
        
        # Stop subsystems
        if self.gpio_monitor:
            self.gpio_monitor.stop_monitoring()
            self.gpio_monitor.cleanup()
        
        if self.audio_processor:
            self.audio_processor.cleanup()
        
        if self.voip_client:
            self.voip_client.cleanup()
        
        self.logger.info("Shutdown complete")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle system signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}")
        self.running = False
    
    async def _on_crank_turn(self) -> None:
        """Handle crank turn event — summon the operator and run the conversation loop.

        The earpiece must be off-hook (picked up) for the crank to work.
        The loop continues until the call is placed (CONNECTING_CALL), the operator
        returns to IDLE, or the session safety timeout expires.
        """
        if self.in_call:
            self.logger.info("Crank turned during call - ignoring")
            return

        if not self.gpio_monitor.get_hook_state():
            self.logger.info("Crank turned while on-hook - ignoring (pick up earpiece first)")
            return

        self.logger.info("Crank turned - summoning operator")
        self.gpio_monitor.set_status_led(True, "OPERATOR ACTIVE")

        _TERMINAL_STATES = {OperatorState.CONNECTING_CALL, OperatorState.IDLE, OperatorState.ERROR}
        _SESSION_TIMEOUT = 120.0  # seconds

        try:
            self.ai_operator.reset_conversation()
            self._stt_failures = 0

            await self.ai_operator.handle_operator_session(
                on_response=self._on_operator_response,
                on_speech_audio=self._on_operator_speech,
            )

            session_start = time.time()
            while self.running:
                if self.ai_operator.state in _TERMINAL_STATES:
                    self.logger.info(
                        f"Conversation ended (state={self.ai_operator.state.value})"
                    )
                    if self.ai_operator.state == OperatorState.CONNECTING_CALL:
                        self.gpio_monitor.set_status_led(True, "CONNECTING")
                        # Keep session alive while the VoIP call is active.
                        # The call state monitor fires _handle_call_ended on
                        # remote hangup; _on_hook_on handles local hangup.
                        while self.in_call and self.running:
                            await asyncio.sleep(0.5)
                    else:
                        self.gpio_monitor.set_status_led(True, "READY")
                    break

                elapsed = time.time() - session_start
                if elapsed >= _SESSION_TIMEOUT:
                    self.logger.info(f"Session timeout ({_SESSION_TIMEOUT}s) — ending")
                    self.gpio_monitor.set_status_led(True, "READY")
                    break

                self.gpio_monitor.set_status_led(True, "LISTENING")
                await self._start_listening_session()

        except Exception as e:
            self.logger.error(f"Error handling crank turn: {e}")
            self.gpio_monitor.set_status_led(True, "READY")
    
    async def _on_hook_off(self) -> None:
        """Handle phone picked up"""
        self.logger.info("Phone picked up")
        
        if not self.in_call:
            # If not in call, this starts/continues operator interaction
            self.logger.info("Ready for conversation")
        else:
            # If in call, this might be call answer
            if self.current_call_id and self.voip_client:
                await self.voip_client.answer_call(self.current_call_id)
    
    async def _on_hook_on(self) -> None:
        """Handle phone hung up — terminate any active call and reset."""
        self.logger.info("Phone hung up")

        if self.in_call and self.voip_client and self.current_call_id:
            await self.voip_client.hangup_call(self.current_call_id)
            self._end_call()

        # Stop any audio processing (recording or bridge)
        if self.audio_processor:
            self.audio_processor.stop_recording()
            self.audio_processor.stop_audio_bridge()

        # Reset operator
        if self.ai_operator:
            self.ai_operator.reset_conversation()

        self.gpio_monitor.set_status_led(True, "READY")
    
    def _on_operator_response(self, response_text: str) -> None:
        """Handle operator text response"""
        self.logger.info(f"Operator: {response_text}")

    async def _on_operator_speech(self, audio_data: bytes) -> None:
        """Play operator speech audio through the output device."""
        if not self.audio_processor or not audio_data:
            return
        self.logger.info(f"Playing operator speech ({len(audio_data)} bytes)")
        self.gpio_monitor.set_status_led(True, "SPEAKING")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.audio_processor.play_wav_bytes, audio_data)
        self.gpio_monitor.set_status_led(True, "LISTENING")
    
    async def _start_listening_session(self) -> None:
        """Record a single utterance, transcribe it, and generate a response.

        Accumulates all audio chunks until the caller has been silent for
        SILENCE_DURATION seconds, or MAX_DURATION seconds have elapsed.
        Chunk-by-chunk transcription produces garbage with sub-second
        fragments so we wait for end-of-turn before passing to Whisper.
        """
        if not self.audio_processor:
            self.logger.error("Audio processor not available")
            return

        audio_status = self.audio_processor.test_audio_system()
        if audio_status["status"] != "available":
            self.logger.info("Audio not available - skipping listening session")
            return

        silence_threshold = self.config.get('audio.silence_threshold', 500)
        silence_duration = self.config.get('audio.silence_duration', 3.0)
        max_duration = self.config.get('audio.max_listen_duration', 20.0)

        self.logger.info(
            f"Listening (silence_threshold={silence_threshold}, "
            f"silence_duration={silence_duration}s, max={max_duration}s)..."
        )

        frames: list = []
        last_speech_time = [time.time()]
        rms_log_window: list = []

        def collect(chunk: np.ndarray) -> None:
            frames.append(chunk.copy())
            rms = math.sqrt(float(np.mean(chunk.astype(np.float32) ** 2)))
            rms_log_window.append(rms)
            if rms >= silence_threshold:
                last_speech_time[0] = time.time()

        if not self.audio_processor.start_recording(collect):
            self.logger.error("Failed to start audio recording")
            return

        start_time = time.time()
        last_rms_log = start_time

        while True:
            await asyncio.sleep(0.1)
            now = time.time()
            elapsed = now - start_time

            # Debug: log max RMS once per second to help tune threshold
            if now - last_rms_log >= 1.0 and rms_log_window:
                self.logger.debug(f"Audio RMS max (last 1s): {max(rms_log_window):.0f}")
                rms_log_window.clear()
                last_rms_log = now

            if elapsed >= max_duration:
                self.logger.info(f"Max listen duration ({max_duration}s) reached")
                break
            if now - last_speech_time[0] >= silence_duration and elapsed > 0.5:
                self.logger.info(f"Silence detected ({silence_duration}s) — end of turn")
                break

        self.audio_processor.stop_recording()

        if not frames:
            return

        audio_buffer = np.concatenate(frames)
        await self._process_user_audio(audio_buffer)

    async def _process_user_audio(self, audio_data: np.ndarray) -> None:
        """Transcribe a complete audio buffer and generate an operator response.

        Tracks consecutive STT failures. After 2 failures the operator asks the
        caller to speak up; after 3 it gives up and resets the session.
        """
        _MAX_STT_FAILURES = 3
        try:
            sample_rate = self.audio_processor.sample_rate if self.audio_processor else 44100
            transcription = await self.ai_operator.process_speech(audio_data, sample_rate)

            if not transcription:
                self._stt_failures += 1
                self.logger.warning(
                    f"STT returned no transcription (consecutive failures: {self._stt_failures})"
                )
                if self._stt_failures >= _MAX_STT_FAILURES:
                    self.logger.info("Too many STT failures — ending session")
                    give_up = self.ai_operator._random_phrase("stt_give_up")
                    self._on_operator_response(give_up)
                    await self._on_operator_speech(
                        await self.ai_operator.synthesize_speech(give_up) or b""
                    )
                    self.ai_operator.state = OperatorState.IDLE
                else:
                    nudge = self.ai_operator._random_phrase("stt_trouble")
                    self._on_operator_response(nudge)
                    await self._on_operator_speech(
                        await self.ai_operator.synthesize_speech(nudge) or b""
                    )
                return

            # Successful transcription — reset failure counter
            self._stt_failures = 0
            self.logger.info(f"User said: {transcription}")
            await self.ai_operator.process_user_request(
                transcription,
                on_response=self._on_operator_response,
                on_speech_audio=self._on_operator_speech,
                on_call_request=self._on_call_request
            )

        except Exception as e:
            self.logger.error(f"Error processing user audio: {e}")
    
    async def _on_call_request(self, call_request: CallRequest) -> None:
        """Place an outgoing VoIP call and start the bidirectional audio bridge.

        In simulation mode (no SIP credentials), the call is logged but not
        placed so the rest of the operator flow still works on a dev machine.
        """
        if not self.voip_client:
            self.logger.info(
                f"SIMULATION: Would connect call to {call_request.requested_number}"
            )
            return

        self.logger.info(f"Placing call to: {call_request.requested_number}")

        try:
            call_id = await self.voip_client.make_call(call_request.requested_number)
            if not call_id:
                self.logger.error("Failed to place call")
                return

            self.current_call_id = call_id
            self.in_call = True
            device_rate = self.audio_processor.sample_rate if self.audio_processor else 44100

            # Stateful resamplers for this call — soxr carries FIR filter
            # state between chunks, eliminating the crackling artifacts that
            # scipy.signal.resample_poly produces at chunk boundaries.
            downsample = soxr.ResampleStream(
                device_rate, VOIP_SAMPLE_RATE, 1, dtype='int16', quality='HQ'
            )
            upsample = soxr.ResampleStream(
                VOIP_SAMPLE_RATE, device_rate, 1, dtype='int16', quality='HQ'
            )

            def voip_in():
                pcm_8k = self.voip_client.read_call_audio(call_id)
                if pcm_8k is None:
                    return None
                resampled = upsample.resample_chunk(pcm_8k)
                return resampled if len(resampled) > 0 else None

            # Buffer for soxr's variable-length output — accumulate and
            # drain in exactly 160-sample frames so RTP timing is steady.
            out_8k_buf = np.array([], dtype=np.int16)

            def voip_out(pcm_device):
                nonlocal out_8k_buf
                pcm_8k = downsample.resample_chunk(pcm_device)
                if len(pcm_8k) > 0:
                    out_8k_buf = np.concatenate([out_8k_buf, pcm_8k])
                while len(out_8k_buf) >= 160:
                    frame = out_8k_buf[:160]
                    out_8k_buf = out_8k_buf[160:]
                    self.voip_client.write_call_audio(call_id, frame)

            self._bridge_voip_in = voip_in
            self._bridge_voip_out = voip_out

            # Monitor pyVoIP call state for ANSWERED / ENDED transitions.
            # The bridge starts in _on_call_state_change(CONNECTED).
            self.voip_client.start_call_monitor(call_id, self._on_call_state_change)
            self.logger.info(f"Call initiated: {call_id}")

        except Exception as e:
            self.logger.error(f"Error placing call: {e}")

    def _on_call_state_change(self, call_id: str, new_state: CallState) -> None:
        """Called from the VoIPClient monitor thread on pyVoIP state transitions."""
        if new_state == CallState.CONNECTED:
            self.logger.info(f"Call {call_id[:8]} answered by remote party")
            self.gpio_monitor.set_status_led(True, "IN CALL")
            # RTP is now established — safe to start the audio bridge
            if self.audio_processor and self._bridge_voip_in and self._bridge_voip_out:
                self.audio_processor.start_audio_bridge(
                    voip_audio_in=self._bridge_voip_in,
                    voip_audio_out=self._bridge_voip_out,
                )
        elif new_state == CallState.DISCONNECTED:
            self.logger.info(f"Call {call_id[:8]} ended by remote party")
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._handle_call_ended(call_id), self._loop)

    async def _handle_call_ended(self, call_id: str) -> None:
        """Clean up after the remote party disconnects."""
        self._end_call()
        self.gpio_monitor.set_status_led(True, "READY")

        if self.ai_operator:
            await self.ai_operator.announce_and_reset(
                "The other party has disconnected. Good day.",
                on_response=self._on_operator_response,
                on_speech_audio=self._on_operator_speech,
            )
    
    def _end_call(self) -> None:
        """End current call and reset state."""
        call_id = self.current_call_id
        self.in_call = False
        self.current_call_id = None
        self._bridge_voip_in = None
        self._bridge_voip_out = None

        if self.audio_processor:
            self.audio_processor.stop_audio_bridge()

        if self.voip_client and call_id:
            self.voip_client.stop_call_monitor(call_id)

        self.logger.info("Call ended")


async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Antique Telephone AI Operator")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration directory"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (simulation)"
    )
    
    args = parser.parse_args()
    
    # Override log level if debug mode
    if args.debug:
        import os
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Create and start the system
    system = AntiquePhoneSystem(args.config)
    
    if args.test:
        # Test mode - run brief test and exit
        print("Running in test mode...")
        if await system.initialize():
            print("✓ System initialization successful")
            
            # Test basic functionality
            if system.gpio_monitor:
                gpio_status = system.gpio_monitor.test_inputs()
                print(f"✓ GPIO status: {gpio_status['status']}")
            
            if system.audio_processor:
                audio_status = system.audio_processor.test_audio_system()
                print(f"✓ Audio status: {audio_status['status']}")
            
            if system.ai_operator:
                ai_status = system.ai_operator.get_status()
                print(f"✓ AI providers: {ai_status['providers']}")
            
            await system.shutdown()
            print("✓ Test completed successfully")
        else:
            print("✗ System initialization failed")
            sys.exit(1)
    else:
        # Normal operation
        await system.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)