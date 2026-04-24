"""
Whisper STT Provider - Antique Telephone AI Operator

Local speech-to-text using OpenAI Whisper. Runs entirely on-device —
no API key required. Resamples input audio to 16 kHz before transcription.
"""

import asyncio
import logging
from typing import Optional

import numpy as np

from providers.base import STTProvider

logger = logging.getLogger(__name__)

_WHISPER_SR = 16_000  # Whisper always expects 16 kHz float32


class WhisperProvider(STTProvider):
    """
    STT via a local Whisper ONNX model.

    The model is loaded once on init. Inference runs in an executor so
    it doesn't block the asyncio event loop during transcription.
    """

    def __init__(self, model_name: str = "base", language: str = "en"):
        self.language = language
        self._model = None
        self._load(model_name)

    def _load(self, model_name: str) -> None:
        try:
            import whisper
            self._model = whisper.load_model(model_name)
            logger.info(f"Whisper model '{model_name}' loaded")
        except ImportError:
            logger.warning("openai-whisper not installed — STT unavailable")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_name}': {e}")

    @property
    def is_available(self) -> bool:
        return self._model is not None

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 44100) -> Optional[str]:
        """Transcribe audio, resampling to 16 kHz as required by Whisper."""
        if not self._model:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe_sync, audio, sample_rate
        )

    def _transcribe_sync(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        try:
            audio_float = audio.astype(np.float32) / 32768.0

            if sample_rate != _WHISPER_SR and len(audio_float) > 0:
                target_len = int(len(audio_float) * _WHISPER_SR / sample_rate)
                audio_float = np.interp(
                    np.linspace(0, len(audio_float) - 1, target_len),
                    np.arange(len(audio_float)),
                    audio_float,
                ).astype(np.float32)

            result = self._model.transcribe(
                audio_float,
                language=self.language,
                fp16=False,
            )
            text = result["text"].strip()
            logger.info(f"Transcribed: '{text}'")
            return text or None

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return None
