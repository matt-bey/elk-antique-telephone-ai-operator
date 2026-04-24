"""
Piper TTS Provider - Antique Telephone AI Operator

Local text-to-speech using Piper (piper-tts). Designed for Raspberry Pi —
real-time synthesis on CPU with small ONNX models.

Models are auto-downloaded from HuggingFace on first use and cached in
~/.local/share/piper-voices/.
"""

import asyncio
import io
import logging
import urllib.request
import wave
from pathlib import Path
from typing import Optional

from providers.base import TTSProvider

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".local" / "share" / "piper-voices"
_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# name → relative path prefix within the HuggingFace repo
KNOWN_VOICES: dict[str, str] = {
    "en_US-lessac-high":          "en/en_US/lessac/high/en_US-lessac-high",
    "en_US-amy-medium":           "en/en_US/amy/medium/en_US-amy-medium",
    "en_GB-jenny_dioco-medium":   "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium",
    "en_GB-alan-medium":          "en/en_GB/alan/medium/en_GB-alan-medium",
}


class PiperProvider(TTSProvider):
    """
    TTS via local Piper ONNX voice models.

    On first use, downloads the requested voice model (~65 MB) from HuggingFace
    to ~/.local/share/piper-voices/ and caches it for subsequent runs.
    """

    def __init__(
        self,
        voice_name: str = "en_US-lessac-high",
        models_dir: Optional[Path] = None,
    ):
        self.voice_name = voice_name
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self._voice = None
        self._available = False
        self._load()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from piper.voice import PiperVoice  # noqa: F401 — confirm installed
        except ImportError:
            logger.warning("piper-tts not installed — run: uv add piper-tts")
            return

        model_path = self._ensure_model()
        if model_path is None:
            return

        try:
            from piper.voice import PiperVoice
            self._voice = PiperVoice.load(model_path, download_dir=self.models_dir)
            self._available = True
            logger.info(f"Piper TTS ready: {self.voice_name}")
        except Exception as e:
            logger.error(f"Failed to load Piper model '{self.voice_name}': {e}")

    def _ensure_model(self) -> Optional[Path]:
        if self.voice_name not in KNOWN_VOICES:
            logger.error(
                f"Unknown Piper voice '{self.voice_name}'. "
                f"Available: {list(KNOWN_VOICES)}"
            )
            return None

        rel = KNOWN_VOICES[self.voice_name]
        model_path = self.models_dir / f"{self.voice_name}.onnx"
        config_path = self.models_dir / f"{self.voice_name}.onnx.json"

        self.models_dir.mkdir(parents=True, exist_ok=True)

        if not model_path.exists() or not config_path.exists():
            logger.info(f"Downloading Piper voice '{self.voice_name}' to {self.models_dir} …")
            try:
                self._download(f"{_HF_BASE}/{rel}.onnx", model_path)
                self._download(f"{_HF_BASE}/{rel}.onnx.json", config_path)
                logger.info("Download complete.")
            except Exception as e:
                logger.error(f"Model download failed: {e}")
                logger.error(f"Download manually: {_HF_BASE}/{rel}.onnx")
                return None

        return model_path

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        logger.info(f"  {url}")
        urllib.request.urlretrieve(url, dest)

    # ------------------------------------------------------------------
    # TTSProvider interface
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text and return WAV bytes, or None on failure."""
        if not self._available or self._voice is None:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> Optional[bytes]:
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self._voice.synthesize_wav(text, wf)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            return None
