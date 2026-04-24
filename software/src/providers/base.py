"""
Provider Base Classes - Antique Telephone AI Operator

Abstract interfaces for swappable AI service providers.
Each layer (TTS, conversation) has one interface; implementations live in subpackages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


class STTProvider(ABC):
    """Speech-to-text provider interface."""

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int = 44100) -> Optional[str]:
        """Transcribe audio to text. Returns None if unavailable or silent."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is ready to use."""


class TTSProvider(ABC):
    """Text-to-speech provider interface."""

    @abstractmethod
    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to WAV bytes. Returns None if unavailable."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is ready to use."""


class ConversationProvider(ABC):
    """Conversation / response-generation provider interface."""

    @abstractmethod
    async def get_response(self, system_prompt: str, messages: list) -> Optional[str]:
        """Generate a response given a system prompt and message history.

        messages: list of {"role": "user"|"assistant", "content": str}
        Returns None if unavailable.
        """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is ready to use."""


@dataclass
class LookupResult:
    """A single business/directory listing returned by a LookupProvider."""

    name: str
    address: str
    phone_number: str
    confidence: float
    source: str = "business"


class LookupProvider(ABC):
    """Business / directory lookup provider interface."""

    @abstractmethod
    async def search(
        self,
        query: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> List[LookupResult]:
        """Search for a business by name or description.

        Returns a ranked list of matches (empty on failure).
        lat/lon bias the search toward the caller's location when provided.
        """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is ready to use."""
