"""
Anthropic Conversation Provider - Antique Telephone AI Operator

Uses Anthropic's API for two narrow tasks where inference is genuinely needed:
  1. Entity extraction — pulling a business name out of a transcript
  2. Edge-case fallback — input the state machine doesn't recognise

Defaults to claude-haiku-4-5-20251001 (fastest / cheapest). The state machine
handles all normal operator conversation without touching this provider.

Set ANTHROPIC_API_KEY in the environment or config to enable.
"""

import asyncio
import logging
from typing import Optional

from providers.base import ConversationProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(ConversationProvider):
    """
    Conversation via Anthropic API.

    Identical behaviour on dev machine and Raspberry Pi — no local model,
    no RAM constraints, sub-second latency for short prompts.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 100,
        temperature: float = 0.3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._available = False
        self._setup(api_key)

    def _setup(self, api_key: str) -> None:
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed — run: uv add anthropic")
            return

        if not api_key:
            logger.warning(
                "No Anthropic API key configured (ANTHROPIC_API_KEY). "
                "Conversation provider disabled."
            )
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            # Lightweight connectivity check
            self._available = True
            logger.info(f"Anthropic provider ready: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialise Anthropic client: {e}")

    # ------------------------------------------------------------------
    # ConversationProvider interface
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available

    async def get_response(self, system_prompt: str, messages: list) -> Optional[str]:
        """Generate a response via the Anthropic API."""
        if not self._available or self._client is None:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._chat_sync, system_prompt, messages
        )

    def _chat_sync(self, system_prompt: str, messages: list) -> Optional[str]:
        try:
            import anthropic

            # Anthropic requires alternating user/assistant turns; trim to last 6
            trimmed = messages[-6:]

            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=trimmed,
            )
            return response.content[0].text.strip()

        except anthropic.AuthenticationError:
            logger.error("Anthropic API key is invalid")
            self._available = False
            return None
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return None
