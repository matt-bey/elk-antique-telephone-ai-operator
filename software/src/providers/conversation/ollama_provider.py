"""
Ollama Conversation Provider - Antique Telephone AI Operator

Local LLM inference via Ollama. Connects to a running Ollama server
(default: http://localhost:11434) and uses a small model for generating
1920s telephone operator responses.

Install Ollama: https://ollama.com/download
Pull a model:   ollama pull llama3.2:1b
"""

import asyncio
import logging
from typing import Optional

from providers.base import ConversationProvider

logger = logging.getLogger(__name__)


class OllamaProvider(ConversationProvider):
    """
    Conversation via a local Ollama LLM.

    Checks server availability on init. If the server isn't running,
    is_available returns False and get_response returns None so the
    caller can fall back to pattern matching.
    """

    def __init__(
        self,
        model: str = "llama3.2:1b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host
        self._available = False
        self._check()

    def _check(self) -> None:
        try:
            import ollama
            client = ollama.Client(host=self.host)
            models = client.list()
            available_names = [m.model for m in models.models]
            if self.model not in available_names:
                logger.warning(
                    f"Ollama model '{self.model}' not found on server. "
                    f"Run: ollama pull {self.model}"
                )
                logger.warning(f"Available models: {available_names or '(none)'}")
                return
            self._available = True
            logger.info(f"Ollama ready: {self.model} at {self.host}")
        except ImportError:
            logger.warning("ollama package not installed — run: uv add ollama")
        except Exception as e:
            logger.warning(f"Ollama server not reachable at {self.host}: {e}")
            logger.warning("Start Ollama with: ollama serve")

    # ------------------------------------------------------------------
    # ConversationProvider interface
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available

    async def get_response(self, system_prompt: str, messages: list) -> Optional[str]:
        """Generate a response using the Ollama model."""
        if not self._available:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._chat_sync, system_prompt, messages)

    def _chat_sync(self, system_prompt: str, messages: list) -> Optional[str]:
        try:
            import ollama
            client = ollama.Client(host=self.host)

            full_messages = [
                {"role": "system", "content": system_prompt},
                *messages[-6:],  # last 3 turns (user + assistant each)
            ]

            response = client.chat(
                model=self.model,
                messages=full_messages,
                options={"temperature": 0.3, "num_predict": 150},
            )
            return response.message.content.strip()

        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            self._available = False
            return None
