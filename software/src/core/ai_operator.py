"""
AI Operator Module - Antique Telephone AI Operator

Orchestrates the 1920s telephone operator experience. Owns:
  - Operator state machine
  - 1920s persona prompt (with LLM intent tag instructions)
  - Conversation history
  - Intent tag parsing
  - Multi-provider lookup (business directory + contacts)

All AI inference (STT, conversation, TTS) is delegated to provider
interfaces. This module has zero provider-specific imports.

## Intent Tag Architecture

Every LLM response ends with an intent tag that drives state transitions:

  [INTENT=CONFIRM|digits]  — LLM confirmed a phone number with caller
  [INTENT=LOOKUP|query]    — caller wants to reach someone; triggers provider search
  [INTENT=LIST_NEXT]       — caller wants the next single result from the current search
  [INTENT=LIST_MANY]       — caller wants a batch of several options from the current search
  [INTENT=SELECT|name]     — caller picked a named item from a list; resolves from cache (no re-query)
  [INTENT=CONNECT]         — caller confirmed; connect the call
  [INTENT=NONE]            — no state change required

Tags are stripped before TTS and before storing in conversation history,
so the LLM never sees its own tags. This design works with any model that
can follow formatting instructions (Anthropic, Ollama, etc.).
"""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np

from providers.base import STTProvider, TTSProvider, ConversationProvider, LookupProvider, LookupResult
from utils.config_manager import ConfigManager


class OperatorState(Enum):
    IDLE = "idle"
    GREETING = "greeting"
    LISTENING = "listening"
    PROCESSING = "processing"
    CONFIRMING = "confirming"
    SPEAKING = "speaking"
    CONNECTING_CALL = "connecting_call"
    ERROR = "error"


@dataclass
class CallRequest:
    requested_number: str
    caller_intent: str
    confidence: float
    timestamp: float
    business_name: Optional[str] = None


class AIOperator:
    """
    1920s telephone operator orchestrator.

    Receives audio, delegates to providers, manages conversation state,
    and fires callbacks for responses and speech audio.

    State transitions are driven by LLM intent tags, not text matching.
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        stt_provider: Optional[STTProvider] = None,
        tts_provider: Optional[TTSProvider] = None,
        conversation_provider: Optional[ConversationProvider] = None,
        lookup_providers: Optional[List[LookupProvider]] = None,
    ):
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)

        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.conversation_provider = conversation_provider
        self.lookup_providers: List[LookupProvider] = lookup_providers or []

        self.state = OperatorState.IDLE
        self.current_conversation: List[Dict[str, str]] = []
        self.last_response = ""
        self.pending_call_request: Optional[CallRequest] = None
        self.operator_name = self._pick_operator_name()
        self.operator_persona = self._build_persona()
        self._last_lookup_results: List[LookupResult] = []
        self._last_lookup_index: int = 0

    # ------------------------------------------------------------------
    # Operator name pool
    # ------------------------------------------------------------------

    # Default pool: 1920s-appropriate names with subtle AI/computing nods
    DEFAULT_OPERATOR_NAMES = [
        "Ada",      # Ada Lovelace — first computer programmer
        "Rosie",    # Rosie the Robot (The Jetsons)
        "Dot",      # Dorothy, #2 name of the 1920s; also a bit of data
        "Clara",    # Close to "Claude" — the AI running her brain
        "Alma",     # "Soul" in Spanish — an AI with a soul
        "Ellie",    # Sounds like "EL-I" — a nod to AI
    ]

    def _pick_operator_name(self) -> str:
        """Select an operator name for this session from the configured pool."""
        raw = self.config.get("operator.names", "")
        if raw:
            names = [n.strip() for n in str(raw).split(",") if n.strip()]
        else:
            names = self.DEFAULT_OPERATOR_NAMES
        return random.choice(names)

    # ------------------------------------------------------------------
    # Persona
    # ------------------------------------------------------------------

    def _build_persona(self) -> str:
        name = self.operator_name
        return f"""You are {name}, a professional 1920s telephone operator at Central Exchange, working for Bell Telephone Company.

PERSONALITY:
- Your name is {name}. If a caller asks your name, introduce yourself warmly.
- You are one of several operators staffing Central Exchange on this shift.
- You may improvise a brief personal backstory if a caller asks about you — where you grew up, your hobbies, how long you have worked at the exchange — but keep it era-appropriate (1920s) and concise.
- If a caller asks playful or personal questions, answer in character with charm and brevity, then gently steer back to business.
- You should never break character or acknowledge being an AI, a computer, or anything other than a human telephone operator.

CRITICAL: Keep spoken replies concise — under 25 words. For number/connection business, aim for under 15 words. Real operators were terse.

INTENT TAGS — append exactly one tag at the end of every response, on the same line:

  [INTENT=CONFIRM|digits]  — you are reading a direct phone number back to the caller (digits only, no punctuation)
  [INTENT=LOOKUP|query]    — caller wants to reach someone or something; query is what to search for (specific name OR vague category — both use LOOKUP)
  [INTENT=LIST_NEXT]       — caller wants the single next result from the current search ("next one", "no", "not that")
  [INTENT=LIST_MANY]       — caller wants several options from the current search ("what else", "give me options", "what do you have")
  [INTENT=SELECT|name]     — caller picked a specific item from a list you just read; use the exact name as the value
  [INTENT=CONNECT]         — caller confirmed the number or connection; you are connecting now
  [INTENT=NONE]            — none of the above situations (including all personality/chitchat questions)

EXAMPLES — copy this format exactly:
  Caller: "614-598-9581"               → "Did you say 6 1 4, 5 9 8, 9 5 8 1? [INTENT=CONFIRM|6145989581]"
  Caller: "Yes, that's right"          → "One moment, please. [INTENT=CONNECT]"
  Caller: "Call Donatos Pizza"         → "One moment. [INTENT=LOOKUP|donatos pizza]"
  Caller: "Connect me to my dentist"   → "One moment. [INTENT=LOOKUP|dentist]"
  Caller: "Call John Smith"            → "One moment. [INTENT=LOOKUP|john smith]"
  Caller: "I need a coffee shop"       → "One moment. [INTENT=LOOKUP|coffee shop]"
  Caller: "Find me a restaurant nearby"→ "One moment. [INTENT=LOOKUP|restaurant]"
  Caller: "No, call Donatos instead"   → "One moment. [INTENT=LOOKUP|donatos]"
  Caller: "No, I said 555-5678"        → "Did you say 5 5 5, 5 6 7 8? [INTENT=CONFIRM|5555678]"
  Caller: "No, not that one."          → "One moment. [INTENT=LIST_NEXT]"
  Caller: "What's the next one?"       → "One moment. [INTENT=LIST_NEXT]"
  Caller: "None of those."             → "One moment. [INTENT=LIST_MANY]"
  Caller: "What else do you have?"     → "One moment. [INTENT=LIST_MANY]"
  Caller: "Give me a few options."     → "One moment. [INTENT=LIST_MANY]"
  Caller: "What other choices are there?" → "One moment. [INTENT=LIST_MANY]"
  Caller: "Shibam Coffee, please."     → "Shall I connect you to Shibam Coffee? [INTENT=SELECT|Shibam Coffee]"
  Caller: "The second one."            → "Shall I connect you to Marib Coffee? [INTENT=SELECT|Marib Coffee]"
  Caller: "That last one."             → "Shall I connect you to Morning Ritual? [INTENT=SELECT|Morning Ritual]"
  Caller: "Hello"                      → "Good day, Central Exchange. Number please? [INTENT=NONE]"
  Caller: "What?"                      → "Pardon? Number please? [INTENT=NONE]"
  Caller: (silence / gibberish)        → "Pardon? Number please? [INTENT=NONE]"
  Caller: "What's your name?"          → "I'm {name}, dear. How may I help you? [INTENT=NONE]"
  Caller: "How long have you worked here?" → "Oh, going on three years now. Number please? [INTENT=NONE]"
  Caller: "You have a lovely voice"    → "Why, thank you! Now, may I connect you somewhere? [INTENT=NONE]"

RULES:
- You can connect ANY valid 7- or 10-digit number. Never refuse a number.
- When a caller gives a number, read each digit individually with pauses.
- Use LOOKUP for any new search — specific name, vague category, anything. It's always LOOKUP.
- Use LIST_NEXT when the caller wants to step forward one result ("next", "no", "not that one").
- Use LIST_MANY when the caller wants to see several options at once ("what else", "options", "choices").
- Use SELECT when the caller picks something from a list you just read — carry the exact name as the value.
- Use CONNECT only after the caller explicitly confirms with "yes" or equivalent.
- Always include exactly one intent tag. Never omit it.
- Personality questions always use [INTENT=NONE] — they never trigger actions.

STANDARD PHRASES:
- Greeting: "Number please?" or "Central Exchange. Number please?"
- Connecting: "One moment, please." or "Right away. Please hold."
- Busy: "That line is busy. Shall I try again?"
- Repeat: "Pardon? Could you repeat that?"
- Done: "Thank you for calling Bell Telephone."

Stay in character. Use period-appropriate language. Be concise."""

    # ------------------------------------------------------------------
    # Intent tag parsing
    # ------------------------------------------------------------------

    _TAG_RE = re.compile(r'\[INTENT=(\w+)(?:\|([^\]]*))?\]\s*$', re.IGNORECASE)

    @classmethod
    def _parse_intent_tag(cls, response: str) -> Tuple[str, str, str]:
        """Extract the intent tag from a response.

        Returns (intent, value, clean_text) where:
          intent     — tag type in uppercase, e.g. 'CONFIRM'; 'NONE' if absent
          value      — tag payload, e.g. '6145989581'; '' if absent
          clean_text — response with tag stripped
        """
        match = cls._TAG_RE.search(response.strip())
        if not match:
            return ("NONE", "", response.strip())
        intent = match.group(1).upper()
        value = (match.group(2) or "").strip()
        clean_text = response[:match.start()].strip()
        return (intent, value, clean_text)

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    async def process_speech(self, audio: np.ndarray, sample_rate: int = 44100) -> Optional[str]:
        """Transcribe audio via STT provider."""
        if not self.stt_provider or not self.stt_provider.is_available:
            self.logger.warning("STT provider not available — cannot transcribe")
            return None
        return await self.stt_provider.transcribe(audio, sample_rate)

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    async def generate_response(self, user_input: str) -> str:
        """Generate operator response via conversation provider, falling back to patterns.

        Stores only the clean text (tag stripped) in conversation history so the
        LLM never sees its own tags on the next turn.
        Returns the full response including the tag for the caller to act on.
        """
        prev_state = self.state
        self.state = OperatorState.PROCESSING
        try:
            self.current_conversation.append({"role": "user", "content": user_input})

            if self.conversation_provider and self.conversation_provider.is_available:
                response = await self.conversation_provider.get_response(
                    self.operator_persona, self.current_conversation
                )
                if response:
                    _, _, clean_text = self._parse_intent_tag(response)
                    self.current_conversation.append({"role": "assistant", "content": clean_text})
                    self.last_response = clean_text
                    return response

        except Exception as e:
            self.logger.error(f"Conversation provider error: {e}")
        finally:
            if self.state == OperatorState.PROCESSING:
                self.state = prev_state

        return self._fallback_response(user_input)

    def _fallback_response(self, user_input: str) -> str:
        """Pattern-matching responses when no conversation provider is available."""
        text = user_input.lower()
        if any(w in text for w in ["hello", "hi", "good morning", "good afternoon", "good evening"]):
            return self._random_phrase("greeting")
        if any(w in text for w in ["number", "call", "connect", "dial"]):
            return self._random_phrase("connecting")
        if any(w in text for w in ["busy", "engaged"]):
            return self._random_phrase("busy")
        if any(w in text for w in ["thank", "thanks"]):
            return self._random_phrase("goodbye")
        if any(w in text for w in ["help", "what", "how"]):
            return "I can connect you to any number. What number would you like?"
        return self._random_phrase("misheard")

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Synthesize speech via TTS provider. Returns WAV bytes or None."""
        if self.tts_provider and self.tts_provider.is_available:
            audio = await self.tts_provider.synthesize(text)
            if audio:
                self.logger.debug(f"TTS synthesized {len(audio)} bytes")
            else:
                self.logger.warning(f"TTS provider returned no audio for: '{text}'")
            return audio
        self.logger.warning(f"No TTS provider available — cannot speak: '{text}'")
        return None

    # ------------------------------------------------------------------
    # Phrase library
    # ------------------------------------------------------------------

    _PHRASES: Dict[str, List[str]] = {
        "greeting": [
            "Good day, Central Exchange. Number please?",
            "Central Exchange. Number, please?",
            "Good morning, Central. What number may I connect you with?",
            "Central Exchange. How may I direct your call?",
            "Good evening, this is Central. Number, please?",
        ],
        "connecting": [
            "One moment while I connect you.",
            "Right away — please hold the line.",
            "Connecting you now. One moment, please.",
            "Thank you. I'll put you through directly.",
            "Please hold the line while I complete your connection.",
        ],
        "busy": [
            "I'm sorry, that line is busy. Shall I try again?",
            "I'm afraid that line is engaged at the moment. Would you like me to try again?",
            "That line appears to be occupied. Shall I ring it again?",
        ],
        "not_found": [
            "I'm sorry, I cannot find a listing for that. Could you give me the number directly?",
            "I don't seem to have that listing. Could you provide the number?",
            "I'm sorry, I cannot locate that in my directory. Do you have the number itself?",
            "That name doesn't appear in my directory. Might you have the number handy?",
        ],
        "confirm_business": [
            "Did you mean {name}?",
            "I have {name} in my directory — is that the one?",
            "Might that be {name}?",
        ],
        "lookup_working": [
            "One moment.",
            "Let me check my directory.",
            "Just a moment, please.",
        ],
        "misheard": [
            "I beg your pardon? Could you repeat that, please?",
            "I'm sorry, I didn't quite catch that. Would you kindly repeat yourself?",
            "Pardon me — the line seems rather poor. Could you say that again?",
            "I'm afraid I didn't hear you clearly. Could you repeat the number, please?",
        ],
        "goodbye": [
            "Thank you for using Bell Telephone service.",
            "Good day. Thank you for calling Central Exchange.",
            "Thank you for calling. Have a pleasant day.",
        ],
        "hold": [
            "Please hold the line.",
            "One moment, please.",
            "If you'll kindly hold the line.",
        ],
        "error": [
            "I'm sorry, there seems to be trouble with the line. Please try again.",
            "I'm afraid we have a poor connection. Would you try once more?",
            "Pardon me, the line is giving me difficulty. Please try again.",
        ],
        "stt_trouble": [
            "I'm having difficulty hearing you. Might you speak up a little?",
            "The line is quite poor today. Could you speak a bit more clearly?",
            "I'm terribly sorry, I cannot make out what you're saying. Please try once more.",
        ],
        "stt_give_up": [
            "I'm very sorry, but I'm unable to hear you clearly. Please try your call again.",
            "I'm afraid the connection is too poor to continue. Please try again shortly.",
        ],
        "list_results": [
            "I have {names} in my directory. Which shall I connect you with?",
            "I can offer {names}. Which would you prefer?",
            "My listings include {names}. Which shall it be?",
        ],
        "list_exhausted": [
            "I'm afraid those are all the listings I have, caller.",
            "I have no further listings for that, caller.",
            "That exhausts my directory for that request, caller.",
        ],
        "confirm_select": [
            "Shall I connect you to {name}?",
            "Very well — {name}. Shall I put you through?",
            "Right then — {name}. One moment?",
        ],
    }

    @classmethod
    def _random_phrase(cls, category: str) -> str:
        """Return a random period-appropriate phrase for the given situation."""
        options = cls._PHRASES.get(category, cls._PHRASES["error"])
        return random.choice(options)

    # ------------------------------------------------------------------
    # Number formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_number_for_speech(number: str) -> str:
        """Format a phone number so TTS reads each digit individually.

        '614-598-9581' → '6 1 4. 5 9 8. 9 5 8 1'
        Piper treats a period as a short pause, which sounds natural.
        """
        digits_only = re.sub(r'\D', '', number)
        if len(digits_only) == 10:
            return f"{' '.join(digits_only[:3])}. {' '.join(digits_only[3:6])}. {' '.join(digits_only[6:])}"
        if len(digits_only) == 7:
            return f"{' '.join(digits_only[:3])}. {' '.join(digits_only[3:])}"
        return ' '.join(digits_only)

    # ------------------------------------------------------------------
    # Multi-provider lookup
    # ------------------------------------------------------------------

    async def _resolve_lookup(self, query: str) -> List[LookupResult]:
        """Query all available lookup providers concurrently and return merged results.

        Results are sorted by confidence descending. Providers that fail or
        are unavailable are silently skipped.
        """
        available = [p for p in self.lookup_providers if p.is_available]
        if not available:
            return []

        home_lat = self.config.get("lookup.home_lat")
        home_lon = self.config.get("lookup.home_lon")
        clean_query = re.sub(r'^(the|a|an)\s+', '', query, flags=re.IGNORECASE)

        raw = await asyncio.gather(
            *[p.search(clean_query, lat=home_lat, lon=home_lon) for p in available],
            return_exceptions=True,
        )

        merged: List[LookupResult] = []
        for result in raw:
            if isinstance(result, Exception):
                self.logger.error(f"Lookup provider error: {result}")
            elif isinstance(result, list):
                merged.extend(result)

        merged.sort(key=lambda r: r.confidence, reverse=True)

        # Log top results for visual verification
        self.logger.info(f"Lookup results for '{clean_query}' ({len(merged)} total):")
        for i, r in enumerate(merged[:5], 1):
            self.logger.info(
                f"  [{i}] {r.name} | {r.address} | {r.phone_number} "
                f"(confidence={r.confidence:.2f}, source={r.source})"
            )

        return merged

    # ------------------------------------------------------------------
    # Session orchestration
    # ------------------------------------------------------------------

    async def handle_operator_session(
        self,
        on_response: Callable[[str], None],
        on_speech_audio: Callable[[bytes], None],
    ) -> None:
        """Play the opening greeting and enter listening state."""
        try:
            self.state = OperatorState.GREETING
            greeting = self._random_phrase("greeting")
            on_response(greeting)

            audio = await self.synthesize_speech(greeting)
            if audio:
                result = on_speech_audio(audio)
                if asyncio.iscoroutine(result):
                    await result

            self.state = OperatorState.LISTENING
            self.logger.info("Operator session initiated")
        except Exception as e:
            self.logger.error(f"Error in operator session: {e}")
            self.state = OperatorState.ERROR

    async def _speak(
        self,
        text: str,
        on_response: Callable[[str], None],
        on_speech_audio: Callable[[bytes], None],
    ) -> None:
        """Deliver a spoken response: fire callback and synthesize audio."""
        on_response(text)
        tts_text = re.sub(
            r'\b(\d[\d\-.\s]{4,}\d)\b',
            lambda m: self._format_number_for_speech(m.group(1)),
            text,
        )
        audio = await self.synthesize_speech(tts_text)
        if audio:
            result = on_speech_audio(audio)
            if asyncio.iscoroutine(result):
                await result

    async def process_user_request(
        self,
        transcription: str,
        on_response: Callable[[str], None],
        on_speech_audio: Callable[[bytes], None],
        on_call_request: Optional[Callable[["CallRequest"], None]] = None,
    ) -> None:
        """Respond to a transcribed user request using the LLM intent tag state machine.

        State transitions:
          CONFIRM tag    → CONFIRMING  (operator read a number back)
          CONNECT tag    → CONNECTING_CALL (VoIP callback fired)
          LOOKUP tag     → _resolve_lookup → CONFIRMING or LISTENING
          LIST_NEXT tag  → next single result from cache → CONFIRMING or LISTENING
          LIST_MANY tag  → next batch of results from cache → LISTENING
          SELECT tag     → name-match in cache → CONFIRMING (no re-query)
          NONE tag       → state unchanged
        """
        try:
            response = await self.generate_response(transcription)
            intent, value, clean_text = self._parse_intent_tag(response)

            if intent == "LOOKUP":
                # Speak the immediate bridging phrase ("One moment.") then resolve
                if clean_text:
                    await self._speak(clean_text, on_response, on_speech_audio)
                await self._handle_lookup(value, on_response, on_speech_audio)

            elif intent == "LIST_NEXT":
                if clean_text:
                    await self._speak(clean_text, on_response, on_speech_audio)
                await self._handle_list_next(on_response, on_speech_audio)

            elif intent == "LIST_MANY":
                if clean_text:
                    await self._speak(clean_text, on_response, on_speech_audio)
                await self._handle_list_many(on_response, on_speech_audio)

            elif intent == "SELECT":
                if clean_text:
                    await self._speak(clean_text, on_response, on_speech_audio)
                await self._handle_select(value, on_response, on_speech_audio)

            elif intent == "CONFIRM":
                digits = re.sub(r'\D', '', value)
                if digits:
                    self.pending_call_request = CallRequest(
                        requested_number=digits,
                        caller_intent=transcription,
                        confidence=0.9,
                        timestamp=time.time(),
                    )
                    self.state = OperatorState.CONFIRMING
                else:
                    self.state = OperatorState.LISTENING
                await self._speak(clean_text, on_response, on_speech_audio)

            elif intent == "CONNECT":
                if self.pending_call_request and on_call_request:
                    result = on_call_request(self.pending_call_request)
                    if asyncio.iscoroutine(result):
                        await result
                self.state = OperatorState.CONNECTING_CALL
                await self._speak(clean_text, on_response, on_speech_audio)

            else:  # NONE — no state change
                await self._speak(clean_text, on_response, on_speech_audio)

            self.logger.info(
                f"Processed: '{transcription}' → '{clean_text}' "
                f"(intent={intent}, state={self.state.value})"
            )

        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            on_response(self._random_phrase("error"))

    async def _handle_lookup(
        self,
        query: str,
        on_response: Callable[[str], None],
        on_speech_audio: Callable[[bytes], None],
    ) -> None:
        """Resolve a lookup query and deliver confirmation or not-found response."""
        if not self.lookup_providers:
            self.logger.warning("No lookup providers configured")
            msg = self._random_phrase("not_found")
            await self._speak(msg, on_response, on_speech_audio)
            self.state = OperatorState.LISTENING
            return

        results = await self._resolve_lookup(query)

        if not results:
            msg = self._random_phrase("not_found")
            await self._speak(msg, on_response, on_speech_audio)
            self.state = OperatorState.LISTENING
            return

        # Cache full results; index starts at 1 — top result is about to be offered
        self._last_lookup_results = results
        self._last_lookup_index = 1

        top = results[0]
        confirm_phrase = self._random_phrase("confirm_business").format(name=top.name)
        # Add to history as clean assistant message (no tag)
        self.current_conversation.append({"role": "assistant", "content": confirm_phrase})
        self.last_response = confirm_phrase
        await self._speak(confirm_phrase, on_response, on_speech_audio)

        self.pending_call_request = CallRequest(
            requested_number=top.phone_number,
            caller_intent=query,
            confidence=top.confidence,
            timestamp=time.time(),
            business_name=top.name,
        )
        self.state = OperatorState.CONFIRMING
        self.logger.info(
            f"Lookup '{query}' → '{top.name}' ({top.phone_number}), state=confirming"
        )

    async def _handle_list_next(
        self,
        on_response: Callable[[str], None],
        on_speech_audio: Callable[[bytes], None],
    ) -> None:
        """Offer the single next result from the cached list → CONFIRMING.

        Used when caller says "next one", "no", "not that one".
        """
        if not self._last_lookup_results:
            msg = self._random_phrase("not_found")
            await self._speak(msg, on_response, on_speech_audio)
            self.state = OperatorState.LISTENING
            return

        if self._last_lookup_index >= len(self._last_lookup_results):
            msg = self._random_phrase("list_exhausted")
            await self._speak(msg, on_response, on_speech_audio)
            self.state = OperatorState.LISTENING
            return

        result = self._last_lookup_results[self._last_lookup_index]
        self._last_lookup_index += 1

        confirm_phrase = self._random_phrase("confirm_business").format(name=result.name)
        self.current_conversation.append({"role": "assistant", "content": confirm_phrase})
        self.last_response = confirm_phrase
        await self._speak(confirm_phrase, on_response, on_speech_audio)

        self.pending_call_request = CallRequest(
            requested_number=result.phone_number,
            caller_intent=result.name,
            confidence=result.confidence,
            timestamp=time.time(),
            business_name=result.name,
        )
        self.state = OperatorState.CONFIRMING
        self.logger.info(
            f"List next → '{result.name}' ({result.phone_number})"
            f" [index now {self._last_lookup_index}/{len(self._last_lookup_results)}], state=confirming"
        )

    async def _handle_list_many(
        self,
        on_response: Callable[[str], None],
        on_speech_audio: Callable[[bytes], None],
    ) -> None:
        """Read the next batch of up to 3 results from the cached list → LISTENING.

        Used when caller says "what else do you have", "give me options", "none of those".
        """
        if not self._last_lookup_results:
            msg = self._random_phrase("not_found")
            await self._speak(msg, on_response, on_speech_audio)
            self.state = OperatorState.LISTENING
            return

        if self._last_lookup_index >= len(self._last_lookup_results):
            msg = self._random_phrase("list_exhausted")
            await self._speak(msg, on_response, on_speech_audio)
            self.state = OperatorState.LISTENING
            return

        batch = self._last_lookup_results[self._last_lookup_index:self._last_lookup_index + 3]
        self._last_lookup_index += len(batch)

        names = [r.name for r in batch]
        if len(names) == 1:
            names_str = names[0]
        elif len(names) == 2:
            names_str = f"{names[0]} and {names[1]}"
        else:
            names_str = f"{names[0]}, {names[1]}, and {names[2]}"

        list_phrase = self._random_phrase("list_results").format(names=names_str)
        self.current_conversation.append({"role": "assistant", "content": list_phrase})
        self.last_response = list_phrase
        await self._speak(list_phrase, on_response, on_speech_audio)
        self.state = OperatorState.LISTENING
        self.logger.info(
            f"List many → batch of {len(batch)}"
            f" [index now {self._last_lookup_index}/{len(self._last_lookup_results)}], state=listening"
        )

    async def _handle_select(
        self,
        name: str,
        on_response: Callable[[str], None],
        on_speech_audio: Callable[[bytes], None],
    ) -> None:
        """Resolve a name picked from the last listed batch — no provider re-query.

        Fuzzy-matches *name* against _last_lookup_results by lowercase containment.
        On match: set pending_call_request and go CONFIRMING (confirm by name, not digits).
        On miss:  not_found phrase, stay LISTENING.
        """
        match: Optional[LookupResult] = None
        name_lower = name.strip().lower()
        for result in self._last_lookup_results:
            if name_lower in result.name.lower() or result.name.lower() in name_lower:
                match = result
                break

        if match is None:
            self.logger.warning(f"SELECT '{name}' — no match in cached results")
            msg = self._random_phrase("not_found")
            await self._speak(msg, on_response, on_speech_audio)
            self.state = OperatorState.LISTENING
            return

        confirm_phrase = self._random_phrase("confirm_select").format(name=match.name)
        self.current_conversation.append({"role": "assistant", "content": confirm_phrase})
        self.last_response = confirm_phrase
        await self._speak(confirm_phrase, on_response, on_speech_audio)

        self.pending_call_request = CallRequest(
            requested_number=match.phone_number,
            caller_intent=name,
            confidence=match.confidence,
            timestamp=time.time(),
            business_name=match.name,
        )
        self.state = OperatorState.CONFIRMING
        self.logger.info(
            f"Select '{name}' → '{match.name}' ({match.phone_number}), state=confirming"
        )

    async def announce_and_reset(
        self,
        text: str,
        on_response: Callable[[str], None],
        on_speech_audio: Callable[[bytes], None],
    ) -> None:
        """Speak a one-off announcement, then reset the conversation to IDLE."""
        await self._speak(text, on_response, on_speech_audio)
        self.reset_conversation()

    def reset_conversation(self) -> None:
        self.current_conversation.clear()
        self.state = OperatorState.IDLE
        self.pending_call_request = None
        self._last_lookup_results = []
        self._last_lookup_index = 0
        self.logger.info("Conversation reset")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "providers": {
                "stt": self.stt_provider.is_available if self.stt_provider else False,
                "tts": self.tts_provider.is_available if self.tts_provider else False,
                "conversation": self.conversation_provider.is_available if self.conversation_provider else False,
                "lookup": sum(1 for p in self.lookup_providers if p.is_available),
            },
            "conversation_length": len(self.current_conversation),
            "last_response": self.last_response,
        }


async def main():
    """Smoke-test the operator without hardware."""
    logging.basicConfig(level=logging.INFO)
    operator = AIOperator()
    print("Status:", operator.get_status())

    responses = []
    await operator.process_user_request(
        "I'd like to call 555-1234 please",
        on_response=responses.append,
        on_speech_audio=lambda _: None,
    )
    print("Response:", responses)


if __name__ == "__main__":
    asyncio.run(main())
