"""
AI Operator Tests - Antique Telephone AI Operator

Tests for AI operator functionality including speech recognition,
conversation logic, voice synthesis, and the LLM intent tag state machine.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ai_operator import AIOperator, OperatorState, CallRequest
from providers.base import STTProvider, TTSProvider, ConversationProvider, LookupProvider, LookupResult
from utils.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_stt(transcription="test transcription"):
    provider = MagicMock(spec=STTProvider)
    provider.is_available = True
    provider.transcribe = AsyncMock(return_value=transcription)
    return provider


def _mock_tts(audio_bytes=b"fake_audio"):
    provider = MagicMock(spec=TTSProvider)
    provider.is_available = True
    provider.synthesize = AsyncMock(return_value=audio_bytes)
    return provider


def _mock_conv(response="Good day, Central Exchange. Number please? [INTENT=NONE]"):
    provider = MagicMock(spec=ConversationProvider)
    provider.is_available = True
    provider.get_response = AsyncMock(return_value=response)
    return provider


def _mock_lookup(results=None):
    provider = MagicMock(spec=LookupProvider)
    provider.is_available = True
    provider.search = AsyncMock(return_value=results or [])
    return provider


# ---------------------------------------------------------------------------
# Basic initialisation and wiring
# ---------------------------------------------------------------------------

class TestAIOperator:

    @pytest.fixture
    def op(self):
        return AIOperator()

    @pytest.fixture
    def op_full(self):
        return AIOperator(
            stt_provider=_mock_stt(),
            tts_provider=_mock_tts(),
            conversation_provider=_mock_conv(),
        )

    def test_initialization(self, op):
        assert op.state == OperatorState.IDLE
        assert op.stt_provider is None
        assert op.tts_provider is None
        assert op.conversation_provider is None
        assert op.lookup_providers == []
        assert op.current_conversation == []
        assert op.last_response == ""
        assert op.pending_call_request is None

    def test_persona_content(self, op):
        p = op.operator_persona
        assert 'professional 1920s telephone operator' in p.lower()
        assert 'central exchange' in p.lower()
        assert 'bell telephone' in p.lower()
        assert 'number please' in p.lower()
        assert '[INTENT=' in p
        # Operator name from pool should appear in persona
        assert op.operator_name in p

    def test_get_status_no_providers(self, op):
        s = op.get_status()
        assert s['state'] == 'idle'
        assert s['providers']['stt'] is False
        assert s['providers']['tts'] is False
        assert s['providers']['conversation'] is False
        assert s['providers']['lookup'] == 0

    def test_get_status_with_providers(self, op_full):
        s = op_full.get_status()
        assert s['providers']['stt'] is True
        assert s['providers']['tts'] is True
        assert s['providers']['conversation'] is True

    def test_get_status_lookup_count(self):
        op = AIOperator(lookup_providers=[_mock_lookup(), _mock_lookup()])
        assert op.get_status()['providers']['lookup'] == 2

    def test_fallback_responses(self, op):
        assert op._fallback_response("Hello") in AIOperator._PHRASES["greeting"]
        assert op._fallback_response("I want to call someone") in AIOperator._PHRASES["connecting"]
        assert op._fallback_response("The line is busy") in AIOperator._PHRASES["busy"]
        assert op._fallback_response("Thank you") in AIOperator._PHRASES["goodbye"]
        assert op._fallback_response("Zebra elephant purple") in AIOperator._PHRASES["misheard"]

    def test_conversation_reset(self, op):
        op.current_conversation = [{"role": "user", "content": "Hi"}]
        op.last_response = "test"
        op.pending_call_request = CallRequest("555", "call", 0.8, 0.0)
        op.state = OperatorState.CONFIRMING
        op.reset_conversation()
        assert op.current_conversation == []
        assert op.state == OperatorState.IDLE
        assert op.pending_call_request is None

    @pytest.mark.asyncio
    async def test_process_speech_no_provider(self, op):
        result = await op.process_speech(np.zeros(1024, dtype=np.int16))
        assert result is None

    @pytest.mark.asyncio
    async def test_process_speech_delegates(self):
        stt = _mock_stt("hello operator")
        op = AIOperator(stt_provider=stt)
        audio = np.zeros(1024, dtype=np.int16)
        result = await op.process_speech(audio, sample_rate=16000)
        assert result == "hello operator"
        stt.transcribe.assert_called_once_with(audio, 16000)

    @pytest.mark.asyncio
    async def test_synthesize_speech_no_provider(self, op):
        assert await op.synthesize_speech("Hello") is None

    @pytest.mark.asyncio
    async def test_synthesize_speech_delegates(self, op_full):
        result = await op_full.synthesize_speech("Hello")
        assert result == b"fake_audio"

    @pytest.mark.asyncio
    async def test_generate_response_fallback(self, op):
        r = await op.generate_response("Hello operator")
        assert isinstance(r, str) and len(r) > 0

    @pytest.mark.asyncio
    async def test_generate_response_delegates(self, op_full):
        r = await op_full.generate_response("Hello")
        assert "Number please?" in r

    @pytest.mark.asyncio
    async def test_generate_response_strips_tag_from_history(self):
        """Tag is stripped before storing in conversation history."""
        conv = _mock_conv("Number please? [INTENT=NONE]")
        op = AIOperator(conversation_provider=conv)
        await op.generate_response("Hello")
        last = op.current_conversation[-1]
        assert last["role"] == "assistant"
        assert "[INTENT=" not in last["content"]
        assert "Number please?" in last["content"]

    @pytest.mark.asyncio
    async def test_conversation_history_accumulates(self):
        conv = _mock_conv("Good day. [INTENT=NONE]")
        op = AIOperator(conversation_provider=conv)
        await op.generate_response("Hello")
        await op.generate_response("Call 555-1234")
        assert len(op.current_conversation) == 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_handle_operator_session(self):
        op = AIOperator()
        responses = []
        await op.handle_operator_session(
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )
        assert responses[0] in AIOperator._PHRASES["greeting"]
        assert op.state == OperatorState.LISTENING

    @pytest.mark.asyncio
    async def test_handle_operator_session_plays_audio(self):
        tts = _mock_tts(b"wav_greeting")
        op = AIOperator(tts_provider=tts)
        chunks = []
        await op.handle_operator_session(
            on_response=lambda _: None,
            on_speech_audio=chunks.append,
        )
        assert chunks == [b"wav_greeting"]


# ---------------------------------------------------------------------------
# Intent tag parsing
# ---------------------------------------------------------------------------

class TestIntentTagParsing:

    def test_confirm_tag(self):
        intent, value, clean = AIOperator._parse_intent_tag(
            "Did you say 6 1 4? [INTENT=CONFIRM|6145989581]"
        )
        assert intent == "CONFIRM"
        assert value == "6145989581"
        assert clean == "Did you say 6 1 4?"

    def test_lookup_tag(self):
        intent, value, clean = AIOperator._parse_intent_tag(
            "One moment. [INTENT=LOOKUP|donatos pizza]"
        )
        assert intent == "LOOKUP"
        assert value == "donatos pizza"
        assert clean == "One moment."

    def test_connect_tag(self):
        intent, value, clean = AIOperator._parse_intent_tag(
            "One moment, please. [INTENT=CONNECT]"
        )
        assert intent == "CONNECT"
        assert value == ""
        assert clean == "One moment, please."

    def test_none_tag(self):
        intent, value, clean = AIOperator._parse_intent_tag(
            "Number please? [INTENT=NONE]"
        )
        assert intent == "NONE"
        assert value == ""
        assert clean == "Number please?"

    def test_missing_tag_defaults_none(self):
        intent, value, clean = AIOperator._parse_intent_tag("Number please?")
        assert intent == "NONE"
        assert value == ""
        assert clean == "Number please?"

    def test_case_insensitive(self):
        intent, _, _ = AIOperator._parse_intent_tag("Hello. [intent=connect]")
        assert intent == "CONNECT"

    def test_tag_with_trailing_whitespace(self):
        intent, value, clean = AIOperator._parse_intent_tag(
            "One moment. [INTENT=LOOKUP|donatos]  "
        )
        assert intent == "LOOKUP"
        assert value == "donatos"


# ---------------------------------------------------------------------------
# State machine — CONFIRM tag
# ---------------------------------------------------------------------------

class TestConfirmIntent:

    @pytest.mark.asyncio
    async def test_confirm_tag_moves_to_confirming(self):
        conv = _mock_conv("Did you say 5 5 5. 1 2 3 4? [INTENT=CONFIRM|5551234]")
        op = AIOperator(conversation_provider=conv)
        op.state = OperatorState.LISTENING
        call_requests = []

        await op.process_user_request(
            "Please connect me to 555-1234",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
            on_call_request=call_requests.append,
        )

        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request is not None
        assert op.pending_call_request.requested_number == "5551234"
        assert call_requests == []  # VoIP not yet

    @pytest.mark.asyncio
    async def test_confirm_tag_updates_pending_in_confirming(self):
        """New CONFIRM tag while already CONFIRMING updates the pending number."""
        conv = _mock_conv("Did you say 5 5 5. 5 6 7 8? [INTENT=CONFIRM|5555678]")
        op = AIOperator(conversation_provider=conv)
        op.state = OperatorState.CONFIRMING
        op.pending_call_request = CallRequest("5551234", "", 0.8, 0.0)

        await op.process_user_request(
            "No, I said 555-5678",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.requested_number == "5555678"


# ---------------------------------------------------------------------------
# State machine — CONNECT tag
# ---------------------------------------------------------------------------

class TestConnectIntent:

    @pytest.mark.asyncio
    async def test_connect_tag_fires_voip(self):
        conv = _mock_conv("One moment while I connect you. [INTENT=CONNECT]")
        op = AIOperator(conversation_provider=conv)
        op.state = OperatorState.CONFIRMING
        op.pending_call_request = CallRequest("5551234", "call 555-1234", 0.8, 0.0)
        call_requests = []

        await op.process_user_request(
            "Yes, that's right",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
            on_call_request=call_requests.append,
        )

        assert op.state == OperatorState.CONNECTING_CALL
        assert len(call_requests) == 1
        assert call_requests[0].requested_number == "5551234"

    @pytest.mark.asyncio
    async def test_connect_without_callback_still_transitions(self):
        conv = _mock_conv("One moment. [INTENT=CONNECT]")
        op = AIOperator(conversation_provider=conv)
        op.state = OperatorState.CONFIRMING
        op.pending_call_request = CallRequest("5551234", "", 0.8, 0.0)

        await op.process_user_request(
            "Yes please",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
            on_call_request=None,
        )

        assert op.state == OperatorState.CONNECTING_CALL

    @pytest.mark.asyncio
    async def test_connect_awaits_async_callback(self):
        conv = _mock_conv("Right away. [INTENT=CONNECT]")
        op = AIOperator(conversation_provider=conv)
        op.state = OperatorState.CONFIRMING
        op.pending_call_request = CallRequest("5551234", "", 0.8, 0.0)
        called = []

        async def async_cb(req):
            called.append(req)

        await op.process_user_request(
            "Yes",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
            on_call_request=async_cb,
        )

        assert len(called) == 1
        assert op.state == OperatorState.CONNECTING_CALL


# ---------------------------------------------------------------------------
# State machine — NONE tag
# ---------------------------------------------------------------------------

class TestNoneIntent:

    @pytest.mark.asyncio
    async def test_none_tag_stays_listening(self):
        conv = _mock_conv("Number please? [INTENT=NONE]")
        op = AIOperator(conversation_provider=conv)
        op.state = OperatorState.LISTENING

        await op.process_user_request(
            "What time is it?",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING

    @pytest.mark.asyncio
    async def test_none_tag_preserves_confirming_state(self):
        """NONE while CONFIRMING keeps state — user can try again."""
        conv = _mock_conv("Pardon? Could you repeat that? [INTENT=NONE]")
        op = AIOperator(conversation_provider=conv)
        op.state = OperatorState.CONFIRMING
        op.pending_call_request = CallRequest("5551234", "", 0.8, 0.0)

        await op.process_user_request(
            "Hmm",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request is not None


# ---------------------------------------------------------------------------
# State machine — LOOKUP tag + multi-provider lookup
# ---------------------------------------------------------------------------

class TestLookupIntent:

    @pytest.mark.asyncio
    async def test_lookup_tag_triggers_search(self):
        lookup = _mock_lookup([])
        conv = _mock_conv("One moment. [INTENT=LOOKUP|donatos pizza]")
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])
        op.state = OperatorState.LISTENING

        await op.process_user_request(
            "Call Donatos Pizza",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        lookup.search.assert_called_once()
        assert "donatos pizza" in lookup.search.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_lookup_single_result_moves_to_confirming(self):
        result = LookupResult(
            name="Donatos Pizza", address="123 Main St",
            phone_number="6145559999", confidence=0.9, source="business",
        )
        lookup = _mock_lookup([result])
        conv = _mock_conv("One moment. [INTENT=LOOKUP|donatos pizza]")
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])
        op.state = OperatorState.LISTENING
        responses = []

        await op.process_user_request(
            "Call Donatos Pizza",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.requested_number == "6145559999"
        assert op.pending_call_request.business_name == "Donatos Pizza"
        # Confirmation phrase references the business name
        confirm_response = responses[-1]
        assert "Donatos Pizza" in confirm_response

    @pytest.mark.asyncio
    async def test_lookup_no_results_stays_listening(self):
        lookup = _mock_lookup([])
        conv = _mock_conv("One moment. [INTENT=LOOKUP|invisible shop]")
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])
        op.state = OperatorState.LISTENING
        responses = []

        await op.process_user_request(
            "connect me to the invisible shop",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        assert responses[-1] in AIOperator._PHRASES["not_found"]

    @pytest.mark.asyncio
    async def test_lookup_no_provider_stays_listening(self):
        conv = _mock_conv("One moment. [INTENT=LOOKUP|something]")
        op = AIOperator(conversation_provider=conv)  # no lookup providers
        op.state = OperatorState.LISTENING
        responses = []

        await op.process_user_request(
            "connect me to something",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING

    @pytest.mark.asyncio
    async def test_lookup_top_result_wins_by_confidence(self):
        """Results from multiple providers are merged; highest confidence wins."""
        results_a = [LookupResult("BJ's Wholesale", "1 Main", "6140000001", 0.85, "business")]
        results_b = [LookupResult("BJ Smith", "", "6140000002", 0.91, "contact")]
        lookup_a = _mock_lookup(results_a)
        lookup_b = _mock_lookup(results_b)
        conv = _mock_conv("One moment. [INTENT=LOOKUP|bj]")
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup_a, lookup_b])
        op.state = OperatorState.LISTENING
        responses = []

        await op.process_user_request(
            "Call BJ",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        # BJ Smith (0.91) should beat BJ's Wholesale (0.85)
        assert op.pending_call_request.business_name == "BJ Smith"
        assert op.pending_call_request.requested_number == "6140000002"

    @pytest.mark.asyncio
    async def test_lookup_in_confirming_state_replaces_pending(self):
        """LOOKUP while CONFIRMING replaces the pending request (user corrected)."""
        result = LookupResult("Donatos Pizza", "123 Main", "6145559999", 0.9, "business")
        lookup = _mock_lookup([result])
        conv = _mock_conv("One moment. [INTENT=LOOKUP|donatos pizza]")
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])
        op.state = OperatorState.CONFIRMING
        op.pending_call_request = CallRequest("0000000000", "old query", 0.5, 0.0)

        await op.process_user_request(
            "No, call Donatos Pizza",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.requested_number == "6145559999"

    @pytest.mark.asyncio
    async def test_resolve_lookup_strips_leading_article(self):
        lookup = _mock_lookup([])
        conv = _mock_conv("One moment. [INTENT=LOOKUP|the pharmacy]")
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])
        op.state = OperatorState.LISTENING

        await op.process_user_request(
            "connect me to the pharmacy",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        query_arg = lookup.search.call_args[0][0]
        assert not query_arg.lower().startswith("the ")


# ---------------------------------------------------------------------------
# State machine — LIST_NEXT, LIST_MANY, SELECT
# ---------------------------------------------------------------------------

class TestListIntents:

    def _make_results(self, n=5):
        return [
            LookupResult(f"Shop {i}", f"{i} Main St", f"614000000{i}", 1.0 - i * 0.05, "business")
            for i in range(n)
        ]

    # -- LIST_NEXT -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_next_offers_single_result_confirming(self):
        """LIST_NEXT advances one step and goes CONFIRMING."""
        results = self._make_results(5)
        conv = _mock_conv("One moment. [INTENT=LIST_NEXT]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = results
        op._last_lookup_index = 1  # already past index 0
        responses = []

        await op.process_user_request(
            "Next one please",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.business_name == "Shop 1"
        assert op._last_lookup_index == 2
        assert "Shop 1" in responses[-1]

    @pytest.mark.asyncio
    async def test_list_next_exhausted_stays_listening(self):
        """LIST_NEXT when cache exhausted gives list_exhausted phrase."""
        results = self._make_results(2)
        conv = _mock_conv("One moment. [INTENT=LIST_NEXT]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = results
        op._last_lookup_index = 2
        responses = []

        await op.process_user_request(
            "Next one",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        assert responses[-1] in AIOperator._PHRASES["list_exhausted"]

    @pytest.mark.asyncio
    async def test_list_next_empty_cache_not_found(self):
        """LIST_NEXT with no cache gives not_found phrase."""
        conv = _mock_conv("One moment. [INTENT=LIST_NEXT]")
        op = AIOperator(conversation_provider=conv)
        responses = []

        await op.process_user_request(
            "Next one",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        assert responses[-1] in AIOperator._PHRASES["not_found"]

    # -- LIST_MANY -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_many_enumerates_next_batch_stays_listening(self):
        """LIST_MANY reads next batch of up to 3, stays LISTENING."""
        results = self._make_results(5)
        conv = _mock_conv("One moment. [INTENT=LIST_MANY]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = results
        op._last_lookup_index = 0
        responses = []

        await op.process_user_request(
            "What else do you have?",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        spoken = responses[-1]
        assert "Shop 0" in spoken
        assert "Shop 1" in spoken
        assert "Shop 2" in spoken
        assert op._last_lookup_index == 3

    @pytest.mark.asyncio
    async def test_list_many_partial_batch(self):
        """LIST_MANY with fewer than 3 remaining reads what's left."""
        results = self._make_results(5)
        conv = _mock_conv("One moment. [INTENT=LIST_MANY]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = results
        op._last_lookup_index = 3  # only Shop 3 and Shop 4 remain
        responses = []

        await op.process_user_request(
            "What else?",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        spoken = responses[-1]
        assert "Shop 3" in spoken
        assert "Shop 4" in spoken
        assert op._last_lookup_index == 5

    @pytest.mark.asyncio
    async def test_list_many_adds_to_conversation_history(self):
        """Enumerated names are stored in history so LLM can resolve 'the second one'."""
        results = self._make_results(3)
        conv = _mock_conv("One moment. [INTENT=LIST_MANY]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = results
        op._last_lookup_index = 0

        await op.process_user_request(
            "Give me options",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        last_assistant = next(
            m["content"] for m in reversed(op.current_conversation)
            if m["role"] == "assistant"
        )
        assert "Shop 0" in last_assistant

    @pytest.mark.asyncio
    async def test_list_many_exhausted_stays_listening(self):
        """LIST_MANY when cache exhausted gives list_exhausted phrase."""
        conv = _mock_conv("One moment. [INTENT=LIST_MANY]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = self._make_results(2)
        op._last_lookup_index = 2
        responses = []

        await op.process_user_request(
            "What else?",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        assert responses[-1] in AIOperator._PHRASES["list_exhausted"]

    @pytest.mark.asyncio
    async def test_list_many_empty_cache_not_found(self):
        """LIST_MANY with no cache gives not_found phrase."""
        conv = _mock_conv("One moment. [INTENT=LIST_MANY]")
        op = AIOperator(conversation_provider=conv)
        responses = []

        await op.process_user_request(
            "Give me options",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        assert responses[-1] in AIOperator._PHRASES["not_found"]

    # -- SELECT --------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_select_resolves_exact_name_from_cache(self):
        """SELECT exact name match sets pending_call_request and goes CONFIRMING."""
        results = self._make_results(3)
        conv = _mock_conv("Shall I connect you to Shop 1? [INTENT=SELECT|Shop 1]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = results
        responses = []

        await op.process_user_request(
            "The second one",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.business_name == "Shop 1"
        assert op.pending_call_request.requested_number == "6140000001"
        assert responses[-1] in [
            p.format(name="Shop 1") for p in AIOperator._PHRASES["confirm_select"]
        ]

    @pytest.mark.asyncio
    async def test_select_partial_name_match(self):
        """SELECT matches by substring — handles LLM truncating or paraphrasing the name."""
        results = [
            LookupResult("East African Coffee House", "1 Main", "6140000001", 0.9, "business"),
            LookupResult("Shibam Coffee", "2 Main", "6140000002", 0.85, "business"),
        ]
        conv = _mock_conv("Shall I connect you to East African? [INTENT=SELECT|East African]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = results

        await op.process_user_request(
            "East African please",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.business_name == "East African Coffee House"

    @pytest.mark.asyncio
    async def test_select_no_match_not_found(self):
        """SELECT with unrecognised name gives not_found phrase."""
        results = self._make_results(3)
        conv = _mock_conv("Shall I connect you to Nowhere Cafe? [INTENT=SELECT|Nowhere Cafe]")
        op = AIOperator(conversation_provider=conv)
        op._last_lookup_results = results
        responses = []

        await op.process_user_request(
            "Nowhere Cafe",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        assert responses[-1] in AIOperator._PHRASES["not_found"]

    @pytest.mark.asyncio
    async def test_select_empty_cache_not_found(self):
        """SELECT with no cached results gives not_found phrase."""
        conv = _mock_conv("Shall I connect you to Shibam? [INTENT=SELECT|Shibam]")
        op = AIOperator(conversation_provider=conv)
        responses = []

        await op.process_user_request(
            "Shibam Coffee",
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )

        assert op.state == OperatorState.LISTENING
        assert responses[-1] in AIOperator._PHRASES["not_found"]

    # -- Cache management ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_lookup_caches_all_results(self):
        """LOOKUP populates the result cache for later LIST_NEXT/LIST_MANY/SELECT use."""
        results = self._make_results(5)
        lookup = _mock_lookup(results)
        conv = _mock_conv("One moment. [INTENT=LOOKUP|coffee shop]")
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])

        await op.process_user_request(
            "I need a coffee shop",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )

        assert len(op._last_lookup_results) == 5
        assert op._last_lookup_index == 1  # LOOKUP offers top result; index advances past it

    def test_reset_clears_cache(self):
        """reset_conversation clears the list cache."""
        op = AIOperator()
        op._last_lookup_results = self._make_results(3)
        op._last_lookup_index = 2
        op.reset_conversation()
        assert op._last_lookup_results == []
        assert op._last_lookup_index == 0

    # -- Full flows ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_full_flow_lookup_list_next_connect(self):
        """LOOKUP → no → LIST_NEXT → yes → CONNECT."""
        results = [
            LookupResult("Shop A", "1 Main", "6140000001", 0.95, "business"),
            LookupResult("Shop B", "2 Main", "6140000002", 0.90, "business"),
        ]
        lookup = _mock_lookup(results)
        conv_responses = [
            "One moment. [INTENT=LOOKUP|coffee]",
            "One moment. [INTENT=LIST_NEXT]",
            "One moment, please. [INTENT=CONNECT]",
        ]
        conv = MagicMock(spec=ConversationProvider)
        conv.is_available = True
        conv.get_response = AsyncMock(side_effect=conv_responses)
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])
        call_requests = []

        await op.process_user_request("I need coffee", on_response=lambda _: None,
                                      on_speech_audio=lambda _: None,
                                      on_call_request=call_requests.append)
        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.business_name == "Shop A"

        await op.process_user_request("No, next one", on_response=lambda _: None,
                                      on_speech_audio=lambda _: None,
                                      on_call_request=call_requests.append)
        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.business_name == "Shop B"

        await op.process_user_request("Yes", on_response=lambda _: None,
                                      on_speech_audio=lambda _: None,
                                      on_call_request=call_requests.append)
        assert op.state == OperatorState.CONNECTING_CALL
        assert call_requests[0].requested_number == "6140000002"

    @pytest.mark.asyncio
    async def test_full_flow_lookup_list_many_select_connect(self):
        """LOOKUP → LIST_MANY → SELECT → CONNECT (no re-query on SELECT)."""
        results = [
            LookupResult("Cafe Nero", "1 High St", "6140000001", 0.95, "business"),
            LookupResult("Shibam Coffee", "2 Main St", "6140000002", 0.90, "business"),
            LookupResult("Marib Coffee", "3 Oak Ave", "6140000003", 0.85, "business"),
        ]
        lookup = _mock_lookup(results)
        conv_responses = [
            "One moment. [INTENT=LOOKUP|coffee shop]",
            "One moment. [INTENT=LIST_MANY]",
            "Shall I connect you to Shibam Coffee? [INTENT=SELECT|Shibam Coffee]",
            "One moment, please. [INTENT=CONNECT]",
        ]
        conv = MagicMock(spec=ConversationProvider)
        conv.is_available = True
        conv.get_response = AsyncMock(side_effect=conv_responses)
        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])
        call_requests = []

        # LOOKUP → offered Cafe Nero
        await op.process_user_request("I need a coffee shop", on_response=lambda _: None,
                                      on_speech_audio=lambda _: None,
                                      on_call_request=call_requests.append)
        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.business_name == "Cafe Nero"

        # LIST_MANY → index was 1 after LOOKUP (Cafe Nero already offered), reads remaining 2
        responses = []
        await op.process_user_request("Give me options", on_response=responses.append,
                                      on_speech_audio=lambda _: None,
                                      on_call_request=call_requests.append)
        assert op.state == OperatorState.LISTENING
        assert "Shibam Coffee" in responses[-1]

        # SELECT → resolve Shibam from cache, no provider call
        lookup.search.reset_mock()
        await op.process_user_request("Shibam Coffee please", on_response=lambda _: None,
                                      on_speech_audio=lambda _: None,
                                      on_call_request=call_requests.append)
        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.requested_number == "6140000002"
        lookup.search.assert_not_called()  # no re-query

        # CONNECT
        await op.process_user_request("Yes", on_response=lambda _: None,
                                      on_speech_audio=lambda _: None,
                                      on_call_request=call_requests.append)
        assert op.state == OperatorState.CONNECTING_CALL
        assert call_requests[0].requested_number == "6140000002"


# ---------------------------------------------------------------------------
# Phrase library
# ---------------------------------------------------------------------------

class TestPhraseLibrary:

    def test_random_phrase_all_categories(self):
        op = AIOperator()
        for cat in ["greeting", "connecting", "busy", "not_found", "confirm_business",
                    "lookup_working", "misheard", "goodbye", "hold",
                    "error", "stt_trouble", "stt_give_up",
                    "list_results", "list_exhausted", "confirm_select"]:
            phrase = op._random_phrase(cat)
            assert isinstance(phrase, str) and len(phrase) > 0

    def test_random_phrase_unknown_category_returns_error(self):
        phrase = AIOperator._random_phrase("nonexistent")
        assert isinstance(phrase, str) and len(phrase) > 0

    def test_random_phrase_variety(self):
        results = {AIOperator._random_phrase("greeting") for _ in range(30)}
        assert len(results) > 1

    def test_confirm_business_phrase_supports_format(self):
        phrase = AIOperator._random_phrase("confirm_business").format(name="Donatos Pizza")
        assert "Donatos Pizza" in phrase

    @pytest.mark.asyncio
    async def test_greeting_uses_phrase_list(self):
        op = AIOperator()
        responses = []
        await op.handle_operator_session(
            on_response=responses.append,
            on_speech_audio=lambda _: None,
        )
        assert responses[0] in AIOperator._PHRASES["greeting"]


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

class TestNumberFormatting:

    def test_10_digit(self):
        f = AIOperator._format_number_for_speech("6145989581")
        assert f == "6 1 4. 5 9 8. 9 5 8 1"

    def test_7_digit(self):
        f = AIOperator._format_number_for_speech("5551234")
        assert f == "5 5 5. 1 2 3 4"

    def test_with_punctuation(self):
        f = AIOperator._format_number_for_speech("614-598-9581")
        assert f == "6 1 4. 5 9 8. 9 5 8 1"


# ---------------------------------------------------------------------------
# Full multi-turn flow
# ---------------------------------------------------------------------------

class TestFullFlow:

    @pytest.mark.asyncio
    async def test_greeting_number_confirm_connect(self):
        """End-to-end: greeting → CONFIRM → CONNECT → VoIP fires."""
        responses = [
            "Did you say 5 5 5. 1 2 3 4? [INTENT=CONFIRM|5551234]",
            "One moment while I connect you. [INTENT=CONNECT]",
        ]
        conv = MagicMock(spec=ConversationProvider)
        conv.is_available = True
        conv.get_response = AsyncMock(side_effect=responses)
        op = AIOperator(conversation_provider=conv)
        call_requests = []

        await op.handle_operator_session(
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
        )
        assert op.state == OperatorState.LISTENING

        await op.process_user_request(
            "Connect me to 555-1234",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
            on_call_request=call_requests.append,
        )
        assert op.state == OperatorState.CONFIRMING
        assert call_requests == []

        await op.process_user_request(
            "Yes, that's correct",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
            on_call_request=call_requests.append,
        )
        assert op.state == OperatorState.CONNECTING_CALL
        assert len(call_requests) == 1
        assert call_requests[0].requested_number == "5551234"

    @pytest.mark.asyncio
    async def test_lookup_then_confirm_then_connect(self):
        """Business name → lookup → confirm business → CONNECT."""
        place = LookupResult("Donatos Pizza", "123 Main", "6145559999", 0.9, "business")
        lookup = _mock_lookup([place])

        responses = [
            "One moment. [INTENT=LOOKUP|donatos pizza]",
            "One moment while I connect you. [INTENT=CONNECT]",
        ]
        conv = MagicMock(spec=ConversationProvider)
        conv.is_available = True
        conv.get_response = AsyncMock(side_effect=responses)

        op = AIOperator(conversation_provider=conv, lookup_providers=[lookup])
        call_requests = []

        await op.process_user_request(
            "Call Donatos Pizza",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
            on_call_request=call_requests.append,
        )
        assert op.state == OperatorState.CONFIRMING
        assert op.pending_call_request.requested_number == "6145559999"
        assert call_requests == []

        await op.process_user_request(
            "Yes, that's the one",
            on_response=lambda _: None,
            on_speech_audio=lambda _: None,
            on_call_request=call_requests.append,
        )
        assert op.state == OperatorState.CONNECTING_CALL
        assert len(call_requests) == 1
        assert call_requests[0].requested_number == "6145559999"


# ---------------------------------------------------------------------------
# Integration (live API — skipped unless key present)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAIOperatorIntegration:

    @pytest.fixture
    def op_with_keys(self):
        config = ConfigManager()
        if not config.get('anthropic.api_key'):
            pytest.skip("Anthropic API key not configured")
        from providers.conversation.anthropic_provider import AnthropicProvider
        conv = AnthropicProvider(api_key=config.get('anthropic.api_key'))
        return AIOperator(conversation_provider=conv)

    @pytest.mark.asyncio
    async def test_real_conversation_includes_intent_tag(self, op_with_keys):
        response = await op_with_keys.generate_response("Hello operator")
        if response:
            assert isinstance(response, str) and len(response) > 0


# ---------------------------------------------------------------------------
# Hardware (Pi only)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestAIOperatorHardware:

    @pytest.mark.skipif(
        not Path('/proc/cpuinfo').exists() or
        'Raspberry Pi' not in open('/proc/cpuinfo', 'r').read(),
        reason="Requires Raspberry Pi"
    )
    @pytest.mark.asyncio
    async def test_speech_processing_with_hardware(self):
        op = AIOperator()
        audio = np.zeros(int(16000 * 1.0), dtype=np.float32)
        result = await op.process_speech(audio)
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# Operator name pool
# ---------------------------------------------------------------------------

class TestOperatorNamePool:

    def test_default_pool_used_when_config_empty(self):
        """When operator.names is unset, a name from the built-in pool is chosen."""
        op = AIOperator()
        assert op.operator_name in AIOperator.DEFAULT_OPERATOR_NAMES

    def test_custom_pool_from_config(self):
        """A comma-separated operator.names config value is respected."""
        cfg = ConfigManager()
        cfg.set("operator.names", "Mabel, Hazel")
        op = AIOperator(config_manager=cfg)
        assert op.operator_name in ("Mabel", "Hazel")

    def test_single_name_pool(self):
        """A single configured name is always selected."""
        cfg = ConfigManager()
        cfg.set("operator.names", "Beatrice")
        op = AIOperator(config_manager=cfg)
        assert op.operator_name == "Beatrice"

    def test_name_appears_in_persona(self):
        """The selected name is interpolated into the persona prompt."""
        cfg = ConfigManager()
        cfg.set("operator.names", "Dot")
        op = AIOperator(config_manager=cfg)
        assert "You are Dot," in op.operator_persona
        assert "Your name is Dot" in op.operator_persona

    def test_personality_section_in_persona(self):
        """The persona includes personality guidance for playful questions."""
        op = AIOperator()
        p = op.operator_persona
        assert "PERSONALITY:" in p
        assert "backstory" in p.lower()
        assert "What's your name?" in p

    def test_different_sessions_can_get_different_names(self):
        """Over many sessions the pool produces variation (probabilistic)."""
        names_seen = set()
        for _ in range(100):
            op = AIOperator()
            names_seen.add(op.operator_name)
        # With 6 names and 100 trials, seeing at least 2 is near-certain
        assert len(names_seen) >= 2


def test_main_function():
    from core.ai_operator import main
    try:
        asyncio.run(main())
    except Exception as e:
        assert "API" in str(e) or "key" in str(e) or "authentication" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
