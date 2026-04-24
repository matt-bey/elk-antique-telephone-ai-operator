# Antique Telephone AI Operator

**Status:** Active
**Hardware:** Raspberry Pi 5 (8GB)
**Started:** 2025

Convert a Kellogg Chicago antique wood wall telephone (early 1900s) into a fully functional AI-powered communication device. A Raspberry Pi 5 handles crank detection, a 1920s-style AI telephone operator interaction (speech recognition → LLM → TTS), and outgoing/incoming VoIP calls via SIP — all while preserving the authentic look and feel of the original hardware.

## Development Approach

Development is intentionally split into two phases:

1. **Modern hardware phase** — validate all software logic (AI operator, VoIP, audio pipeline) using a USB microphone, speaker, and push buttons before touching the antique hardware
2. **Antique hardware integration phase** — replace modern stand-ins with the original earpiece, carbon microphone, crank mechanism, and ringer

## Features

- 1920s-style AI telephone operator — Whisper STT → Claude Haiku LLM → Piper TTS, all running on the Pi
- State-machine conversation flow driven by structured LLM intent tags (LOOKUP / CONFIRM / CONNECT)
- Contact resolution via Google Contacts and Google Places
- Outgoing VoIP calls over SIP (pyVoIP with a custom RTP µ-law path; soxr resampling between 44.1 kHz mic and 8 kHz G.711)
- Remote answer / hangup detection; the operator narrates disconnects
- Antique hardware preserved: original earpiece, carbon microphone, crank, hook switch, and mechanical ringer are all driven by the Pi — no modern-looking components exposed

## Hardware

See [HARDWARE.md](HARDWARE.md) for the full bill of materials across both phases.

## Architecture

### Software Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 (`uv` package manager) |
| Speech recognition | OpenAI Whisper (local) |
| LLM | Anthropic Claude Haiku (default); Ollama local fallback |
| Text-to-speech | Piper TTS — `en_US-lessac-high` ONNX model (local) |
| VoIP signaling | pyVoIP 1.6.8 (SIP) with runtime monkey-patch for Callcentric proxy auth |
| VoIP audio | Custom `RTPStream` — direct int16→µ-law via numpy (replaces pyVoIP's lossy 8-bit path) |
| Audio resampling | `soxr` stateful `ResampleStream` — 44.1 kHz ↔ 8 kHz, 160-sample frame alignment |
| Contact lookup | Google Contacts API; Google Places API |

### Call Flow (Outgoing)

```
Crank detected
    → AI operator: "Number please?"
    → Whisper STT captures response
    → LLM extracts intent tag (LOOKUP / CONFIRM / CONNECT)
    → LOOKUP: search Google Contacts + Google Places concurrently
    → CONFIRM: operator reads back number, user confirms
    → CONNECT: SIP INVITE → pyVoIP → custom RTP ↔ audio bridge
    → soxr resamples mic audio (44.1 kHz → 8 kHz) for RTP
    → soxr resamples inbound RTP (8 kHz → 44.1 kHz) for speaker
    → Hook on-hook → BYE, bridge stops, operator resets to IDLE
```

### Notable Implementation Details

- **Callcentric proxy auth patch**: pyVoIP assumes 401/WWW-Authenticate; Callcentric uses 407/Proxy-Authenticate. A runtime patch (`pyvoip_patch.py`) handles REGISTER, INVITE, ACK, and BYE against Callcentric's SBC.
- **LLM intent tags**: the LLM appends a structured `[INTENT=TYPE|value]` token to every response; the state machine reads the tag instead of doing regex/keyword matching on free text.
- **Operator name pool**: a random operator name is selected per session; the LLM can answer personality questions in character.

## AI Operator Personality

- **Era**: 1920s telephone operator
- **Voice**: Professional female, Mid-Atlantic accent, measured pace
- **Phrases**: "Number please?", "One moment while I connect you", "I'm sorry, that line is busy", "Please hold the line"

## References

- [HARDWARE.md](HARDWARE.md) — bill of materials, wiring, and power budget
- [SOFTWARE.md](SOFTWARE.md) — setup, configuration, and architecture details
