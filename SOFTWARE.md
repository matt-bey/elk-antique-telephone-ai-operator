# Software — Antique Telephone AI Operator

## Architecture

### Core Components

| Component | File | Responsibility |
|---|---|---|
| Main entrypoint | `src/main.py` | Orchestrates all subsystems, lifecycle |
| GPIO Monitor | `src/core/gpio_monitor.py` | Crank/hook detection; ringer control; keyboard simulation on dev |
| Audio Processor | `src/core/audio_processor.py` | Mic capture, speaker output, VoIP audio bridge |
| AI Operator | `src/core/ai_operator.py` | 1920s operator personality — STT → LLM → TTS state machine |
| VoIP Client | `src/core/voip_client.py` | SIP registration and call management |
| pyVoIP Patch | `src/core/pyvoip_patch.py` | Runtime monkey-patches for Callcentric proxy auth |
| RTP Stream | `src/core/rtp_stream.py` | Custom G.711 µ-law codec + UDP RTP sender/receiver |

### Provider Layer

Swap-friendly via `config/ai-service.conf`:

| Provider type | Module | Notes |
|---|---|---|
| STT | `src/providers/stt/whisper_provider.py` | Local Whisper model — no API key |
| LLM | `src/providers/conversation/anthropic_provider.py` | Anthropic Haiku (default) |
| LLM fallback | `src/providers/conversation/ollama_provider.py` | Local Ollama — no internet needed |
| TTS | `src/providers/tts/piper_provider.py` | Local Piper ONNX — auto-downloads voice model |
| Lookup | `src/providers/lookup/google_places_provider.py` | Google Places API |

### Conversation State Machine

```
IDLE → GREETING → LISTENING → PROCESSING → CONFIRMING → CONNECTING_CALL
                                    ↑
                              ERROR (returns to LISTENING)
```

### LLM Intent Tags

The LLM appends a machine-readable tag to every response: `[INTENT=TYPE|value]`. The state
machine reads the tag instead of doing regex/keyword matching on free text. Tags are stripped
before TTS and before storing in conversation history.

| Tag | Meaning |
|---|---|
| `CONFIRM\|digits` | LLM confirmed a phone number with the caller |
| `LOOKUP\|query` | Caller wants to reach a person or business by name |
| `LIST_NEXT` | Caller wants the next single result from a cached lookup |
| `LIST_MANY` | Caller wants a batch of results from a cached lookup |
| `SELECT\|name` | Caller picked a named item from the list (fuzzy-matched) |
| `CONNECT` | Caller confirmed; place the call |
| `NONE` | No state change — conversational turn only |

If the LLM omits the tag, `_parse_intent_tag` returns `NONE` — graceful degradation.

### VoIP Layer

pyVoIP 1.6.8 is the SIP client. Callcentric uses 407 proxy auth instead of 401, and requires
ACK/BYE routing to the SBC contact address rather than the registrar. Seven runtime
monkey-patches in `src/core/pyvoip_patch.py` address these gaps. `src/core/rtp_stream.py`
replaces pyVoIP's RTP layer with a direct int16→µ-law path via numpy (pyVoIP's path loses 8
bits through an 8-bit linear intermediary).

Audio bridge uses `soxr.ResampleStream` (stateful, no chunk-boundary artifacts) at
44.1 kHz ↔ 8 kHz with 882-sample device-rate alignment → 160-sample RTP frames.

## Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.11 (`uv` package manager) |
| STT | OpenAI Whisper (local) |
| LLM | Anthropic Claude Haiku (default); Ollama local fallback |
| TTS | Piper TTS — `en_US-lessac-high` ONNX model (local) |
| VoIP signaling | pyVoIP 1.6.8 (SIP) with runtime Callcentric proxy-auth patch |
| VoIP audio | Custom `RTPStream` — direct int16→µ-law via numpy |
| Audio resampling | `soxr` stateful `ResampleStream` — 44.1 kHz ↔ 8 kHz |
| Contact lookup | Google Contacts API; Google Places API |
| GPIO | RPi.GPIO on Pi; keyboard simulation on dev machine |

## Prerequisites

### Raspberry Pi

```bash
sudo apt install -y \
    python3-dev libasound2-dev portaudio19-dev \
    alsa-utils pulseaudio pulseaudio-utils \
    libsndfile1-dev ffmpeg cmake pkg-config build-essential \
    python3-rpi.gpio gpio raspi-gpio
```

### Mac (dev machine)

```bash
brew install portaudio   # required for PyAudio
```

## Installation

All commands run from the `software/` directory.

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

cd software/

# Install all dependencies
uv sync --all-groups

# Copy env template and fill in keys
cp .env.example .env
nano .env   # set ANTHROPIC_API_KEY at minimum; SIP creds for calling
```

Run the included setup script on first Pi install (installs apt packages, creates log dirs,
adds user to gpio and audio groups):

```bash
./scripts/setup.sh
```

## Configuration

Config files live in `config/` at the project root (one level above `software/`).

| File | Purpose |
|---|---|
| `config/audio.conf` | Sample rate, buffer size, noise gate, device selection |
| `config/gpio.conf` | GPIO pin assignments (BCM), debounce, ringer pulse config |
| `config/ai-service.conf` | LLM provider + model, TTS voice, Whisper model, operator name pool |
| `config/sip.conf` | SIP registrar defaults, codecs, RTP port range |

Key settings to tune:

- `audio.noise_gate` — silence threshold (0–32767). Start at 100; increase in noisy environments.
- `audio.input_gain` — mic pre-gain multiplier. Carbon mic with bias circuit may need 0.5–2.0.
- `whisper.model` — `base` is the default. `small` or `medium` gives better accuracy at cost of CPU.
- `conversation.provider` — `anthropic` (default) or `ollama` (local, no internet required).

## Running

```bash
# From software/ directory
uv run python src/main.py

# Development keyboard controls:
#   c — simulate crank (summon operator)
#   h — simulate hook switch
#   q — quit
```

The system auto-detects whether it's running on a Pi (real GPIO) or dev machine (keyboard simulation).

## Testing

```bash
# From software/ directory

# Full test suite
uv run pytest tests/ -v --tb=short

# With coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Hardware-only tests (Pi required)
uv run pytest tests/ -m hardware -v

# Skip tests that require live APIs
uv run pytest tests/ -m "not integration" -v

# Smoke test imports
uv run python -c "from src.main import AntiquePhoneSystem; print('OK')"
```

Or use the included test runner script:

```bash
./scripts/run-tests.sh           # all tests
./scripts/run-tests.sh --unit    # unit tests only
./scripts/run-tests.sh --coverage
```

## Troubleshooting

### No audio output

```bash
aplay -l                              # list playback devices
alsamixer                             # check volume levels
speaker-test -t sine -f 1000 -c 1    # test output
```

Set the default device in `config/audio.conf` (`output_device`) or in `~/.asoundrc`.

### Microphone not detected

```bash
arecord -l                                              # list recording devices
arecord -D default -f S16_LE -r 44100 -d 5 test.wav    # 5-second test recording
aplay test.wav
```

Adjust gain: `alsamixer → F4` (capture devices).

### GPIO permission denied (Pi)

```bash
sudo usermod -a -G gpio $USER
newgrp gpio
```

### Whisper accuracy poor

Switch to a larger model in `config/ai-service.conf`: `model = small` or `model = medium`.
Each step up roughly doubles RAM usage and inference time.

### SIP registration fails

1. Confirm SIP credentials are set in `software/.env`.
2. Check Callcentric account status in the web portal.
3. Monitor SIP traffic: `sudo tcpdump -i any port 5060 -A`
4. See `adr/003-voip-stack-replacement.md` for known pyVoIP gaps and patch coverage.
