#!/bin/bash
# Install Python dependencies for Antique Telephone AI Operator
#
# Requires uv: https://docs.astral.sh/uv/
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOFTWARE_DIR="$PROJECT_ROOT/software"

if ! command -v uv &>/dev/null; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$SOFTWARE_DIR"

echo "Installing Python dependencies..."
uv sync --all-groups

echo ""
echo "Checking critical imports..."
uv run python -c "
import importlib

required = [
    ('numpy',       'NumPy'),
    ('pyaudio',     'PyAudio — audio I/O'),
    ('whisper',     'Whisper — speech recognition'),
    ('anthropic',   'Anthropic — conversation AI'),
    ('soxr',        'soxr — audio resampling'),
    ('pyvoip',      'pyVoIP — SIP calling'),
    ('pytest',      'pytest — testing'),
]

optional = [
    ('gpiozero',    'gpiozero — GPIO (simulation on non-Pi)'),
    ('piper_tts',   'Piper TTS — text to speech'),
    ('ollama',      'Ollama — local LLM fallback'),
]

all_ok = True
print('Required:')
for mod, desc in required:
    try:
        importlib.import_module(mod)
        print(f'  ok  {mod:<20} {desc}')
    except ImportError:
        print(f'  MISSING  {mod:<20} {desc}')
        all_ok = False

print()
print('Optional:')
for mod, desc in optional:
    try:
        importlib.import_module(mod)
        print(f'  ok  {mod:<20} {desc}')
    except ImportError:
        print(f'  -   {mod:<20} {desc}  (not available)')

import sys
if not all_ok:
    sys.exit(1)
print()
print('All required dependencies installed.')
"

echo ""
echo "Done. Run: cd $SOFTWARE_DIR && uv run python src/main.py"
