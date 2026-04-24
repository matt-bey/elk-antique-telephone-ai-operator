#!/bin/bash
# Antique Telephone AI Operator — initial setup
#
# Installs system packages, uv, and Python dependencies.
# Safe to run multiple times (idempotent). Run directly on the Pi or dev machine.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOFTWARE_DIR="$PROJECT_ROOT/software"

echo "Antique Telephone AI Operator — Setup"
echo "Project root: $PROJECT_ROOT"
echo "Software dir: $SOFTWARE_DIR"
echo ""

# Detect Raspberry Pi vs dev machine
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    RASPBERRY_PI=true
    echo "Detected Raspberry Pi — hardware mode"
else
    RASPBERRY_PI=false
    echo "Detected development machine — simulation mode"
fi

# Install system packages (Pi / Debian/Ubuntu only)
if [ "$RASPBERRY_PI" = true ]; then
    echo ""
    echo "Installing system packages..."
    sudo apt update
    sudo apt install -y \
        python3-dev libasound2-dev portaudio19-dev \
        alsa-utils pulseaudio pulseaudio-utils \
        libsndfile1-dev ffmpeg cmake pkg-config build-essential \
        curl wget git

    echo "Installing GPIO packages..."
    sudo apt install -y python3-rpi.gpio gpio raspi-gpio

    echo "Adding user to gpio and audio groups..."
    sudo usermod -a -G gpio,audio "$USER"

    echo "Creating log directory..."
    sudo mkdir -p /var/log/antique-telephone
    sudo chown "$USER:$USER" /var/log/antique-telephone
fi

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo ""
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
cd "$SOFTWARE_DIR"
uv sync --all-groups

# Copy .env template if no .env exists
if [ ! -f "$SOFTWARE_DIR/.env" ]; then
    echo ""
    echo "Creating .env from template — edit it to add your API keys:"
    echo "  $SOFTWARE_DIR/.env"
    cp "$SOFTWARE_DIR/.env.example" "$SOFTWARE_DIR/.env"
    chmod 600 "$SOFTWARE_DIR/.env"
fi

# Smoke test
echo ""
echo "Smoke testing imports..."
cd "$SOFTWARE_DIR"
uv run python -c "from src.main import AntiquePhoneSystem; print('OK')"

echo ""
echo "=========================================="
echo "Setup complete."
echo "Next steps:"
echo "  1. Edit $SOFTWARE_DIR/.env with your API keys"
if [ "$RASPBERRY_PI" = true ]; then
    echo "  2. Log out and back in for group membership (gpio, audio)"
    echo "  3. Set ALSA default device in /etc/asound.conf (see DEPLOYMENT.md)"
fi
echo "  Run: cd $SOFTWARE_DIR && uv run python src/main.py"
echo "=========================================="
