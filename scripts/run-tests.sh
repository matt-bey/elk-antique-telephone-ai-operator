#!/bin/bash
# Antique Telephone AI Operator — test runner
#
# Usage:
#   ./scripts/run-tests.sh                run all safe tests
#   ./scripts/run-tests.sh --unit         unit tests only
#   ./scripts/run-tests.sh --coverage     with coverage report
#   ./scripts/run-tests.sh --hardware     include hardware tests (Pi only)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOFTWARE_DIR="$PROJECT_ROOT/software"

if ! command -v uv &>/dev/null; then
    echo "uv not found. Run scripts/setup.sh first."
    exit 1
fi

# Parse arguments
PYTEST_ARGS="-v --tb=short"
HARDWARE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit)       PYTEST_ARGS="$PYTEST_ARGS -m 'not integration and not hardware'"; shift ;;
        -c|--coverage)   PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=term-missing"; shift ;;
        -h|--hardware)   HARDWARE=true; shift ;;
        --help)
            sed -n '2,7p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Skip hardware tests unless on Pi or explicitly requested
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    IS_PI=true
else
    IS_PI=false
    if [ "$HARDWARE" = false ]; then
        PYTEST_ARGS="$PYTEST_ARGS -m 'not hardware'"
    fi
fi

cd "$SOFTWARE_DIR"
uv sync --all-groups --quiet

echo "Running tests from $SOFTWARE_DIR/tests/ ..."
uv run pytest $PYTEST_ARGS tests/
