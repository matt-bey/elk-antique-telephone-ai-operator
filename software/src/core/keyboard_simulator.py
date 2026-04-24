"""
Keyboard Simulator - Antique Telephone AI Operator

Maps keyboard keypresses to GPIO simulation events for dev-machine testing.
Used automatically when GPIO hardware is not available (HAS_GPIO=False).
"""

import asyncio
import sys
import logging
import termios
import tty
from typing import Callable, Optional

from core.gpio_monitor import GPIOMonitor


class KeyboardSimulator:
    """
    Maps keyboard keypresses to GPIO simulation events.

    Runs on the asyncio event loop thread via add_reader — no executor threading,
    so asyncio.create_task() calls from GPIO handlers are safe.

    Uses setcbreak (not setraw) so output newlines render correctly in logs.

    Keys:
      c / Enter  — crank turn (summon the operator)
      h          — toggle hook state (pick up / hang up)
      q / Ctrl+C — quit
    """

    HELP = (
        "\n--- Keyboard Controls (simulation mode) ---\n"
        "  c / Enter : Crank turn  →  summon operator\n"
        "  h         : Toggle hook →  pick up / hang up\n"
        "  q / Ctrl+C: Quit\n"
        "-------------------------------------------\n"
    )

    def __init__(self, gpio_monitor: GPIOMonitor, on_quit: Optional[Callable] = None):
        self.gpio_monitor = gpio_monitor
        self.on_quit = on_quit
        self.logger = logging.getLogger(__name__)
        self._hook_off = False
        self._fd: Optional[int] = None
        self._old_settings = None

    def start(self) -> None:
        """Enter cbreak mode and register stdin as a readable source on the event loop."""
        if not sys.stdin.isatty():
            self.logger.warning("stdin is not a TTY — keyboard simulator disabled")
            return

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        asyncio.get_event_loop().add_reader(self._fd, self._on_stdin_ready)
        print(self.HELP, flush=True)
        self.logger.info("Keyboard simulator active")

    def stop(self) -> None:
        """Restore terminal settings and remove stdin reader."""
        if self._fd is not None:
            try:
                asyncio.get_event_loop().remove_reader(self._fd)
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception as e:
                self.logger.warning(f"Keyboard simulator cleanup error: {e}")
            finally:
                self._fd = None
                print(flush=True)  # restore cursor position
        self.logger.info("Keyboard simulator stopped")

    def _on_stdin_ready(self) -> None:
        """Called on the event loop thread when stdin has data."""
        try:
            char = sys.stdin.read(1)
        except OSError:
            return
        self._handle_key(char)

    def _handle_key(self, char: str) -> None:
        if char in ('c', '\r', '\n'):
            print("[KEY] Crank turn", flush=True)
            self.logger.info("Keyboard: crank turn triggered")
            # In simulation, cranking implies the earpiece is off-hook.
            # Set both local state and GPIO monitor state so the
            # off-hook check in _on_crank_turn passes, and the first
            # H press correctly transitions to on-hook (hangup).
            if not self._hook_off:
                self._hook_off = True
                self.gpio_monitor.hook_state = True
            self.gpio_monitor.simulate_crank_turn()

        elif char == 'h':
            self._hook_off = not self._hook_off
            state = "off-hook (picked up)" if self._hook_off else "on-hook (hung up)"
            print(f"[KEY] Hook → {state}", flush=True)
            self.logger.info(f"Keyboard: hook state → {state}")
            self.gpio_monitor.simulate_hook_change(self._hook_off)

        elif char in ('q', '\x03'):  # q or Ctrl+C
            print("[KEY] Quit", flush=True)
            self.logger.info("Keyboard: quit requested")
            if self.on_quit:
                self.on_quit()
