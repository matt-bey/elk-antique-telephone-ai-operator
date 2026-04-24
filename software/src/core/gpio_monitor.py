"""
GPIO Monitor Module - Antique Telephone AI Operator

Handles GPIO monitoring for crank detection, hook switch monitoring,
and ringer control. Designed for iterative development starting with
modern button simulation before antique hardware integration.
"""

import logging
import time
import threading
from typing import Callable, Optional, Dict, Any
from enum import Enum

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    logging.warning("RPi.GPIO not available - using simulation mode")

from utils.config_manager import ConfigManager


class GPIOEvent(Enum):
    """GPIO event types"""
    CRANK_TURN = "crank_turn"
    HOOK_OFF = "hook_off"
    HOOK_ON = "hook_on"


class GPIOMonitor:
    """
    GPIO monitoring service for antique telephone components
    
    Supports both modern button simulation (Phase 1) and antique
    hardware integration (Phase 4) through configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize GPIO monitor with configuration"""
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # GPIO pin assignments from config
        self.crank_pin = self.config.get('gpio.crank_pin', 18)
        self.hook_pin = self.config.get('gpio.hook_pin', 19)
        self.status_led_pin = self.config.get('gpio.status_led_pin', 20)
        self.ringer_pin = self.config.get('gpio.ringer_pin', 21)
        
        # Debounce timing
        self.debounce_time = self.config.get('gpio.debounce_time', 50) / 1000.0
        
        # Event callbacks
        self.event_handlers: Dict[GPIOEvent, Callable] = {}
        
        # State tracking
        self.hook_state = False  # False = on-hook, True = off-hook
        self.last_crank_time = 0
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # GPIO setup
        self._setup_gpio()
    
    def _setup_gpio(self) -> None:
        """Initialize GPIO configuration"""
        if not HAS_GPIO:
            self.logger.warning("GPIO hardware not available - using simulation")
            return
        
        try:
            # Set GPIO mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Configure input pins with pull-up resistors
            GPIO.setup(self.crank_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.hook_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Configure output pins
            GPIO.setup(self.status_led_pin, GPIO.OUT)
            GPIO.setup(self.ringer_pin, GPIO.OUT)
            
            # Initialize output states
            GPIO.output(self.status_led_pin, GPIO.LOW)
            GPIO.output(self.ringer_pin, GPIO.LOW)
            
            # Set up interrupt handlers
            GPIO.add_event_detect(
                self.crank_pin, 
                GPIO.FALLING, 
                callback=self._crank_callback,
                bouncetime=int(self.debounce_time * 1000)
            )
            
            GPIO.add_event_detect(
                self.hook_pin,
                GPIO.BOTH,
                callback=self._hook_callback,
                bouncetime=int(self.debounce_time * 1000)
            )
            
            self.logger.info("GPIO initialized successfully")
            
        except Exception as e:
            self.logger.error(f"GPIO setup failed: {e}")
            raise
    
    def register_event_handler(self, event: GPIOEvent, handler: Callable) -> None:
        """Register callback for GPIO events"""
        self.event_handlers[event] = handler
        self.logger.debug(f"Registered handler for {event.value}")
    
    def _crank_callback(self, channel: int) -> None:
        """Handle crank turn detection (falling edge)"""
        current_time = time.time()
        
        # Debounce check
        if current_time - self.last_crank_time < self.debounce_time:
            return
        
        self.last_crank_time = current_time
        self.logger.info("Crank turn detected")
        
        # Trigger event handler
        if GPIOEvent.CRANK_TURN in self.event_handlers:
            try:
                self.event_handlers[GPIOEvent.CRANK_TURN]()
            except Exception as e:
                self.logger.error(f"Error in crank event handler: {e}")
    
    def _hook_callback(self, channel: int) -> None:
        """Handle hook switch state changes"""
        if not HAS_GPIO:
            return
        
        # Read current state (inverted because of pull-up)
        new_state = not GPIO.input(self.hook_pin)
        
        if new_state != self.hook_state:
            self.hook_state = new_state
            event = GPIOEvent.HOOK_OFF if new_state else GPIOEvent.HOOK_ON
            
            self.logger.info(f"Hook state changed: {'off-hook' if new_state else 'on-hook'}")
            
            # Trigger event handler
            if event in self.event_handlers:
                try:
                    self.event_handlers[event]()
                except Exception as e:
                    self.logger.error(f"Error in hook event handler: {e}")
    
    def set_status_led(self, state: bool, label: str = "") -> None:
        """Control status LED.

        On dev machine (no GPIO) prints a visible console indicator so
        operator state is legible without enabling DEBUG logging.
        """
        if HAS_GPIO:
            GPIO.output(self.status_led_pin, GPIO.HIGH if state else GPIO.LOW)
            self.logger.debug(f"Status LED: {'ON' if state else 'OFF'}")
        else:
            indicator = f"● {label}" if state else f"○ {label}"
            print(f"[LED] {indicator.strip()}", flush=True)
            self.logger.debug(f"Status LED (sim): {'ON' if state else 'OFF'} {label}")
    
    def trigger_ringer(self, duration: float = 1.0) -> None:
        """Trigger ringer for specified duration"""
        if not HAS_GPIO:
            self.logger.info(f"SIMULATION: Ringer triggered for {duration}s")
            return
        
        def ringer_pulse():
            try:
                GPIO.output(self.ringer_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(self.ringer_pin, GPIO.LOW)
            except Exception as e:
                self.logger.error(f"Ringer control error: {e}")
        
        # Run ringer in separate thread to avoid blocking
        threading.Thread(target=ringer_pulse, daemon=True).start()
        self.logger.info(f"Ringer triggered for {duration}s")
    
    def get_hook_state(self) -> bool:
        """Get current hook switch state"""
        if HAS_GPIO:
            # Update state from hardware
            self.hook_state = not GPIO.input(self.hook_pin)
        return self.hook_state
    
    def start_monitoring(self) -> None:
        """Start GPIO monitoring"""
        if self.monitoring:
            self.logger.warning("GPIO monitoring already active")
            return
        
        self.monitoring = True
        self.set_status_led(True)
        self.logger.info("GPIO monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop GPIO monitoring"""
        self.monitoring = False
        self.set_status_led(False)
        self.logger.info("GPIO monitoring stopped")
    
    def cleanup(self) -> None:
        """Clean up GPIO resources"""
        if HAS_GPIO:
            GPIO.cleanup()
        self.logger.info("GPIO cleanup completed")
    
    def test_inputs(self) -> Dict[str, Any]:
        """Test GPIO inputs and return current states"""
        if not HAS_GPIO:
            return {
                "mode": "simulation",
                "crank_pin": self.crank_pin,
                "hook_pin": self.hook_pin,
                "hook_state": "simulated",
                "status": "GPIO hardware not available"
            }
        
        return {
            "mode": "hardware",
            "crank_pin": self.crank_pin,
            "hook_pin": self.hook_pin,
            "hook_state": "off-hook" if self.get_hook_state() else "on-hook",
            "crank_state": GPIO.input(self.crank_pin),
            "status": "GPIO hardware active"
        }
    
    def simulate_crank_turn(self) -> None:
        """Simulate crank turn for testing without hardware"""
        self.logger.info("SIMULATION: Crank turn simulated")
        self._crank_callback(0)
    
    def simulate_hook_change(self, off_hook: bool) -> None:
        """Simulate hook state change for testing without hardware"""
        old_state = self.hook_state
        self.hook_state = off_hook
        event = GPIOEvent.HOOK_OFF if off_hook else GPIOEvent.HOOK_ON
        
        self.logger.info(f"SIMULATION: Hook state changed to {'off-hook' if off_hook else 'on-hook'}")
        
        if event in self.event_handlers and old_state != self.hook_state:
            try:
                self.event_handlers[event]()
            except Exception as e:
                self.logger.error(f"Error in simulated hook event handler: {e}")


def main():
    """Test GPIO monitor functionality"""
    logging.basicConfig(level=logging.INFO)
    
    gpio_monitor = GPIOMonitor()
    
    def on_crank():
        print("Crank turned!")
    
    def on_hook_off():
        print("Phone picked up!")
    
    def on_hook_on():
        print("Phone hung up!")
    
    gpio_monitor.register_event_handler(GPIOEvent.CRANK_TURN, on_crank)
    gpio_monitor.register_event_handler(GPIOEvent.HOOK_OFF, on_hook_off)
    gpio_monitor.register_event_handler(GPIOEvent.HOOK_ON, on_hook_on)
    
    gpio_monitor.start_monitoring()
    
    try:
        print("GPIO monitor test running. Press Ctrl+C to exit.")
        print("Test status:", gpio_monitor.test_inputs())
        
        if not HAS_GPIO:
            print("Running in simulation mode. Testing simulated events...")
            time.sleep(1)
            gpio_monitor.simulate_crank_turn()
            time.sleep(1)
            gpio_monitor.simulate_hook_change(True)
            time.sleep(1)
            gpio_monitor.simulate_hook_change(False)
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down GPIO monitor...")
    finally:
        gpio_monitor.stop_monitoring()
        gpio_monitor.cleanup()


if __name__ == "__main__":
    main()