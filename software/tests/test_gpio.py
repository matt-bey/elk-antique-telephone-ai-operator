"""
GPIO Monitor Tests - Antique Telephone AI Operator

Tests for GPIO monitoring functionality including crank detection,
hook switch monitoring, and ringer control.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gpio_monitor import GPIOMonitor, GPIOEvent
from utils.config_manager import ConfigManager


class TestGPIOMonitor:
    """Test cases for GPIO monitor functionality"""
    
    @pytest.fixture
    def config_manager(self):
        """Create test configuration manager"""
        config = ConfigManager()
        config.set('gpio.crank_pin', 18)
        config.set('gpio.hook_pin', 19)
        config.set('gpio.status_led_pin', 20)
        config.set('gpio.ringer_pin', 21)
        config.set('gpio.debounce_time', 50)
        return config
    
    @pytest.fixture
    def gpio_monitor(self, config_manager):
        """Create GPIO monitor instance for testing"""
        return GPIOMonitor(config_manager)
    
    def test_gpio_monitor_initialization(self, gpio_monitor):
        """Test GPIO monitor initializes correctly"""
        assert gpio_monitor.crank_pin == 18
        assert gpio_monitor.hook_pin == 19
        assert gpio_monitor.status_led_pin == 20
        assert gpio_monitor.ringer_pin == 21
        assert gpio_monitor.debounce_time == 0.05  # 50ms in seconds
        assert not gpio_monitor.monitoring
        assert not gpio_monitor.hook_state
    
    def test_register_event_handler(self, gpio_monitor):
        """Test event handler registration"""
        handler_called = False
        
        def test_handler():
            nonlocal handler_called
            handler_called = True
        
        gpio_monitor.register_event_handler(GPIOEvent.CRANK_TURN, test_handler)
        
        # Simulate crank turn
        gpio_monitor.simulate_crank_turn()
        
        assert handler_called
    
    def test_hook_state_simulation(self, gpio_monitor):
        """Test hook state change simulation"""
        hook_events = []
        
        def on_hook_off():
            hook_events.append('off')
        
        def on_hook_on():
            hook_events.append('on')
        
        gpio_monitor.register_event_handler(GPIOEvent.HOOK_OFF, on_hook_off)
        gpio_monitor.register_event_handler(GPIOEvent.HOOK_ON, on_hook_on)
        
        # Test hook state changes
        gpio_monitor.simulate_hook_change(True)  # Pick up
        assert gpio_monitor.hook_state is True
        assert 'off' in hook_events
        
        gpio_monitor.simulate_hook_change(False)  # Hang up
        assert gpio_monitor.hook_state is False
        assert 'on' in hook_events
    
    def test_status_led_control(self, gpio_monitor):
        """Test status LED control functionality"""
        # These calls should not raise exceptions
        gpio_monitor.set_status_led(True)
        gpio_monitor.set_status_led(False)
        
        # In simulation mode, should log but not fail
        assert True  # Test passes if no exceptions raised
    
    def test_ringer_trigger(self, gpio_monitor):
        """Test ringer triggering functionality"""
        # Test ringer triggering
        gpio_monitor.trigger_ringer(0.1)  # Short pulse for testing
        
        # Wait briefly for threaded operation
        time.sleep(0.2)
        
        # In simulation mode, should log but not fail
        assert True  # Test passes if no exceptions raised
    
    def test_monitoring_lifecycle(self, gpio_monitor):
        """Test monitoring start/stop lifecycle"""
        assert not gpio_monitor.monitoring
        
        gpio_monitor.start_monitoring()
        assert gpio_monitor.monitoring
        
        gpio_monitor.stop_monitoring()
        assert not gpio_monitor.monitoring
    
    def test_cleanup(self, gpio_monitor):
        """Test GPIO cleanup"""
        gpio_monitor.start_monitoring()
        gpio_monitor.cleanup()
        
        # Should not raise exceptions
        assert True
    
    def test_input_testing(self, gpio_monitor):
        """Test GPIO input testing functionality"""
        test_result = gpio_monitor.test_inputs()
        
        assert 'mode' in test_result
        assert 'crank_pin' in test_result
        assert 'hook_pin' in test_result
        assert 'status' in test_result
        
        # In most test environments, should be simulation mode
        assert test_result['mode'] in ['simulation', 'hardware']
    
    def test_debounce_timing(self, gpio_monitor):
        """Test debounce timing works correctly"""
        handler_count = 0
        
        def count_handler():
            nonlocal handler_count
            handler_count += 1
        
        gpio_monitor.register_event_handler(GPIOEvent.CRANK_TURN, count_handler)
        
        # Rapid successive calls should be debounced
        gpio_monitor.simulate_crank_turn()
        gpio_monitor.simulate_crank_turn()
        gpio_monitor.simulate_crank_turn()
        
        # Should only register one call due to debouncing
        assert handler_count == 1
        
        # Wait for debounce period
        time.sleep(0.1)
        
        # Now another call should register
        gpio_monitor.simulate_crank_turn()
        assert handler_count == 2
    
    def test_multiple_event_handlers(self, gpio_monitor):
        """Test multiple event types work correctly"""
        events_received = []
        
        def crank_handler():
            events_received.append('crank')
        
        def hook_off_handler():
            events_received.append('hook_off')
        
        def hook_on_handler():
            events_received.append('hook_on')
        
        gpio_monitor.register_event_handler(GPIOEvent.CRANK_TURN, crank_handler)
        gpio_monitor.register_event_handler(GPIOEvent.HOOK_OFF, hook_off_handler)
        gpio_monitor.register_event_handler(GPIOEvent.HOOK_ON, hook_on_handler)
        
        # Trigger all events
        gpio_monitor.simulate_crank_turn()
        gpio_monitor.simulate_hook_change(True)
        gpio_monitor.simulate_hook_change(False)
        
        assert 'crank' in events_received
        assert 'hook_off' in events_received
        assert 'hook_on' in events_received
    
    def test_configuration_override(self):
        """Test GPIO monitor with custom configuration"""
        config = ConfigManager()
        config.set('gpio.crank_pin', 22)
        config.set('gpio.hook_pin', 23)
        config.set('gpio.debounce_time', 100)
        
        monitor = GPIOMonitor(config)
        
        assert monitor.crank_pin == 22
        assert monitor.hook_pin == 23
        assert monitor.debounce_time == 0.1  # 100ms in seconds
        
        monitor.cleanup()
    
    @pytest.mark.asyncio
    async def test_async_event_handling(self, gpio_monitor):
        """Test GPIO events work with async handlers"""
        event_received = False
        
        async def async_handler():
            nonlocal event_received
            await asyncio.sleep(0.01)  # Small async operation
            event_received = True
        
        # Wrap async handler for sync callback
        def sync_wrapper():
            asyncio.create_task(async_handler())
        
        gpio_monitor.register_event_handler(GPIOEvent.CRANK_TURN, sync_wrapper)
        gpio_monitor.simulate_crank_turn()
        
        # Wait for async operation
        await asyncio.sleep(0.1)
        
        assert event_received


@pytest.mark.hardware
class TestGPIOHardware:
    """Hardware-specific GPIO tests (run only on Raspberry Pi)"""
    
    @pytest.fixture
    def gpio_monitor(self):
        """Create GPIO monitor for hardware testing"""
        config = ConfigManager()
        return GPIOMonitor(config)
    
    def test_hardware_detection(self, gpio_monitor):
        """Test if hardware is properly detected"""
        test_result = gpio_monitor.test_inputs()
        
        # On Raspberry Pi, should detect hardware mode
        try:
            is_pi = 'Raspberry Pi' in open('/proc/cpuinfo', 'r').read()
        except FileNotFoundError:
            is_pi = False
        if is_pi:
            assert test_result['mode'] == 'hardware'
        else:
            assert test_result['mode'] == 'simulation'
        
        gpio_monitor.cleanup()
    
    @pytest.mark.skipif(
        not (lambda: 'Raspberry Pi' in open('/proc/cpuinfo').read() if __import__('os').path.exists('/proc/cpuinfo') else False)(),
        reason="Requires Raspberry Pi hardware"
    )
    def test_real_gpio_setup(self, gpio_monitor):
        """Test real GPIO setup on Raspberry Pi"""
        # This test only runs on actual Raspberry Pi hardware
        gpio_monitor.start_monitoring()
        
        # Test that GPIO pins are configured
        test_result = gpio_monitor.test_inputs()
        assert test_result['mode'] == 'hardware'
        assert 'crank_state' in test_result
        
        gpio_monitor.stop_monitoring()
        gpio_monitor.cleanup()


def test_main_function():
    """Test the main function runs without errors"""
    from core.gpio_monitor import main

    # main() runs a while True loop; interrupt it after the simulation steps complete
    with patch('core.gpio_monitor.HAS_GPIO', False), \
         patch('core.gpio_monitor.time.sleep', side_effect=[None, None, None, KeyboardInterrupt]):
        try:
            main()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])