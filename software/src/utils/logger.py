"""
Logger Configuration - Antique Telephone AI Operator

Provides centralized logging configuration with support for file and console output,
structured logging, and different log levels for development and production.
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional
from pathlib import Path
from datetime import datetime


class AntiquePhoneLogger:
    """
    Centralized logging configuration for antique telephone system
    
    Provides structured logging with appropriate formatting, file rotation,
    and console output suitable for both development and production use.
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 console_output: bool = True,
                 file_max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 file_backup_count: int = 5):
        """Initialize logging configuration"""
        
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file
        self.console_output = console_output
        self.file_max_bytes = file_max_bytes
        self.file_backup_count = file_backup_count
        
        # Set up root logger
        self._configure_root_logger()
        
        # Create application logger
        self.logger = logging.getLogger('antique_telephone')
        self.logger.info("Logging system initialized")
    
    def _configure_root_logger(self) -> None:
        """Configure the root logger with handlers and formatters"""
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set root logger level
        root_logger.setLevel(self.log_level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_file:
            try:
                # Create log directory if it doesn't exist
                log_path = Path(self.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.file_max_bytes,
                    backupCount=self.file_backup_count
                )
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(file_handler)
                
            except Exception as e:
                root_logger.error(f"Failed to create file logger: {e}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific module"""
        return logging.getLogger(f'antique_telephone.{name}')
    
    def set_level(self, level: str) -> None:
        """Change logging level for all handlers"""
        new_level = getattr(logging, level.upper(), logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(new_level)
        
        for handler in root_logger.handlers:
            handler.setLevel(new_level)
        
        self.log_level = new_level
        self.logger.info(f"Log level changed to {level.upper()}")
    
    def add_debug_handler(self, log_file: str) -> None:
        """Add additional debug-level file handler"""
        try:
            debug_handler = logging.FileHandler(log_file)
            debug_handler.setLevel(logging.DEBUG)
            
            debug_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s:%(lineno)-4d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S.%f'
            )
            debug_handler.setFormatter(debug_formatter)
            
            root_logger = logging.getLogger()
            root_logger.addHandler(debug_handler)
            
            self.logger.info(f"Debug handler added: {log_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to add debug handler: {e}")
    
    def log_system_info(self) -> None:
        """Log system information for debugging"""
        import platform
        import psutil
        
        self.logger.info("=== System Information ===")
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Python: {platform.python_version()}")
        self.logger.info(f"CPU: {platform.processor()}")
        self.logger.info(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
        self.logger.info(f"Disk: {psutil.disk_usage('/').total // (1024**3)} GB")
        self.logger.info("=== End System Information ===")
    
    def log_startup_banner(self) -> None:
        """Log application startup banner"""
        banner = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                     ANTIQUE TELEPHONE AI OPERATOR                           ║
║                                                                             ║
║  Converting early 1900s telephony to modern AI-powered communication        ║
║                                                                             ║
║  Phase 1: Modern Component Validation                                       ║
║  Target: Authentic 1920s operator experience with VoIP calling              ║
╚═════════════════════════════════════════════════════════════════════════════╝
        """.strip()
        
        for line in banner.split('\n'):
            self.logger.info(line)
    
    def create_session_log(self, session_name: str) -> str:
        """Create a new session-specific log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_file = f"/tmp/antique_telephone_session_{session_name}_{timestamp}.log"
        
        try:
            session_handler = logging.FileHandler(session_log_file)
            session_handler.setLevel(logging.DEBUG)
            
            session_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
                datefmt='%H:%M:%S.%f'
            )
            session_handler.setFormatter(session_formatter)
            
            # Add to specific logger
            session_logger = logging.getLogger(f'antique_telephone.session.{session_name}')
            session_logger.addHandler(session_handler)
            session_logger.setLevel(logging.DEBUG)
            
            self.logger.info(f"Session log created: {session_log_file}")
            return session_log_file
            
        except Exception as e:
            self.logger.error(f"Failed to create session log: {e}")
            return ""


def setup_logging(config_manager=None) -> AntiquePhoneLogger:
    """
    Set up logging for the antique telephone application
    
    Args:
        config_manager: Optional configuration manager for log settings
    
    Returns:
        Configured logger instance
    """
    
    # Default settings
    log_level = "INFO"
    log_file = "/var/log/antique-telephone.log"
    console_output = True
    
    # Override with config if available
    if config_manager:
        log_level = config_manager.get('system.log_level', log_level)
        log_file = config_manager.get('system.log_file', log_file)
        console_output = config_manager.get('system.console_output', console_output)
    
    # Check for environment overrides
    log_level = os.getenv('LOG_LEVEL', log_level)
    log_file = os.getenv('LOG_FILE', log_file)
    
    # Create logger instance
    logger_instance = AntiquePhoneLogger(
        log_level=log_level,
        log_file=log_file,
        console_output=console_output
    )
    
    return logger_instance


def get_component_logger(component_name: str) -> logging.Logger:
    """
    Get a logger for a specific component
    
    Args:
        component_name: Name of the component (e.g., 'gpio', 'audio', 'ai_operator')
    
    Returns:
        Logger instance for the component
    """
    return logging.getLogger(f'antique_telephone.{component_name}')


class LoggingMixin:
    """
    Mixin class to add logging capability to any class
    
    Usage:
        class MyClass(LoggingMixin):
            def __init__(self):
                super().__init__()
                self.setup_logging('my_component')
            
            def some_method(self):
                self.logger.info("Something happened")
    """
    
    def setup_logging(self, component_name: str) -> None:
        """Set up logging for this component"""
        self.logger = get_component_logger(component_name)
    
    def log_method_entry(self, method_name: str, **kwargs) -> None:
        """Log method entry with parameters"""
        if hasattr(self, 'logger'):
            params = ', '.join(f'{k}={v}' for k, v in kwargs.items())
            self.logger.debug(f"Entering {method_name}({params})")
    
    def log_method_exit(self, method_name: str, result=None) -> None:
        """Log method exit with result"""
        if hasattr(self, 'logger'):
            if result is not None:
                self.logger.debug(f"Exiting {method_name} -> {result}")
            else:
                self.logger.debug(f"Exiting {method_name}")


def main():
    """Test logging configuration"""
    
    # Test basic logging setup
    logger_instance = setup_logging()
    logger_instance.log_startup_banner()
    logger_instance.log_system_info()
    
    # Test component loggers
    gpio_logger = get_component_logger('gpio')
    audio_logger = get_component_logger('audio')
    ai_logger = get_component_logger('ai_operator')
    
    # Test different log levels
    gpio_logger.debug("GPIO debug message")
    audio_logger.info("Audio info message")
    ai_logger.warning("AI warning message")
    ai_logger.error("AI error message")
    
    # Test session logging
    session_file = logger_instance.create_session_log("test_session")
    session_logger = logging.getLogger('antique_telephone.session.test_session')
    session_logger.info("This is a session-specific log message")
    
    # Test logging mixin
    class TestComponent(LoggingMixin):
        def __init__(self):
            super().__init__()
            self.setup_logging('test_component')
        
        def test_method(self, param1, param2="default"):
            self.log_method_entry('test_method', param1=param1, param2=param2)
            self.logger.info("Doing some work...")
            result = "success"
            self.log_method_exit('test_method', result)
            return result
    
    test_comp = TestComponent()
    test_comp.test_method("value1", param2="value2")
    
    print(f"Session log file: {session_file}")
    print("Logging test completed")


if __name__ == "__main__":
    main()