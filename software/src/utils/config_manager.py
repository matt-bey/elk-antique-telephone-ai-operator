"""
Configuration Manager - Antique Telephone AI Operator

Handles loading and managing configuration from files and environment variables.
Supports hierarchical configuration with defaults, files, and environment overrides.
"""

import os
import logging
import configparser
from typing import Any, Dict, Optional, Union
from pathlib import Path


class ConfigManager:
    """
    Configuration management for antique telephone system
    
    Loads configuration from multiple sources in priority order:
    1. Environment variables (highest priority)
    2. Configuration files
    3. Default values (lowest priority)
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager"""
        self.logger = logging.getLogger(__name__)
        
        # Determine configuration directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to config/ directory at project root (one level above software/)
            project_root = Path(__file__).parent.parent.parent.parent
            self.config_dir = project_root / "config"
        
        # Configuration storage
        self.config: Dict[str, Any] = {}
        
        # Default configuration
        self._load_defaults()
        
        # Load configuration files
        self._load_config_files()
        
        # Load environment variables
        self._load_environment()
        
        self.logger.info(f"Configuration loaded from {self.config_dir}")
    
    def _load_defaults(self) -> None:
        """Load default configuration values"""
        self.config.update({
            # Audio configuration
            'audio.sample_rate': 44100,
            'audio.channels': 1,
            'audio.chunk_size': 1024,
            'audio.input_device': 'default',
            'audio.output_device': 'default',
            'audio.volume': 0.8,
            'audio.input_gain': 1.0,
            'audio.noise_gate': 100,
            'audio.enable_noise_gate': True,
            # Silence-detection tuning for end-of-turn detection
            'audio.silence_threshold': 500,
            'audio.silence_duration': 3.0,
            'audio.max_listen_duration': 20.0,
            
            # GPIO configuration
            'gpio.crank_pin': 18,
            'gpio.hook_pin': 19,
            'gpio.status_led_pin': 20,
            'gpio.ringer_pin': 21,
            'gpio.debounce_time': 50,
            
            # Anthropic configuration
            'anthropic.api_key': '',

            # Conversation provider selection ('anthropic' or 'ollama')
            'conversation.provider': 'anthropic',
            'conversation.model': 'claude-haiku-4-5-20251001',
            'conversation.host': 'http://localhost:11434',  # Ollama only

            # TTS provider selection
            'tts.voice': 'en_US-lessac-high',

            # Whisper configuration
            'whisper.model': 'base',
            'whisper.language': 'en',
            
            # Lookup / directory configuration
            'lookup.google_api_key': '',
            'lookup.gcp_project_id': '',
            'lookup.home_lat': None,
            'lookup.home_lon': None,
            'lookup.radius_meters': 50000,

            # SIP configuration
            'sip.username': '',
            'sip.password': '',
            'sip.domain': 'sip.callcentric.net',
            'sip.port': 5060,
            'sip.local_area_code': '',
            
            # System configuration
            'system.log_level': 'INFO',
            'system.log_file': '/var/log/antique-telephone.log',
            'system.startup_delay': 2.0,
        })
    
    def _load_config_files(self) -> None:
        """Load configuration from .conf files"""
        if not self.config_dir.exists():
            self.logger.warning(f"Configuration directory not found: {self.config_dir}")
            return
        
        config_files = [
            'audio.conf',
            'gpio.conf',
            'ai-service.conf',
            'sip.conf',
            'system.conf'
        ]
        
        for config_file in config_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                try:
                    self._load_ini_file(file_path)
                    self.logger.debug(f"Loaded configuration from {config_file}")
                except Exception as e:
                    self.logger.error(f"Failed to load {config_file}: {e}")
    
    def _load_ini_file(self, file_path: Path) -> None:
        """Load configuration from INI-style file"""
        parser = configparser.ConfigParser()
        parser.read(file_path)
        
        for section_name in parser.sections():
            for key, value in parser[section_name].items():
                config_key = f"{section_name}.{key}"
                
                # Type conversion
                converted_value = self._convert_value(value)
                self.config[config_key] = converted_value
    
    def _load_environment(self) -> None:
        """Load configuration from environment variables"""
        # Load .env from project root (two levels up from src/utils/)
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'
        if env_file.exists():
            self._load_env_file(env_file)
            self.logger.debug(f"Loaded .env from {env_file}")
        
        # Override with actual environment variables
        env_mappings = {
            'ANTHROPIC_API_KEY': 'anthropic.api_key',
            'CONVERSATION_PROVIDER': 'conversation.provider',
            'GOOGLE_PLACES_API_KEY': 'lookup.google_api_key',
            'GCP_PROJECT_ID': 'lookup.gcp_project_id',
            'HOME_LAT': 'lookup.home_lat',
            'HOME_LON': 'lookup.home_lon',
            'SIP_USERNAME': 'sip.username',
            'SIP_PASSWORD': 'sip.password',
            'SIP_DOMAIN': 'sip.domain',
            'SIP_LOCAL_AREA_CODE': 'sip.local_area_code',
            'LOG_LEVEL': 'system.log_level',
            'LOG_FILE': 'system.log_file',
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self.config[config_key] = self._convert_value(value)
                self.logger.debug(f"Loaded {config_key} from environment")
    
    def _load_env_file(self, env_file: Path) -> None:
        """Load environment variables from .env file"""
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
            
            self.logger.debug(f"Loaded environment variables from {env_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load .env file {env_file}: {e}")
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        if not isinstance(value, str):
            return value
        
        # Boolean conversion
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        elif value.lower() in ('false', 'no', 'off', '0'):
            return False
        
        # Numeric conversion
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
        self.logger.debug(f"Set {key} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all configuration values for a section"""
        section_config = {}
        prefix = f"{section}."
        
        for key, value in self.config.items():
            if key.startswith(prefix):
                section_key = key[len(prefix):]
                section_config[section_key] = value
        
        return section_config
    
    def has_key(self, key: str) -> bool:
        """Check if configuration key exists"""
        return key in self.config
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.config.copy()
    
    def save_to_file(self, file_path: str, section: Optional[str] = None) -> bool:
        """Save configuration to file"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            parser = configparser.ConfigParser()
            
            if section:
                # Save specific section
                section_config = self.get_section(section)
                if section_config:
                    parser.add_section(section)
                    for key, value in section_config.items():
                        parser.set(section, key, str(value))
            else:
                # Save all sections
                sections = {}
                for key, value in self.config.items():
                    if '.' in key:
                        section_name, section_key = key.split('.', 1)
                        if section_name not in sections:
                            sections[section_name] = {}
                        sections[section_name][section_key] = value
                
                for section_name, section_data in sections.items():
                    parser.add_section(section_name)
                    for key, value in section_data.items():
                        parser.set(section_name, key, str(value))
            
            with open(path, 'w') as f:
                parser.write(f)
            
            self.logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {file_path}: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_optional": []
        }
        
        # Optional API keys — warn if missing but don't block startup
        required_keys = [
            ('anthropic.api_key', 'Anthropic API key for conversation AI (ANTHROPIC_API_KEY)'),
        ]
        
        for key, description in required_keys:
            if not self.get(key):
                validation_result["warnings"].append(f"Missing {description} ({key})")
        
        # Optional SIP configuration (domain defaults to sip.callcentric.net)
        sip_required = ['sip.username', 'sip.password']
        sip_configured = any(self.get(key) for key in sip_required)

        if not sip_configured:
            validation_result["missing_optional"].append("SIP configuration (VoIP calling disabled)")
        elif not all(self.get(key) for key in sip_required):
            validation_result["errors"].append("Partial SIP configuration - username and password both required")
            validation_result["valid"] = False
        
        # Validate GPIO pin assignments
        gpio_pins = [
            self.get('gpio.crank_pin'),
            self.get('gpio.hook_pin'),
            self.get('gpio.status_led_pin'),
            self.get('gpio.ringer_pin')
        ]
        
        # Check for duplicate pin assignments
        pin_counts = {}
        for pin in gpio_pins:
            if pin is not None:
                pin_counts[pin] = pin_counts.get(pin, 0) + 1
        
        for pin, count in pin_counts.items():
            if count > 1:
                validation_result["errors"].append(f"GPIO pin {pin} assigned to multiple functions")
                validation_result["valid"] = False
        
        # Validate audio configuration
        sample_rate = self.get('audio.sample_rate')
        if sample_rate not in [22050, 44100, 48000]:
            validation_result["warnings"].append(f"Unusual sample rate: {sample_rate}")
        
        volume = self.get('audio.volume')
        if not (0.0 <= volume <= 1.0):
            validation_result["errors"].append(f"Volume must be between 0.0 and 1.0, got {volume}")
            validation_result["valid"] = False
        
        return validation_result
    
    def print_config(self, section: Optional[str] = None) -> None:
        """Print configuration values for debugging"""
        print("Configuration:")
        print("=" * 50)
        
        if section:
            section_config = self.get_section(section)
            print(f"[{section}]")
            for key, value in sorted(section_config.items()):
                # Mask sensitive values
                if 'password' in key.lower() or 'key' in key.lower():
                    display_value = '*' * len(str(value)) if value else '<not set>'
                else:
                    display_value = value
                print(f"  {key} = {display_value}")
        else:
            sections = {}
            for key, value in self.config.items():
                if '.' in key:
                    section_name, section_key = key.split('.', 1)
                    if section_name not in sections:
                        sections[section_name] = {}
                    sections[section_name][section_key] = value
            
            for section_name in sorted(sections.keys()):
                print(f"\n[{section_name}]")
                for key, value in sorted(sections[section_name].items()):
                    # Mask sensitive values
                    if 'password' in key.lower() or 'key' in key.lower():
                        display_value = '*' * len(str(value)) if value else '<not set>'
                    else:
                        display_value = value
                    print(f"  {key} = {display_value}")


def main():
    """Test configuration manager"""
    logging.basicConfig(level=logging.INFO)
    
    config = ConfigManager()
    
    print("Configuration Manager Test")
    print("=" * 40)
    
    # Print all configuration
    config.print_config()
    
    # Test validation
    print("\nConfiguration Validation:")
    validation = config.validate_config()
    print(f"Valid: {validation['valid']}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['missing_optional']:
        print("Missing Optional:")
        for missing in validation['missing_optional']:
            print(f"  - {missing}")
    
    # Test individual key access
    print(f"\nSample rate: {config.get('audio.sample_rate')}")
    print(f"Crank pin: {config.get('gpio.crank_pin')}")
    print(f"Anthropic key configured: {bool(config.get('anthropic.api_key'))}")


if __name__ == "__main__":
    main()