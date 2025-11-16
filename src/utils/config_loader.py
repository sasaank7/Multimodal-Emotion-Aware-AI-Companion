"""
Configuration loader for the Emotion-Aware AI Companion.
Loads and manages application configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage application configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the config YAML file. If None, uses default.
        """
        # Load environment variables
        load_dotenv()

        # Determine config path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Example: Override LLM model from environment
        if os.getenv("LLM_MODEL_NAME"):
            self.config["llm"]["model_name"] = os.getenv("LLM_MODEL_NAME")

        if os.getenv("LLM_DEVICE"):
            self.config["llm"]["device"] = os.getenv("LLM_DEVICE")

        if os.getenv("DEBUG"):
            self.config["app"]["debug"] = os.getenv("DEBUG").lower() == "true"

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'llm.model_name')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name (e.g., 'llm', 'emotion_detection')

        Returns:
            Dictionary containing the section configuration
        """
        return self.config.get(section, {})

    def update(self, key: str, value: Any):
        """
        Update a configuration value.

        Args:
            key: Configuration key in dot notation
            value: New value
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration. If None, overwrites original.
        """
        if output_path is None:
            project_root = Path(__file__).parent.parent.parent
            output_path = project_root / "config" / "config.yaml"

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigLoader(sections={list(self.config.keys())})"


# Global configuration instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get the global configuration instance (singleton pattern).

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)

    return _config_instance


def reload_config(config_path: Optional[str] = None):
    """
    Reload the configuration from file.

    Args:
        config_path: Path to config file
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path)
