import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

class Config:
    """Configuration manager for Document-QA system."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    _config_path: Optional[Path] = None
    _last_mtime: float = 0.0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _expand_env_vars(self, data: Any) -> Any:
        import re
        if isinstance(data, str):
            match = re.match(r'^\$\{([^}]+)\}$', data)
            if match:
                var_name = match.group(1)
                value = os.getenv(var_name)
                if value is None:
                    import logging as _logging
                    _logging.getLogger(__name__).debug(f"Env var '{var_name}' not set — provider will be skipped.")
                    return ""
                return value
            else:
                return data
        elif isinstance(data, dict):
            return {key: self._expand_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        else:
            return data

    def _load_config(self):
        load_dotenv()
        config_path = Path(__file__).parent / "config.yaml"
        self._config_path = config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.yaml at {config_path}")
        else:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
            self._config = self._expand_env_vars(loaded_config)
            self._last_mtime = config_path.stat().st_mtime

    def reload(self) -> None:
        if self._config_path and self._config_path.exists():
            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
            self._last_mtime = self._config_path.stat().st_mtime

    def get(self, key_path: str, default: Any = None) -> Any:
        self._check_reload()
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        self._check_reload()
        return self._config.get(section, {})

    def _check_reload(self) -> None:
        if self._config_path and self._config_path.exists():
            try:
                mtime = self._config_path.stat().st_mtime
                if mtime != self._last_mtime:
                    self.reload()
            except OSError:
                pass

config = Config()
