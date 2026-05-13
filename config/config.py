"""Minimal configuration loader for the ICU-Evo open-source release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class Config:
    """Load and validate the small config surface used by ICU-Evo."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)

        self.config_path = config_path

    def get(self, key_path: str, default: Any = None) -> Any:
        value: Any = self._config
        for key in key_path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def require(self, key_path: str) -> Any:
        value = self.get(key_path)
        if value is None:
            raise ValueError(f"Missing required config value: {key_path}")
        return value

    @property
    def events_path(self) -> str:
        return str(self.require("data.events_path"))

    @property
    def icu_stay_path(self) -> str:
        return str(self.require("data.icu_stay_path"))

    @property
    def agent_observation_hours(self) -> Optional[float]:
        value = self.get("data_loading.agent_observation_hours")
        if value is None:
            return None
        return float(value)

    @property
    def agent_current_window_hours(self) -> float:
        return float(self.require("agent_time_windows.current_window_hours"))

    @property
    def agent_include_pre_icu_data(self) -> bool:
        return bool(self.require("agent_time_windows.include_pre_icu_data"))

    @property
    def agent_use_discharge_summary_for_history(self) -> bool:
        return bool(self.require("agent_time_windows.use_discharge_summary_for_history"))

    @property
    def agent_num_discharge_summaries(self) -> int:
        return int(self.require("agent_time_windows.num_discharge_summaries"))

    @property
    def agent_pre_icu_history_hours(self) -> float:
        return float(self.require("agent_time_windows.pre_icu_history_hours"))

    @property
    def llm_provider(self) -> str:
        return str(self.require("llm.provider")).strip()

    @property
    def llm_model(self) -> str:
        return str(self.require("llm.model")).strip()

    @property
    def llm_temperature(self) -> float:
        return float(self.require("llm.temperature"))

    @property
    def llm_max_tokens(self) -> int:
        return int(self.require("llm.max_tokens"))

    @property
    def llm_max_input_tokens(self) -> int:
        return int(self.require("llm.max_input_tokens"))

    @property
    def icu_evo_max_working_windows(self) -> int:
        return int(self.require("icu_evo.max_working_windows"))

    @property
    def icu_evo_max_insights(self) -> int:
        return int(self.require("icu_evo.max_insights"))

    @property
    def icu_evo_max_trajectory_entries(self) -> int:
        return int(self.require("icu_evo.max_trajectory_entries"))

    @property
    def icu_evo_episode_block_windows(self) -> int:
        return int(self.require("icu_evo.episode_block_windows"))

    @property
    def icu_evo_insight_block_windows(self) -> int:
        return int(self.require("icu_evo.insight_block_windows"))

    @property
    def icu_evo_observation_config_path(self) -> str:
        value = str(self.require("icu_evo.observation_config_path")).strip()
        if not value:
            raise ValueError("icu_evo.observation_config_path must be non-empty")
        return value


_CONFIG: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    global _CONFIG
    _CONFIG = Config(config_path=config_path)
    return _CONFIG


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def get(key_path: str, default: Any = None) -> Any:
    return get_config().get(key_path, default)
