"""Convert observed clinical events into working memory and vital trends."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VITAL_CODE = "VITALS"
OBSERVATION_REQUIRED_VITAL_SOURCES = {
    "heart_rate_bpm": {"Heart Rate, bpm", "Heart Rate"},
    "resp_rate_per_min": {"Respiratory Rate, insp/min", "Respiratory Rate"},
    "spo2_percent": {"O2 saturation pulseoxymetry, %", "O2 saturation"},
    "sbp_mmhg": {
        "Non Invasive Blood Pressure systolic, mmHg",
        "Arterial Blood Pressure systolic, mmHg",
        "ART BP Systolic, mmHg",
        "Blood Pressure systolic",
    },
    "dbp_mmhg": {
        "Non Invasive Blood Pressure diastolic, mmHg",
        "Arterial Blood Pressure diastolic, mmHg",
        "ART BP Diastolic, mmHg",
        "Blood Pressure diastolic",
    },
    "map_mmhg": {
        "Non Invasive Blood Pressure mean, mmHg",
        "Arterial Blood Pressure mean, mmHg",
        "ART BP Mean, mmHg",
    },
    "temperature_c": {
        "Temperature Fahrenheit, °F",
        "Temperature Fahrenheit",
        "Temperature Celsius, °C",
    },
}

logger = logging.getLogger(__name__)


from agents.validation import (
    _safe_text,
)
from agents.memory import (
    WorkingWindow,
)
from agents.validation import (
    _to_optional_float,
)

class PerceptionAgent:
    def __init__(self, max_working_windows: int):
        self.max_working_windows = max_working_windows

    def update_working_memory(
        self,
        existing_windows: List[WorkingWindow],
        window_data: Dict[str, Any],
        window_index: int,
    ) -> List[WorkingWindow]:
        hours = float(window_data.get("hours_since_admission", 0.0))
        new_window = WorkingWindow(
            window_id=window_index,
            start_hour=hours,
            end_hour=hours + float(window_data["current_window_hours"]),
            events=deepcopy(window_data.get("current_events", [])),
        )
        updated = list(existing_windows)
        updated.append(new_window)
        if len(updated) > self.max_working_windows:
            updated = updated[-self.max_working_windows :]
        return updated


class ObservationAgent:
    def __init__(self, observation_config_path: str):
        config_path = _safe_text(observation_config_path)
        if not config_path:
            raise ValueError("observation_config_path must be a non-empty string")
        self.observation_config_path = config_path
        self._unmapped_vital_sources_logged: set[str] = set()
        self._out_of_range_vital_sources_logged: set[str] = set()
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Observation config file not found: {path}")
        with open(path, "r") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Observation config must be a JSON object: {path}")
        self._validate_config(payload)
        self._compile_config(payload)

    @staticmethod
    def _required_mapping(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Missing or invalid config mapping: {key}")
        return value

    @staticmethod
    def _required_list(payload: Dict[str, Any], key: str) -> List[Any]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Missing or invalid config list: {key}")
        return value

    @staticmethod
    def _required_number(payload: Dict[str, Any], key: str) -> float:
        value = _to_optional_float(payload.get(key))
        if value is None:
            raise ValueError(f"Missing or invalid numeric config: {key}")
        return value

    @staticmethod
    def _required_string(payload: Dict[str, Any], key: str) -> str:
        value = _safe_text(payload.get(key))
        if not value:
            raise ValueError(f"Missing or invalid string config: {key}")
        return value

    def _validate_config(self, payload: Dict[str, Any]) -> None:
        window_minutes = self._required_number(payload, "window_minutes")
        if window_minutes <= 0:
            raise ValueError("window_minutes must be > 0")

        vital_sanity_ranges = self._required_mapping(payload, "vital_sanity_ranges")
        trend_vitals = self._required_list(payload, "trend_vitals")

        if not vital_sanity_ranges:
            raise ValueError("vital_sanity_ranges must not be empty")
        if not trend_vitals:
            raise ValueError("trend_vitals must not be empty")

        for key, value in vital_sanity_ranges.items():
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Invalid vital_sanity_ranges[{key}]")
            low = _to_optional_float(value[0])
            high = _to_optional_float(value[1])
            if low is None or high is None or low >= high:
                raise ValueError(f"Invalid vital_sanity_ranges[{key}] values")

        for canonical_name in OBSERVATION_REQUIRED_VITAL_SOURCES:
            if canonical_name not in vital_sanity_ranges:
                raise ValueError(f"Missing vital_sanity_ranges for required vital: {canonical_name}")
        for key in vital_sanity_ranges:
            if "temp" in key.lower() and key != "temperature_c":
                raise ValueError(f"Temperature sanity range must use temperature_c only, found {key}")

        trend_sources_by_canonical: Dict[str, set[str]] = {}
        mapped_sources: Dict[str, str] = {}
        for idx, item in enumerate(trend_vitals):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid trend_vitals[{idx}]")
            canonical_name = self._required_string(item, "canonical_name")
            sources = self._required_list(item, "sources")
            if canonical_name not in vital_sanity_ranges:
                raise ValueError(f"Missing sanity range for trend vital: {canonical_name}")
            if not sources:
                raise ValueError(f"trend_vitals[{idx}].sources must not be empty")

            trend_sources_by_canonical.setdefault(canonical_name, set())
            for source in sources:
                source_name = _safe_text(source)
                if not source_name:
                    raise ValueError(f"Invalid empty source for trend vital {canonical_name}")
                existing_mapping = mapped_sources.get(source_name)
                if existing_mapping is not None and existing_mapping != canonical_name:
                    raise ValueError(
                        f"Source {source_name} is mapped to multiple vitals: {existing_mapping}, {canonical_name}"
                    )
                mapped_sources[source_name] = canonical_name
                trend_sources_by_canonical[canonical_name].add(source_name)

        for canonical_name, required_sources in OBSERVATION_REQUIRED_VITAL_SOURCES.items():
            actual_sources = trend_sources_by_canonical.get(canonical_name, set())
            missing_sources = required_sources - actual_sources
            if missing_sources:
                missing_sources_text = ", ".join(sorted(missing_sources))
                raise ValueError(f"trend_vitals[{canonical_name}] is missing required sources: {missing_sources_text}")

    def _compile_config(self, payload: Dict[str, Any]) -> None:
        self.vital_sanity_ranges = {
            key: (float(value[0]), float(value[1])) for key, value in payload["vital_sanity_ranges"].items()
        }
        trend_vital_names: List[str] = []
        self.trend_vital_by_source: Dict[str, str] = {}

        for item in payload["trend_vitals"]:
            canonical_name = self._required_string(item, "canonical_name")
            trend_vital_names.append(canonical_name)
            for source in self._required_list(item, "sources"):
                source_name = _safe_text(source)
                if not source_name:
                    raise ValueError(f"Invalid empty source for trend vital {canonical_name}")
                self.trend_vital_by_source[source_name] = canonical_name

        self.trend_vital_names = list(dict.fromkeys(trend_vital_names))

    def _warn_once(self, *, seen: set[str], key: str, message: str, values: Tuple[Any, ...]) -> None:
        if key in seen:
            return
        seen.add(key)
        logger.warning(message, *values)

    def _normalize_ingested_vital_value(self, *, trend_name: str, source_name: str, raw_value: float) -> float:
        if trend_name != "temperature_c":
            return raw_value
        if "fahrenheit" in source_name.lower():
            return (raw_value - 32.0) * 5.0 / 9.0
        return raw_value

    def _ingest_vital_event(self, *, code_specifics: str, numeric_value: float) -> Optional[Tuple[str, float]]:
        trend_name = self.trend_vital_by_source.get(code_specifics)
        if trend_name is None:


            return None

        normalized_value = self._normalize_ingested_vital_value(
            trend_name=trend_name,
            source_name=code_specifics,
            raw_value=numeric_value,
        )
        sanity_range = self.vital_sanity_ranges.get(trend_name)
        if sanity_range is None:
            raise ValueError(f"Missing sanity range for vital {trend_name}")

        low, high = sanity_range
        if normalized_value < low or normalized_value > high:
            self._warn_once(
                seen=self._out_of_range_vital_sources_logged,
                key=f"{trend_name}:{code_specifics}",
                message="Out-of-range vital dropped: source=%s canonical=%s value=%g valid_range=[%g, %g]",
                values=(code_specifics, trend_name, normalized_value, low, high),
            )
            return None

        return trend_name, normalized_value

    def _compute_trends(self, trend_values: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        for trend_name in self.trend_vital_names:
            values = trend_values.get(trend_name, [])
            if not values:
                output[trend_name] = {"mean": None, "min": None, "max": None, "count": 0}
                continue
            output[trend_name] = {
                "mean": float(sum(values) / len(values)),
                "min": float(min(values)),
                "max": float(max(values)),
                "count": int(len(values)),
            }
        return output

    def analyze(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        current_events = window_data.get("current_events")
        if not isinstance(current_events, list):
            raise ValueError("Window payload must include current_events as a list")

        trend_values: Dict[str, List[float]] = {}
        for raw_event in current_events:
            if not isinstance(raw_event, dict):
                continue
            if _safe_text(raw_event.get("code")) != VITAL_CODE:
                continue

            numeric_value = _to_optional_float(raw_event.get("numeric_value"))
            if numeric_value is None:
                continue

            ingested_vital = self._ingest_vital_event(
                code_specifics=_safe_text(raw_event.get("code_specifics")),
                numeric_value=numeric_value,
            )
            if ingested_vital is None:
                continue

            trend_name, trend_value = ingested_vital
            trend_values.setdefault(trend_name, []).append(trend_value)

        return {
            "subject_id": window_data.get("subject_id"),
            "icu_stay_id": window_data.get("icu_stay_id"),
            "window_index": window_data.get("window_index"),
            "window_start": window_data.get("current_window_start"),
            "window_end": window_data.get("current_window_end"),
            "raw_event_count": len(current_events),
            "vital_trends": self._compute_trends(trend_values),
        }
