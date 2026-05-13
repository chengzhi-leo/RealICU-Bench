"""ICU-Evo: observation-grounded multi-agent pipeline with dynamic memory."""

from __future__ import annotations

import json
import logging
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from model.llms import LLMClient
from prompts.icu_evo_prompts import get_episode_agent_prompt, get_insight_agent_prompt
from prompts.oracle_prompt import format_pre_icu_compression_prompt
from prompts.predictor_prompts import get_survival_prediction_prompt
from utils.event_format import format_event_line as format_shared_event_line
from utils.event_format import format_event_lines as format_shared_event_lines
from utils.json_parse import parse_json_dict_or_raise

PRE_ICU_COMPRESSION_STEP_TYPE = "icu_evo_pre_icu_history_compressor"
OBSERVATION_STEP_TYPE = "icu_evo_observation_agent"
ICU_EVO_SCHEMA_MAX_ATTEMPTS = 4
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


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return 0
    return 0


def _parse_json_response(response: Any) -> Dict[str, Any]:
    return parse_json_dict_or_raise(response)


def _validate_episode_output_schema(payload: Any) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Top-level payload is not a JSON object."

    summary_payload = payload.get("episode_summary", payload.get("episode"))
    if not isinstance(summary_payload, dict):
        return False, "Missing episode_summary object."

    episode_text = _safe_text(summary_payload.get("text"))
    if not episode_text:
        episode_text = _safe_text(summary_payload.get("episode_summary"))
    if not episode_text:
        episode_text = _safe_text(payload.get("text"))
    if not episode_text:
        return False, "Missing episode summary text."

    supporting_event_ids = summary_payload.get(
        "supporting_event_ids",
        summary_payload.get("supporting_evidence", payload.get("supporting_event_ids", [])),
    )
    if not isinstance(supporting_event_ids, list):
        return False, "episode_summary.supporting_event_ids must be a list."

    critical_events = payload.get("critical_events")
    if not isinstance(critical_events, list):
        return False, "critical_events must be a list."

    for item in critical_events:
        if not isinstance(item, dict):
            return False, "Each critical_events item must be an object."
        reason = _safe_text(item.get("reason"))
        if not reason:
            return False, "Each critical_events item must include reason."
        supporting = item.get("supporting_event_ids", item.get("supporting_evidence", []))
        if supporting is not None and not isinstance(supporting, list):
            return False, "Each critical_events item must include list-valued supporting_event_ids."
        event_text = _safe_text(item.get("event"))
        if item.get("event_id") is None and not event_text and not supporting:
            return False, "Each critical_events item must include event_id, event, or supporting_event_ids."

    return True, ""


def _validate_insight_output_schema(payload: Any) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Top-level payload is not a JSON object."

    insight_updates = payload.get("insight_updates")
    updated_insights = payload.get("updated_insights")
    if insight_updates is None and updated_insights is None:
        return False, "Missing insight_updates or updated_insights list."

    updates = insight_updates if insight_updates is not None else updated_insights
    if not isinstance(updates, list):
        return False, "insight_updates/updated_insights must be a list."

    if insight_updates is not None and not isinstance(insight_updates, list):
        return False, "insight_updates must be a list."
    if updated_insights is not None and not isinstance(updated_insights, list):
        return False, "updated_insights must be a list."

    new_insights = payload.get("new_insights")
    if not isinstance(new_insights, list):
        return False, "new_insights must be a list."

    for item in updates:
        if not isinstance(item, dict):
            return False, "Each insight_updates item must be an object."
        raw_id = item.get("insight_id", item.get("hypothesis_id"))
        if raw_id is None:
            return False, "Each insight_updates item must include insight_id or hypothesis_id."

    for item in new_insights:
        if not isinstance(item, dict):
            return False, "Each new_insights item must be an object."
        if not _safe_text(item.get("hypothesis")):
            return False, "Each new_insights item must include hypothesis."

    return True, ""


def _validate_pre_icu_compression_output_schema(payload: Any) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Top-level payload is not a JSON object."
    compressed = _safe_text(payload.get("compressed_pre_icu_history"))
    if not compressed:
        return False, "compressed_pre_icu_history missing in LLM response."
    return True, ""


def _chat_with_schema_validation(
    *,
    llm_client: LLMClient,
    prompt: str,
    schema_validator: Callable[[Any], Tuple[bool, str]],
    max_attempts: int,
) -> Tuple[Dict[str, Any], str, Dict[str, Any], Optional[str], int]:
    attempts = max(1, int(max_attempts))
    total_input_tokens = 0
    total_output_tokens = 0
    last_raw = ""
    last_parsed: Dict[str, Any] = {}
    errors: List[str] = []

    for attempt_index in range(1, attempts + 1):
        response = llm_client.chat(prompt=prompt, response_format="text")
        usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
        total_input_tokens += _normalize_token_count(usage.get("input_tokens", 0))
        total_output_tokens += _normalize_token_count(usage.get("output_tokens", 0))

        raw = _safe_text(response.get("content"))
        parse_error: Optional[str] = None
        parsed: Dict[str, Any] = {}
        for candidate in (response.get("parsed"), raw):
            if candidate is None:
                continue
            try:
                parsed = _parse_json_response(candidate)
                parse_error = None
                break
            except Exception as exc:
                parse_error = str(exc)

        schema_ok = False
        schema_error: Optional[str] = None
        if parse_error is None:
            schema_ok, schema_error = schema_validator(parsed)
        else:
            schema_error = parse_error

        last_raw = raw
        last_parsed = parsed if isinstance(parsed, dict) else {}
        if schema_ok:
            return (
                last_parsed,
                last_raw,
                {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens},
                None,
                attempt_index,
            )

        errors.append(schema_error or "Schema validation failed.")

    error_text = " | ".join(errors)
    return (
        last_parsed,
        last_raw,
        {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens},
        f"Schema validation failed after {attempts} attempts: {error_text}",
        attempts,
    )


def _normalize_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    output: List[int] = []
    seen = set()
    for item in value:
        parsed: Optional[int] = None
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            parsed = item
        elif isinstance(item, float):
            if item.is_integer():
                parsed = int(item)
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            try:
                parsed = int(text)
            except ValueError:
                try:
                    parsed_float = float(text)
                    if parsed_float.is_integer():
                        parsed = int(parsed_float)
                except ValueError:
                    matches = re.findall(r"-?\d+", text)
                    if len(matches) == 1:
                        parsed = int(matches[0])
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        output.append(parsed)
    return output


def _normalize_text_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    seen = set()
    for item in value:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _split_evidence_references(value: Any) -> Tuple[List[int], List[str]]:
    if not isinstance(value, list):
        return [], []

    event_ids: List[int] = []
    trend_evidence: List[str] = []
    seen_events = set()
    seen_trends = set()

    for item in value:
        parsed: Optional[int] = None
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            parsed = item
        elif isinstance(item, float) and item.is_integer():
            parsed = int(item)
        elif isinstance(item, str):
            text = item.strip()
            match = re.fullmatch(r"\[?(-?\d+)\]?", text)
            if match:
                parsed = int(match.group(1))
            elif text:
                trend_text = text
                if trend_text not in seen_trends:
                    seen_trends.add(trend_text)
                    trend_evidence.append(trend_text)
        elif isinstance(item, dict):
            ids = _normalize_int_list([item.get("event_id")])
            if ids:
                parsed = ids[0]
            else:
                trend_text = _safe_text(item.get("evidence") or item.get("trend") or item.get("vital_trend"))
                if trend_text and trend_text not in seen_trends:
                    seen_trends.add(trend_text)
                    trend_evidence.append(trend_text)

        if parsed is None or parsed in seen_events:
            continue
        seen_events.add(parsed)
        event_ids.append(parsed)

    return event_ids, trend_evidence


def _format_event_lines(events: List[Dict[str, Any]], *, empty_text: str = "(No events)") -> List[str]:
    return format_shared_event_lines(events, empty_text=empty_text)


def _format_single_event_line(event: Dict[str, Any]) -> str:
    if not isinstance(event, dict):
        return ""
    return _safe_text(format_shared_event_line(event))


def _serialize_event_string(value: Any) -> str:
    if isinstance(value, str):
        return _safe_text(value)
    if not isinstance(value, dict):
        return _safe_text(value)

    raw_event = value.get("raw_event")
    if isinstance(raw_event, dict):
        line = _format_single_event_line(raw_event)
        if line:
            return line

    name_str = _safe_text(value.get("name_str"))
    if name_str:
        return name_str

    line = _format_single_event_line(value)
    if line:
        return line

    return _safe_text(value.get("event"))


def _serialize_event_strings(events: Any) -> List[str]:
    if not isinstance(events, list):
        return []

    lines: List[str] = []
    for event in events:
        line = _serialize_event_string(event)
        if line:
            lines.append(line)
    return lines


def _serialize_critical_event_strings(critical_events: Any) -> List[str]:
    if not isinstance(critical_events, list):
        return []

    lines: List[str] = []
    seen = set()

    for item in critical_events:
        if isinstance(item, str):
            line = _safe_text(item)
            if line and line not in seen:
                seen.add(line)
                lines.append(line)
            continue
        if not isinstance(item, dict):
            continue

        supporting_event_lines = _serialize_event_strings(item.get("supporting_events"))
        if supporting_event_lines:
            for line in supporting_event_lines:
                if line and line not in seen:
                    seen.add(line)
                    lines.append(line)
            continue

        supporting_ids = _normalize_int_list(item.get("supporting_event_ids", []))
        event_text = _safe_text(item.get("event"))
        if supporting_ids and event_text:
            fallback_line = f"[{supporting_ids[0]}] {event_text}"
        else:
            fallback_line = event_text
        if fallback_line and fallback_line not in seen:
            seen.add(fallback_line)
            lines.append(fallback_line)

    return lines


def _extract_event_reference_id(event: Dict[str, Any]) -> Optional[int]:
    if not isinstance(event, dict):
        return None
    for key in ("icu_event_index", "event_index", "event_id", "event_idx", "prompt_event_id", "relative_event_id"):
        raw_value = event.get(key)
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            continue
    return None


def _derive_raw_event_name(event: Dict[str, Any]) -> str:
    if not isinstance(event, dict):
        return "Unknown event"
    for key in ("event_name", "code_specifics", "code"):
        value = _safe_text(event.get(key))
        if value:
            return value
    return "Unknown event"


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fill_prompt_placeholder(template: str, key: str, value: str) -> str:
    return template.replace(f"{{{{{key}}}}}", value).replace(f"{{{key}}}", value)


_PROMPT_PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}|\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _extract_prompt_placeholders(template: str) -> List[str]:
    placeholders = set()
    for match in _PROMPT_PLACEHOLDER_PATTERN.finditer(template):
        key = match.group(1) or match.group(2)
        if key:
            placeholders.add(key)
    return sorted(placeholders)


def _render_prompt(template: str, values: Dict[str, str]) -> str:
    required_keys = _extract_prompt_placeholders(template)
    missing_keys = sorted(key for key in required_keys if key not in values)
    if missing_keys:
        raise ValueError(f"Missing prompt placeholder values: {', '.join(missing_keys)}")

    prompt = template
    for key, value in values.items():
        prompt = _fill_prompt_placeholder(prompt, key, value)

    unresolved_keys = sorted(key for key in required_keys if f"{{{key}}}" in prompt or f"{{{{{key}}}}}" in prompt)
    if unresolved_keys:
        raise ValueError(f"Unresolved prompt placeholders: {', '.join(unresolved_keys)}")
    return prompt


def _format_patient_metadata_lines(
    patient_metadata: Optional[Dict[str, Any]],
    *,
    include_ids: bool = False,
) -> List[str]:
    if not isinstance(patient_metadata, dict) or not patient_metadata:
        return ["- None"]

    lines: List[str] = []
    for key, value in patient_metadata.items():
        if not include_ids and key in {"subject_id", "icu_stay_id"}:
            continue

        value_text = _safe_text(value)
        if not value_text:
            continue

        if "\n" not in value_text:
            lines.append(f"- {key}: {value_text}")
            continue

        lines.append(f"- {key}:")
        for segment in value_text.splitlines():
            segment_text = _safe_text(segment)
            if segment_text:
                lines.append(f"  {segment_text}")

    return lines or ["- None"]


def _format_trend_value(value: Any) -> str:
    parsed = _to_optional_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.1f}"


def _window_label(window_index: int, start_hour: Optional[float], end_hour: Optional[float]) -> str:
    if start_hour is None or end_hour is None:
        return f"window {window_index}"
    return f"window {window_index} (hour {start_hour:.1f}-{end_hour:.1f})"


def _aggregate_trend_block(
    trend_block: List[TrendMemoryEntry],
) -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for entry in trend_block:
        for vital_name, stats in entry.vital_trends.items():
            if not isinstance(stats, dict):
                continue

            count = _normalize_token_count(stats.get("count"))
            if count <= 0:
                continue

            trend_rows = aggregated.setdefault(
                vital_name,
                {
                    "windows": [],
                    "overall_count": 0,
                    "overall_sum": 0.0,
                    "overall_min": None,
                    "overall_max": None,
                    "start_hour": entry.start_hour,
                    "end_hour": entry.end_hour,
                    "start_window": entry.window_index,
                    "end_window": entry.window_index,
                },
            )

            mean = _to_optional_float(stats.get("mean"))
            min_value = _to_optional_float(stats.get("min"))
            max_value = _to_optional_float(stats.get("max"))

            trend_rows["windows"].append(
                {
                    "window_index": entry.window_index,
                    "start_hour": entry.start_hour,
                    "end_hour": entry.end_hour,
                    "mean": mean,
                    "min": min_value,
                    "max": max_value,
                    "count": count,
                }
            )

            trend_rows["overall_count"] += count
            if mean is not None:
                trend_rows["overall_sum"] += mean * count
            if min_value is not None:
                current_min = trend_rows["overall_min"]
                trend_rows["overall_min"] = min_value if current_min is None else min(current_min, min_value)
            if max_value is not None:
                current_max = trend_rows["overall_max"]
                trend_rows["overall_max"] = max_value if current_max is None else max(current_max, max_value)

            trend_rows["start_hour"] = min(float(trend_rows["start_hour"]), float(entry.start_hour))
            trend_rows["end_hour"] = max(float(trend_rows["end_hour"]), float(entry.end_hour))
            trend_rows["start_window"] = min(int(trend_rows["start_window"]), int(entry.window_index))
            trend_rows["end_window"] = max(int(trend_rows["end_window"]), int(entry.window_index))
    return aggregated


def _format_window_trend_lines(vital_trends: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not isinstance(vital_trends, dict) or not vital_trends:
        return ["- None"]

    for vital_name in sorted(vital_trends.keys()):
        stats = vital_trends.get(vital_name, {})
        if not isinstance(stats, dict):
            continue
        lines.append(
            "- "
            f"{vital_name}: mean={_format_trend_value(stats.get('mean'))}, "
            f"min={_format_trend_value(stats.get('min'))}, "
            f"max={_format_trend_value(stats.get('max'))}, "
            f"count={_normalize_token_count(stats.get('count'))}"
        )
    return lines or ["- None"]


def _format_trend_block_lines(trend_block: List[TrendMemoryEntry]) -> List[str]:
    lines: List[str] = []
    lines.append("## Vital Trends")
    lines.append("Only windows with at least one corresponding vital value are shown. Empty windows are omitted.")
    aggregated_trends = _aggregate_trend_block(trend_block)
    if not aggregated_trends:
        lines.append("- None")
        return lines

    for vital_name in sorted(aggregated_trends.keys()):
        trend_group = aggregated_trends[vital_name]
        lines.append(f"### {vital_name}")
        for window_stats in trend_group["windows"]:
            lines.append(
                "- "
                f"{_window_label(window_stats['window_index'], window_stats['start_hour'], window_stats['end_hour'])}: "
                f"mean={_format_trend_value(window_stats.get('mean'))}, "
                f"min={_format_trend_value(window_stats.get('min'))}, "
                f"max={_format_trend_value(window_stats.get('max'))}, "
                f"count={_normalize_token_count(window_stats.get('count'))}"
            )

        overall_count = int(trend_group["overall_count"])
        overall_mean: Optional[float] = None
        if overall_count > 0:
            overall_mean = float(trend_group["overall_sum"]) / float(overall_count)
        lines.append(
            "- "
            f"overall windows {int(trend_group['start_window'])}-{int(trend_group['end_window'])} "
            f"(hour {float(trend_group['start_hour']):.1f}-{float(trend_group['end_hour']):.1f}): "
            f"mean={_format_trend_value(overall_mean)}, "
            f"min={_format_trend_value(trend_group.get('overall_min'))}, "
            f"max={_format_trend_value(trend_group.get('overall_max'))}, "
            f"count={overall_count}"
        )
        lines.append("")

    return lines


def _format_trend_entry_lines(entry: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    window_index = _normalize_token_count(entry.get("window_index"))
    start_hour = _to_optional_float(entry.get("start_hour"))
    end_hour = _to_optional_float(entry.get("end_hour"))
    if start_hour is None or end_hour is None:
        lines.append(f"### Window {window_index}")
    else:
        lines.append(f"### Window {window_index} (hour {start_hour:.1f}-{end_hour:.1f})")

    vital_trends = entry.get("vital_trends", {})
    lines.append("Vital trends:")
    lines.extend(_format_window_trend_lines(vital_trends))

    raw_event_count = _normalize_token_count(entry.get("raw_event_count"))
    lines.append(f"Window stats: raw_events={raw_event_count}")
    return lines


def _trend_entry_summary(entry: Dict[str, Any]) -> str:
    vital_trends = entry.get("vital_trends", {})
    tracked_trends: List[str] = []
    if isinstance(vital_trends, dict):
        for vital_name, stats in vital_trends.items():
            if not isinstance(stats, dict):
                continue
            if _normalize_token_count(stats.get("count")) > 0:
                tracked_trends.append(str(vital_name))
    if not tracked_trends:
        return "No measurable tracked vital trends in this window block."
    return f"tracked_trends={', '.join(tracked_trends[:5])}"


def _trend_entry_by_window(trend_block: List[TrendMemoryEntry]) -> Dict[int, TrendMemoryEntry]:
    return {entry.window_index: entry for entry in trend_block}


def _format_episode_input_lines(
    raw_window_block: List[WorkingWindow],
    trend_block: List[TrendMemoryEntry],
) -> List[str]:
    lines: List[str] = []

    for window in raw_window_block:
        lines.append(f"### Window {window.window_id} (hour {window.start_hour:.1f}-{window.end_hour:.1f})")
        lines.append("Raw events:")
        lines.extend(_format_event_lines(window.events, empty_text="- None"))
        lines.append("")

    lines.append("### Selected vital trends")
    aggregated_trends = _aggregate_trend_block(trend_block)
    if not aggregated_trends:
        lines.append("- None")
        return lines

    for vital_name in sorted(aggregated_trends.keys()):
        trend_group = aggregated_trends[vital_name]
        lines.append(f"### {vital_name}")
        for window_stats in trend_group["windows"]:
            lines.append(
                "- "
                f"{_window_label(window_stats['window_index'], window_stats['start_hour'], window_stats['end_hour'])}: "
                f"mean={_format_trend_value(window_stats.get('mean'))}, "
                f"min={_format_trend_value(window_stats.get('min'))}, "
                f"max={_format_trend_value(window_stats.get('max'))}, "
                f"count={_normalize_token_count(window_stats.get('count'))}"
            )

        overall_count = int(trend_group["overall_count"])
        overall_mean: Optional[float] = None
        if overall_count > 0:
            overall_mean = float(trend_group["overall_sum"]) / float(overall_count)
        lines.append(
            "- "
            f"overall windows {int(trend_group['start_window'])}-{int(trend_group['end_window'])} "
            f"(hour {float(trend_group['start_hour']):.1f}-{float(trend_group['end_hour']):.1f}): "
            f"mean={_format_trend_value(overall_mean)}, "
            f"min={_format_trend_value(trend_group.get('overall_min'))}, "
            f"max={_format_trend_value(trend_group.get('overall_max'))}, "
            f"count={overall_count}"
        )
        lines.append("")

    if not lines:
        return ["- None"]
    return lines


def _format_prior_episode_summary_text(trajectory_memory: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in trajectory_memory:
        if not isinstance(item, dict):
            continue
        item_type = _safe_text(item.get("type"))
        if item_type and item_type != "episode":
            continue

        summary = _safe_text(item.get("episode_summary", item.get("text")))
        if not summary:
            continue

        start_window = _normalize_token_count(item.get("start_window"))
        end_window = _normalize_token_count(item.get("end_window"))
        start_hour = _to_optional_float(item.get("start_hour"))
        end_hour = _to_optional_float(item.get("end_hour"))

        if start_hour is None or end_hour is None:
            lines.append(f"Window {start_window}-{end_window}: {summary}")
            continue

        lines.append(f"Window {start_window}-{end_window} " f"(hour {start_hour:.1f}-{end_hour:.1f}): {summary}")

    return "\n".join(lines) if lines else "None"


@dataclass
class LLMCallLog:
    timestamp: str
    patient_id: str
    window_index: int
    hours_since_admission: float
    prompt: str
    response: str
    parsed_response: Optional[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "window_index": self.window_index,
            "hours_since_admission": self.hours_since_admission,
            "prompt": self.prompt,
            "response": self.response,
            "parsed_response": self.parsed_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }


@dataclass
class WorkingWindow:
    window_id: int
    start_hour: float
    end_hour: float
    events: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "events": _serialize_event_strings(self.events),
        }


@dataclass
class TrendMemoryEntry:
    window_index: int
    start_hour: float
    end_hour: float
    raw_event_count: int
    vital_trends: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "raw_event_count": self.raw_event_count,
            "vital_trends": deepcopy(self.vital_trends),
        }


@dataclass
class Episode:
    episode_id: int
    start_window: int
    end_window: int
    start_hour: float
    end_hour: float
    episode_summary: str
    supporting_event_ids: List[int]
    supporting_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "episode",
            "episode_id": self.episode_id,
            "start_window": self.start_window,
            "end_window": self.end_window,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "episode_summary": self.episode_summary,
            "supporting_event_ids": self.supporting_event_ids,
            "supporting_events": _serialize_event_strings(self.supporting_events),
        }


@dataclass
class CriticalEventsMemoryEntry:
    episode_id: int
    start_window: int
    end_window: int
    start_hour: float
    end_hour: float
    critical_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "episode_critical_events",
            "episode_id": self.episode_id,
            "start_window": self.start_window,
            "end_window": self.end_window,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "critical_events": _serialize_critical_event_strings(self.critical_events),
        }


@dataclass
class EvidenceEvent:
    event_id: int
    name_str: str

    def to_dict(self) -> str:
        return self.name_str


@dataclass
class Insight:
    insight_id: int
    hypothesis: str
    supporting_event_ids: List[int]
    counter_event_ids: List[int]
    created_at: float
    updated_at: float
    supporting_trend_evidence: List[str] = field(default_factory=list)
    counter_trend_evidence: List[str] = field(default_factory=list)
    supporting_evidence: List[EvidenceEvent] = field(default_factory=list)
    counter_evidence: List[EvidenceEvent] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        supporting_evidence = [item.to_dict() for item in self.supporting_evidence] + list(
            self.supporting_trend_evidence
        )
        counter_evidence = [item.to_dict() for item in self.counter_evidence] + list(self.counter_trend_evidence)
        return {
            "insight_id": self.insight_id,
            "hypothesis": self.hypothesis,
            "supporting_event_ids": self.supporting_event_ids,
            "counter_event_ids": self.counter_event_ids,
            "supporting_trend_evidence": self.supporting_trend_evidence,
            "counter_trend_evidence": self.counter_trend_evidence,
            "supporting_evidence": supporting_evidence,
            "counter_evidence": counter_evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "score": self.score,
        }


@dataclass
class ICUEvoMemory:
    patient_metadata: Dict[str, Any] = field(default_factory=dict)
    working_memory: List[WorkingWindow] = field(default_factory=list)
    trend_memory: List[TrendMemoryEntry] = field(default_factory=list)
    critical_events_memory: List[CriticalEventsMemoryEntry] = field(default_factory=list)
    trajectory_memory: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    last_processed_window_index: int = -1
    last_processed_start_hour: float = 0.0
    last_processed_end_hour: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_metadata": deepcopy(self.patient_metadata),
            "working_memory": [item.to_dict() for item in self.working_memory],
            "trend_memory": [item.to_dict() for item in self.trend_memory],
            "critical_events_memory": [item.to_dict() for item in self.critical_events_memory],
            "trajectory_memory": deepcopy(self.trajectory_memory),
            "insights": [item.to_dict() for item in self.insights],
            "last_processed_window_index": self.last_processed_window_index,
            "last_processed_start_hour": self.last_processed_start_hour,
            "last_processed_end_hour": self.last_processed_end_hour,
        }

    def to_text(self) -> str:
        parts: List[str] = ["## Patient Metadata"]
        for key, value in self.patient_metadata.items():
            if key in {"subject_id", "icu_stay_id"}:
                continue
            parts.append(f"{key}: {value}")

        parts.append("")
        parts.append("## Trajectory of the ICU Stay")
        if self.trajectory_memory:
            for item in self.trajectory_memory:
                if not isinstance(item, dict):
                    parts.append(f"- {item}")
                    continue
                episode_line = (
                    f"- Window {item.get('start_window')}-{item.get('end_window')} "
                    f"(hour {_safe_text(item.get('start_hour'))}-{_safe_text(item.get('end_hour'))}): "
                    f"{item.get('episode_summary', '')}"
                )
                parts.append(episode_line)
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Trend Memory")
        if self.trend_memory:
            for entry in self.trend_memory:
                parts.extend(_format_trend_entry_lines(entry.to_dict()))
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Critical Events Memory")
        if self.critical_events_memory:
            for entry in self.critical_events_memory:
                parts.append(
                    f"Window {entry.start_window}-{entry.end_window} "
                    f"(hour {entry.start_hour:.1f}-{entry.end_hour:.1f})"
                )
                critical_event_lines = _serialize_critical_event_strings(entry.critical_events)
                if critical_event_lines:
                    parts.extend(critical_event_lines)
                else:
                    parts.append("- None")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Patient Specific Insights")
        if self.insights:
            for insight in self.insights:
                parts.append(f"- I{insight.insight_id} score={insight.score:.3f}: {insight.hypothesis}")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Current Window Observation")
        if self.working_memory:
            current_window = self.working_memory[-1]
            parts.append(
                f"Window {current_window.window_id} (Hour {current_window.start_hour:.1f}-{current_window.end_hour:.1f})"
            )
            parts.extend(_format_event_lines(current_window.events, empty_text="- (No events)"))
        else:
            parts.append("- None")

        return "\n".join(parts)


@dataclass
class ICUEvoMemoryDatabase:
    memory_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def add_snapshot(self, memory: ICUEvoMemory) -> None:
        self.memory_snapshots.append(deepcopy(memory.to_dict()))

    def to_dict(self) -> Dict[str, Any]:
        return {"memory_snapshots": self.memory_snapshots}

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)


class PerceptionAgent:
    def __init__(self, max_working_windows: int, window_duration_hours: float):
        self.max_working_windows = max_working_windows
        self.window_duration_hours = window_duration_hours

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
            end_hour=hours + self.window_duration_hours,
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
            # self._warn_once(
            #     seen=self._unmapped_vital_sources_logged,
            #     key=code_specifics or "<empty>",
            #     message="Unmapped vital code_specifics dropped: %s",
            #     values=(code_specifics,),
            # )
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


class AssessmentAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(
        self,
        raw_window_block: List[WorkingWindow],
        trend_block: List[TrendMemoryEntry],
        patient_metadata: Optional[Dict[str, Any]] = None,
        prior_episode_summary: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_episode_agent_prompt()
        patient_metadata_text = "\n".join(_format_patient_metadata_lines(patient_metadata))
        prior_episode_summary_text = _safe_text(prior_episode_summary) or "None"

        episode_input_lines = _format_episode_input_lines(raw_window_block, trend_block)
        episode_input_text = "\n".join(line for line in episode_input_lines if line is not None).strip() or "- None"

        if raw_window_block:
            first_window = raw_window_block[0]
            last_window = raw_window_block[-1]
            episode_start_time = f"hour {first_window.start_hour:.1f}"
            episode_end_time = f"hour {last_window.end_hour:.1f}"
            episode_duration = max(last_window.end_hour - first_window.start_hour, 0.0)
        else:
            episode_start_time = "N/A"
            episode_end_time = "N/A"
            episode_duration = 0.0

        prompt = _render_prompt(
            prompt_template,
            {
                "k": str(len(raw_window_block)),
                "duration": f"{episode_duration:.1f}",
                "episode_start_time": episode_start_time,
                "episode_end_time": episode_end_time,
                "patient_metadata": patient_metadata_text,
                "episode_input": episode_input_text,
                "prior_episode_summary": prior_episode_summary_text,
                "previous_episode_summary": prior_episode_summary_text,
                "prior_episode_summary_text": prior_episode_summary_text,
            },
        )

        parsed, raw, usage, parse_error, attempt_index = _chat_with_schema_validation(
            llm_client=self.llm_client,
            prompt=prompt,
            schema_validator=_validate_episode_output_schema,
            max_attempts=ICU_EVO_SCHEMA_MAX_ATTEMPTS,
        )
        if parse_error is not None:
            parse_error = f"{parse_error} | final_attempt={attempt_index}/{ICU_EVO_SCHEMA_MAX_ATTEMPTS}"

        return parsed, raw, usage, prompt, parse_error


class InsightAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(
        self,
        current_insights: List[Insight],
        episode: Episode,
        critical_events_entry: Optional[CriticalEventsMemoryEntry],
        trend_block: List[TrendMemoryEntry],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_insight_agent_prompt()

        if current_insights:
            insights_text = "\n".join(
                [f"[{insight.insight_id}] {_safe_text(insight.hypothesis)}" for insight in current_insights]
            )
        else:
            insights_text = "- None"

        episode_summary_text = (
            f"Window {episode.start_window}-{episode.end_window} "
            f"(hour {episode.start_hour:.1f}-{episode.end_hour:.1f}): {episode.episode_summary}"
        )
        critical_events_text = "- None"
        if critical_events_entry is not None and critical_events_entry.critical_events:
            critical_events_lines = _serialize_critical_event_strings(critical_events_entry.critical_events)
            if critical_events_lines:
                critical_events_text = "\n".join(critical_events_lines)

        trend_lines = _format_trend_block_lines(trend_block)
        vital_trends_text = "\n".join(line for line in trend_lines if line is not None).strip() or "- None"
        patient_metadata_text = "\n".join(_format_patient_metadata_lines(patient_metadata))

        prompt = _render_prompt(
            prompt_template,
            {
                "hypothesis_bank": insights_text,
                "patient_metadata": patient_metadata_text,
                "episode_summary": episode_summary_text,
                "critical_events": critical_events_text,
                "vital_trends": vital_trends_text,
                "observations": vital_trends_text,
            },
        )

        parsed, raw, usage, parse_error, attempt_index = _chat_with_schema_validation(
            llm_client=self.llm_client,
            prompt=prompt,
            schema_validator=_validate_insight_output_schema,
            max_attempts=ICU_EVO_SCHEMA_MAX_ATTEMPTS,
        )
        if parse_error is not None:
            parse_error = f"{parse_error} | final_attempt={attempt_index}/{ICU_EVO_SCHEMA_MAX_ATTEMPTS}"

        return parsed, raw, usage, prompt, parse_error


class PredictorAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def predict(
        self,
        memory: ICUEvoMemory,
        last_window_events: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_survival_prediction_prompt()
        prompt = _render_prompt(prompt_template, {"context": memory.to_text()})
        response = self.llm_client.chat(prompt=prompt, response_format="text")
        raw = response.get("content", "")
        usage = response.get("usage", {})

        parse_error: Optional[str] = None
        parsed: Dict[str, Any] = {}
        for candidate in (response.get("parsed"), raw):
            if candidate is None:
                continue
            try:
                parsed = _parse_json_response(candidate)
                parse_error = None
                break
            except Exception as exc:
                parse_error = str(exc)

        if parse_error is not None:
            parsed = {
                "prediction": "unknown",
                "confidence": "Low",
                "supporting_evidence": [],
                "rationale": f"Parsing error: {parse_error}",
            }

        return parsed, raw, usage, prompt, parse_error


class ICUEvoAgent:
    def __init__(
        self,
        provider: str,
        observation_config_path: str,
        episode_block_windows: int,
        insight_block_windows: int,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_logging: bool = False,
        window_duration_hours: float = 0.5,
        max_working_windows: int = 3,
        max_insights: int = 5,
        max_trajectory_entries: Optional[int] = None,
    ):
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.window_duration_hours = window_duration_hours
        self.max_working_windows = max(1, int(max_working_windows))
        self.max_insights = max(1, int(max_insights))
        if max_trajectory_entries is None:
            raise ValueError("max_trajectory_entries must be provided")
        self.max_trajectory_entries = max(1, int(max_trajectory_entries))
        self.episode_block_windows = max(1, int(episode_block_windows))
        self.insight_block_windows = max(1, int(insight_block_windows))
        self.observation_config_path = _safe_text(observation_config_path)
        if not self.observation_config_path:
            raise ValueError("observation_config_path must be a non-empty string")

        self.enable_logging = enable_logging
        self.call_logs: List[LLMCallLog] = []

        self.perception_agent = PerceptionAgent(
            max_working_windows=self.max_working_windows,
            window_duration_hours=window_duration_hours,
        )
        self.observation_agent = ObservationAgent(self.observation_config_path)
        self.assessment_agent = AssessmentAgent(self.llm_client)
        self.insight_agent = InsightAgent(self.llm_client)
        self.predictor_agent = PredictorAgent(self.llm_client)

        self.total_patients = 0
        self.total_tokens_used = 0
        self.total_observation_runs = 0
        self.total_episode_calls = 0
        self.total_insight_calls = 0
        self.total_predictor_calls = 0
        self.total_pre_icu_compression_calls = 0
        self.total_pre_icu_compression_tokens = 0
        self.total_grounding_rejections = 0
        self.total_insights_pruned = 0
        self.total_trajectory_entries_pruned = 0

        self._current_patient_id = ""
        self._event_id_to_window: Dict[int, int] = {}
        self._event_id_to_time: Dict[int, str] = {}
        self._event_id_to_raw_event: Dict[int, Dict[str, Any]] = {}
        self._event_id_to_name_str: Dict[int, str] = {}
        self._next_insight_id = 1
        self._next_episode_id = 1

    def clear_logs(self) -> None:
        self.call_logs = []

    def get_logs(self) -> List[Dict[str, Any]]:
        return [log.to_dict() for log in self.call_logs]

    def save_logs(self, path: str) -> None:
        payload = {
            "total_calls": len(self.call_logs),
            "total_tokens": self.total_tokens_used,
            "calls": self.get_logs(),
        }
        with open(path, "w") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_patients": self.total_patients,
            "total_tokens_used": self.total_tokens_used,
            "total_observation_runs": self.total_observation_runs,
            "total_episode_calls": self.total_episode_calls,
            "total_insight_calls": self.total_insight_calls,
            "total_predictor_calls": self.total_predictor_calls,
            "total_pre_icu_compression_calls": self.total_pre_icu_compression_calls,
            "total_pre_icu_compression_tokens": self.total_pre_icu_compression_tokens,
            "total_grounding_rejections": self.total_grounding_rejections,
            "total_insights_pruned": self.total_insights_pruned,
            "total_trajectory_entries_pruned": self.total_trajectory_entries_pruned,
            "episode_block_windows": self.episode_block_windows,
            "insight_block_windows": self.insight_block_windows,
            "max_trajectory_entries": self.max_trajectory_entries,
            "observation_config_path": self.observation_config_path,
            "total_llm_calls": len(self.call_logs) if self.enable_logging else 0,
        }

    def _log_call(
        self,
        step_type: str,
        window_index: int,
        hours_since_admission: float,
        prompt: str,
        response: str,
        usage: Dict[str, Any],
        parsed_response: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        usage_dict = usage if isinstance(usage, dict) else {}
        input_tokens = _normalize_token_count(usage_dict.get("input_tokens", 0))
        output_tokens = _normalize_token_count(usage_dict.get("output_tokens", 0))
        self.total_tokens_used += input_tokens + output_tokens

        if not self.enable_logging:
            return

        log_metadata = dict(metadata or {})
        log_metadata.update(
            {
                "step_type": step_type,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
            }
        )

        self.call_logs.append(
            LLMCallLog(
                timestamp=datetime.now().isoformat(),
                patient_id=self._current_patient_id,
                window_index=window_index,
                hours_since_admission=hours_since_admission,
                prompt=prompt,
                response=response,
                parsed_response=parsed_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=log_metadata,
            )
        )

    @staticmethod
    def _first_window_pre_icu_history(
        windows: List[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if not windows:
            return None
        first_window = windows[0]
        if not isinstance(first_window, dict):
            return None
        pre_icu_history = first_window.get("pre_icu_history")
        if not isinstance(pre_icu_history, dict):
            return None
        return first_window, pre_icu_history

    def _compress_pre_icu_history_for_windows(self, windows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        first_payload = self._first_window_pre_icu_history(windows)
        if first_payload is None:
            return None

        first_window, pre_icu_history = first_payload
        source = _safe_text(pre_icu_history.get("source")).lower()
        raw_content = _safe_text(pre_icu_history.get("content"))
        if source in {"", "none", "disabled"} or not raw_content:
            return None

        if source == "llm_compressed":
            return {
                "summary": raw_content,
                "source": source,
                "items": int(_normalize_token_count(pre_icu_history.get("items"))),
                "history_hours": _to_optional_float(pre_icu_history.get("history_hours")),
                "historical_discharge_summary_items": int(
                    _normalize_token_count(pre_icu_history.get("historical_discharge_summary_items"))
                ),
                "compression": pre_icu_history.get("compression") if isinstance(pre_icu_history, dict) else None,
            }

        prompt = format_pre_icu_compression_prompt(pre_icu_history)
        parsed, response_raw, response_usage, parse_error, attempt_index = _chat_with_schema_validation(
            llm_client=self.llm_client,
            prompt=prompt,
            schema_validator=_validate_pre_icu_compression_output_schema,
            max_attempts=ICU_EVO_SCHEMA_MAX_ATTEMPTS,
        )
        compressed_text = _safe_text(parsed.get("compressed_pre_icu_history"))
        if parse_error is not None:
            parse_error = f"{parse_error} | final_attempt={attempt_index}/{ICU_EVO_SCHEMA_MAX_ATTEMPTS}"

        input_tokens = _normalize_token_count(response_usage.get("input_tokens", 0))
        output_tokens = _normalize_token_count(response_usage.get("output_tokens", 0))
        self.total_pre_icu_compression_calls += 1
        self.total_pre_icu_compression_tokens += input_tokens + output_tokens

        self._log_call(
            step_type=PRE_ICU_COMPRESSION_STEP_TYPE,
            window_index=-1,
            hours_since_admission=float(first_window.get("hours_since_admission", 0.0)),
            prompt=prompt,
            response=response_raw,
            usage=response_usage if isinstance(response_usage, dict) else {},
            parsed_response=parsed if isinstance(parsed, dict) else None,
            metadata={
                "pre_icu_history_source": first_window.get("pre_icu_history_source"),
                "pre_icu_history_items": first_window.get("pre_icu_history_items"),
                "original_source": source or "unknown",
                "original_content_chars": len(raw_content),
                "compressed_chars": len(compressed_text),
                "parse_error": parse_error,
            },
        )

        if not compressed_text:
            return None

        compression_metadata = {
            "method": "llm",
            "step_type": PRE_ICU_COMPRESSION_STEP_TYPE,
            "original_source": source or "unknown",
            "original_content_chars": len(raw_content),
            "compressed_chars": len(compressed_text),
        }

        compressed_pre_icu_history = dict(pre_icu_history)
        compressed_pre_icu_history["source"] = "llm_compressed"
        compressed_pre_icu_history["content"] = compressed_text
        compressed_pre_icu_history["compression"] = compression_metadata

        for window in windows:
            if not isinstance(window, dict):
                continue
            if not isinstance(window.get("pre_icu_history"), dict):
                continue
            window["pre_icu_history"] = dict(compressed_pre_icu_history)
            window["pre_icu_history_source"] = "llm_compressed"

        return {
            "summary": compressed_text,
            "source": "llm_compressed",
            "items": int(_normalize_token_count(pre_icu_history.get("items"))),
            "history_hours": _to_optional_float(pre_icu_history.get("history_hours")),
            "historical_discharge_summary_items": int(
                _normalize_token_count(pre_icu_history.get("historical_discharge_summary_items"))
            ),
            "compression": compression_metadata,
        }

    def _build_patient_metadata_with_pre_icu_summary(
        self,
        *,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = deepcopy(patient_metadata)
        compressed_pre_icu = self._compress_pre_icu_history_for_windows(windows)
        if not isinstance(compressed_pre_icu, dict):
            return enriched

        summary = _safe_text(compressed_pre_icu.get("summary"))
        if not summary:
            return enriched

        enriched["pre_icu_history_summary"] = summary
        enriched["pre_icu_history_source"] = _safe_text(compressed_pre_icu.get("source")) or "llm_compressed"
        enriched["pre_icu_history_items"] = int(_normalize_token_count(compressed_pre_icu.get("items")))
        history_hours = _to_optional_float(compressed_pre_icu.get("history_hours"))
        if history_hours is not None:
            enriched["pre_icu_history_hours"] = history_hours
        enriched["historical_discharge_summary_items"] = int(
            _normalize_token_count(compressed_pre_icu.get("historical_discharge_summary_items"))
        )

        return enriched

    def _register_event_ids(self, current_events: List[Dict[str, Any]], window_index: int) -> None:
        for event in current_events:
            event_id = _extract_event_reference_id(event)
            if event_id is None:
                continue

            self._event_id_to_window[event_id] = window_index
            raw_event = deepcopy(event)
            raw_event["event_id"] = event_id
            raw_event["event_index"] = event_id
            raw_event["icu_event_index"] = event_id

            existing = self._event_id_to_raw_event.get(event_id)
            if existing is None or len(raw_event) >= len(existing):
                self._event_id_to_raw_event[event_id] = raw_event

            event_time = _safe_text(raw_event.get("time"))
            if event_time:
                self._event_id_to_time[event_id] = event_time
            elif event_id not in self._event_id_to_time:
                self._event_id_to_time[event_id] = ""

            name_str = _format_single_event_line(raw_event)
            if name_str:
                existing_name = self._event_id_to_name_str.get(event_id, "")
                if not existing_name or len(name_str) >= len(existing_name):
                    self._event_id_to_name_str[event_id] = name_str
            elif event_id not in self._event_id_to_name_str:
                self._event_id_to_name_str[event_id] = f"[{event_id}]"

    def _resolve_event_name_str(self, event_id: int) -> str:
        resolved = _safe_text(self._event_id_to_name_str.get(event_id))
        if resolved:
            return resolved
        raw_event = self._event_id_to_raw_event.get(event_id)
        if isinstance(raw_event, dict):
            resolved = _format_single_event_line(raw_event)
            if resolved:
                self._event_id_to_name_str[event_id] = resolved
                return resolved
        return f"[{event_id}]"

    def _resolve_grounded_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        raw_event = self._event_id_to_raw_event.get(event_id)
        if not isinstance(raw_event, dict):
            return None
        return {
            "event_id": event_id,
            "timestamp": _safe_text(raw_event.get("time")) or self._event_id_to_time.get(event_id, ""),
            "event_name": _derive_raw_event_name(raw_event),
            "code": _safe_text(raw_event.get("code")),
            "code_specifics": _safe_text(raw_event.get("code_specifics")),
            "numeric_value": _to_optional_float(raw_event.get("numeric_value")),
            "text_value": _safe_text(raw_event.get("text_value")),
            "raw_event": deepcopy(raw_event),
        }

    def _resolve_supporting_events(self, event_ids: List[int]) -> List[Dict[str, Any]]:
        supporting_events: List[Dict[str, Any]] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            grounded = self._resolve_grounded_event(event_id)
            if grounded is None:
                self.total_grounding_rejections += 1
                continue
            supporting_events.append(grounded)
        return supporting_events

    def _validate_event_ids(self, event_ids: List[int]) -> List[int]:
        valid: List[int] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            if event_id not in self._event_id_to_window:
                self.total_grounding_rejections += 1
                continue
            if event_id not in self._event_id_to_raw_event:
                self.total_grounding_rejections += 1
                continue
            valid.append(event_id)
        return valid

    def _trend_entry_from_detector_output(
        self,
        detector_output: Dict[str, Any],
        window_index: int,
        start_hour: float,
        end_hour: float,
    ) -> TrendMemoryEntry:
        vital_trends = detector_output.get("vital_trends", {})
        if not isinstance(vital_trends, dict):
            vital_trends = {}
        return TrendMemoryEntry(
            window_index=window_index,
            start_hour=float(start_hour),
            end_hour=float(end_hour),
            raw_event_count=_normalize_token_count(detector_output.get("raw_event_count", 0)),
            vital_trends=deepcopy(vital_trends),
        )

    def _collect_event_ids_from_raw_window_block(self, raw_window_block: List[WorkingWindow]) -> List[int]:
        event_ids: List[int] = []
        for window in raw_window_block:
            for event in window.events:
                event_id = _extract_event_reference_id(event if isinstance(event, dict) else {})
                if event_id is None:
                    continue
                event_ids.append(event_id)
        return self._validate_event_ids(event_ids)

    def _build_default_episode_payload(
        self,
        raw_window_block: List[WorkingWindow],
        trend_block: List[TrendMemoryEntry],
    ) -> Dict[str, Any]:
        text_segments = []
        supporting_event_ids = self._collect_event_ids_from_raw_window_block(raw_window_block)
        trend_by_window = _trend_entry_by_window(trend_block)
        for window in raw_window_block:
            trend_entry = trend_by_window.get(window.window_id)
            trend_summary = "no tracked vitals"
            if trend_entry is not None:
                trend_summary = _trend_entry_summary(trend_entry.to_dict())
            text_segments.append(
                f"Window {window.window_id} (hour {window.start_hour:.1f}-{window.end_hour:.1f}): "
                f"raw_events={len(window.events)}; {trend_summary}"
            )
        if not text_segments:
            text_segments.append("No clinically meaningful progression identified in this episode block.")
        return {
            "episode_summary": {
                "text": " ".join(text_segments),
                "supporting_event_ids": supporting_event_ids[:12],
            },
            "critical_events": [],
        }

    def _ground_episode_critical_events(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            return []

        grounded_items: List[Dict[str, Any]] = []
        seen = set()
        for item in payload:
            if not isinstance(item, dict):
                continue

            critical_event_text = _safe_text(item.get("event"))
            if not critical_event_text:
                critical_event_text = _safe_text(item.get("critical_event"))
            if not critical_event_text:
                critical_event_text = _safe_text(item.get("name"))

            reason = _safe_text(item.get("reason"))
            if not reason:
                reason = _safe_text(item.get("rationale"))

            supporting_event_ids = _normalize_int_list(
                item.get("supporting_event_ids", item.get("supporting_evidence", []))
            )
            direct_event_ids = _normalize_int_list([item.get("event_id")])
            supporting_event_ids = self._validate_event_ids(direct_event_ids + supporting_event_ids)
            supporting_events = self._resolve_supporting_events(supporting_event_ids)
            grounded_supporting_event_ids = [
                int(event.get("event_id")) for event in supporting_events if event.get("event_id") is not None
            ]

            if not critical_event_text and supporting_events:
                critical_event_text = _safe_text(supporting_events[0].get("event_name"))
            if not critical_event_text:
                continue

            dedupe_key = (critical_event_text, tuple(grounded_supporting_event_ids))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            grounded_items.append(
                {
                    "event": critical_event_text,
                    "reason": reason,
                    "supporting_event_ids": grounded_supporting_event_ids,
                    "supporting_events": supporting_events,
                }
            )

        return grounded_items

    def _build_critical_events_memory_entry(
        self,
        *,
        episode_id: int,
        raw_window_block: List[WorkingWindow],
        critical_events: List[Dict[str, Any]],
    ) -> CriticalEventsMemoryEntry:
        first_window = raw_window_block[0]
        last_window = raw_window_block[-1]
        return CriticalEventsMemoryEntry(
            episode_id=episode_id,
            start_window=first_window.window_id,
            end_window=last_window.window_id,
            start_hour=first_window.start_hour,
            end_hour=last_window.end_hour,
            critical_events=deepcopy(critical_events),
        )

    def _ground_episode(
        self,
        payload: Dict[str, Any],
        raw_window_block: List[WorkingWindow],
        trend_block: List[TrendMemoryEntry],
    ) -> Tuple[Episode, CriticalEventsMemoryEntry]:
        source_payload = payload if isinstance(payload, dict) else {}
        summary_payload = source_payload.get("episode_summary", source_payload.get("episode", {}))
        if not isinstance(summary_payload, dict):
            summary_payload = {}

        episode_text = _safe_text(summary_payload.get("text"))
        if not episode_text:
            episode_text = _safe_text(summary_payload.get("episode_summary"))
        if not episode_text:
            episode_text = _safe_text(source_payload.get("text"))

        default_payload = self._build_default_episode_payload(raw_window_block, trend_block)
        if not episode_text:
            episode_text = _safe_text(default_payload.get("episode_summary", {}).get("text"))

        supporting_event_ids = _normalize_int_list(
            summary_payload.get(
                "supporting_event_ids",
                summary_payload.get("supporting_evidence", source_payload.get("supporting_event_ids", [])),
            )
        )
        supporting_event_ids = self._validate_event_ids(supporting_event_ids)
        if not supporting_event_ids:
            supporting_event_ids = self._validate_event_ids(
                _normalize_int_list(default_payload.get("episode_summary", {}).get("supporting_event_ids", []))
            )

        supporting_events = self._resolve_supporting_events(supporting_event_ids)
        supporting_event_ids = [
            int(item.get("event_id")) for item in supporting_events if item.get("event_id") is not None
        ]

        critical_events_payload = source_payload.get("critical_events", default_payload.get("critical_events", []))
        grounded_critical_events = self._ground_episode_critical_events(critical_events_payload)

        first_window = raw_window_block[0]
        last_window = raw_window_block[-1]
        episode_id = self._next_episode_id
        episode = Episode(
            episode_id=episode_id,
            start_window=first_window.window_id,
            end_window=last_window.window_id,
            start_hour=first_window.start_hour,
            end_hour=last_window.end_hour,
            episode_summary=episode_text,
            supporting_event_ids=supporting_event_ids,
            supporting_events=supporting_events,
        )
        critical_events_entry = self._build_critical_events_memory_entry(
            episode_id=episode_id,
            raw_window_block=raw_window_block,
            critical_events=grounded_critical_events,
        )
        self._next_episode_id += 1
        return episode, critical_events_entry

    def _build_evidence_events(self, event_ids: List[int]) -> List[EvidenceEvent]:
        evidence_events: List[EvidenceEvent] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            evidence_events.append(EvidenceEvent(event_id=event_id, name_str=self._resolve_event_name_str(event_id)))
        return evidence_events

    def _refresh_insight_evidence_objects(self, insight: Insight) -> None:
        insight.supporting_evidence = self._build_evidence_events(insight.supporting_event_ids)
        insight.counter_evidence = self._build_evidence_events(insight.counter_event_ids)

    def _recompute_insight_scores(self, insights: List[Insight]) -> None:
        for insight in insights:
            supporting_count = len(insight.supporting_event_ids) + len(insight.supporting_trend_evidence)
            counter_count = len(insight.counter_event_ids) + len(insight.counter_trend_evidence)
            insight.score = float(supporting_count - counter_count)

    def _apply_insight_updates(
        self,
        existing_insights: List[Insight],
        parsed_payload: Dict[str, Any],
    ) -> List[Insight]:
        now_ts = time.time()
        insights_by_id = {insight.insight_id: deepcopy(insight) for insight in existing_insights}

        updates = parsed_payload.get("insight_updates", parsed_payload.get("updated_insights", []))
        if not isinstance(updates, list):
            updates = []

        new_insights = parsed_payload.get("new_insights", [])
        if not isinstance(new_insights, list):
            new_insights = []

        for item in updates + new_insights:
            if not isinstance(item, dict):
                continue

            hypothesis = _safe_text(item.get("hypothesis"))
            supporting_evidence_ids, supporting_trend_evidence = _split_evidence_references(
                item.get("supporting_evidence", [])
            )
            counter_evidence_ids, counter_trend_evidence = _split_evidence_references(item.get("counter_evidence", []))
            supporting = self._validate_event_ids(
                _normalize_int_list(item.get("supporting_event_ids", [])) + supporting_evidence_ids
            )
            counter = self._validate_event_ids(
                _normalize_int_list(item.get("counter_event_ids", [])) + counter_evidence_ids
            )
            supporting_trends = _normalize_text_list(
                _normalize_text_list(item.get("supporting_trend_evidence", [])) + supporting_trend_evidence
            )
            counter_trends = _normalize_text_list(
                _normalize_text_list(item.get("counter_trend_evidence", [])) + counter_trend_evidence
            )

            if not hypothesis and not supporting and not counter and not supporting_trends and not counter_trends:
                continue

            raw_insight_id = item.get("insight_id", item.get("hypothesis_id"))
            insight_id: Optional[int] = None
            if raw_insight_id is not None:
                try:
                    parsed_id = int(raw_insight_id)
                    if parsed_id in insights_by_id:
                        insight_id = parsed_id
                except (TypeError, ValueError):
                    insight_id = None

            if insight_id is not None:
                insight = insights_by_id[insight_id]
                if hypothesis:
                    insight.hypothesis = hypothesis
                insight.supporting_event_ids = self._validate_event_ids(
                    _normalize_int_list(insight.supporting_event_ids + supporting)
                )
                insight.counter_event_ids = self._validate_event_ids(
                    _normalize_int_list(insight.counter_event_ids + counter)
                )
                insight.supporting_trend_evidence = _normalize_text_list(
                    insight.supporting_trend_evidence + supporting_trends
                )
                insight.counter_trend_evidence = _normalize_text_list(insight.counter_trend_evidence + counter_trends)
                self._refresh_insight_evidence_objects(insight)
                insight.updated_at = now_ts
                insights_by_id[insight_id] = insight
                continue

            if not hypothesis:
                continue

            insight_id = self._next_insight_id
            self._next_insight_id += 1
            new_insight = Insight(
                insight_id=insight_id,
                hypothesis=hypothesis,
                supporting_event_ids=supporting,
                counter_event_ids=counter,
                supporting_trend_evidence=supporting_trends,
                counter_trend_evidence=counter_trends,
                created_at=now_ts,
                updated_at=now_ts,
                score=0.0,
            )
            self._refresh_insight_evidence_objects(new_insight)
            insights_by_id[insight_id] = new_insight

        updated_insights = list(insights_by_id.values())
        for insight in updated_insights:
            self._refresh_insight_evidence_objects(insight)
        self._recompute_insight_scores(updated_insights)

        if len(updated_insights) > self.max_insights:
            updated_insights.sort(key=lambda insight: (insight.score, insight.updated_at))
            prune_count = len(updated_insights) - self.max_insights
            self.total_insights_pruned += prune_count
            updated_insights = updated_insights[prune_count:]

        updated_insights.sort(key=lambda insight: insight.insight_id)
        return updated_insights

    def _build_default_insight_payload(self, episode: Episode) -> Dict[str, Any]:
        if not episode.supporting_event_ids:
            return {"insight_updates": [], "new_insights": []}
        return {
            "insight_updates": [],
            "new_insights": [
                {
                    "hypothesis": (
                        f"Window {episode.start_window}-{episode.end_window} trajectory reveals a patient-specific pattern "
                        f"that requires continued monitoring."
                    ),
                    "supporting_event_ids": episode.supporting_event_ids[:2],
                    "counter_event_ids": [],
                }
            ],
        }

    def _append_trend_entry(self, memory: ICUEvoMemory, entry: TrendMemoryEntry) -> None:
        memory.trend_memory.append(entry)
        memory.last_processed_window_index = entry.window_index
        memory.last_processed_start_hour = entry.start_hour
        memory.last_processed_end_hour = entry.end_hour

    def _prune_episode_level_memory(self, memory: ICUEvoMemory) -> None:
        if len(memory.trajectory_memory) <= self.max_trajectory_entries:
            return

        prune_count = len(memory.trajectory_memory) - self.max_trajectory_entries
        self.total_trajectory_entries_pruned += prune_count
        memory.trajectory_memory = memory.trajectory_memory[prune_count:]

        if prune_count <= 0:
            return
        if prune_count >= len(memory.critical_events_memory):
            memory.critical_events_memory = []
            return
        memory.critical_events_memory = memory.critical_events_memory[prune_count:]

    def create_memory_snapshots(
        self,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
        verbose: bool = True,
    ) -> Tuple[ICUEvoMemory, ICUEvoMemoryDatabase]:
        self.total_patients += 1
        self._current_patient_id = (
            f"{patient_metadata.get('subject_id', 'unknown')}_{patient_metadata.get('icu_stay_id', 'unknown')}"
        )
        self._event_id_to_window = {}
        self._event_id_to_time = {}
        self._event_id_to_raw_event = {}
        self._event_id_to_name_str = {}
        self._next_insight_id = 1
        self._next_episode_id = 1

        patient_metadata_with_pre_icu = self._build_patient_metadata_with_pre_icu_summary(
            windows=windows,
            patient_metadata=patient_metadata,
        )
        memory = ICUEvoMemory(patient_metadata=deepcopy(patient_metadata_with_pre_icu))
        memory_db = ICUEvoMemoryDatabase()
        episode_raw_window_block: List[WorkingWindow] = []
        latest_episode: Optional[Episode] = None
        latest_critical_events_entry: Optional[CriticalEventsMemoryEntry] = None
        latest_episode_trend_block: List[TrendMemoryEntry] = []

        if verbose:
            print(f"Processing patient with {len(windows)} windows...")
            pre_icu_summary = _safe_text(memory.patient_metadata.get("pre_icu_history_summary"))
            if pre_icu_summary:
                print(f"  Pre-ICU summary prepared ({len(pre_icu_summary)} chars)")

        for idx, window in enumerate(windows):
            current_events = window.get("current_events", [])
            if not isinstance(current_events, list):
                current_events = []
            hours = float(window.get("hours_since_admission", 0.0))
            end_hour = hours + self.window_duration_hours

            if verbose:
                print(f"  Window {idx+1}/{len(windows)} (Hour {hours:.1f})...", end=" ")

            self._register_event_ids(current_events, idx)
            memory.working_memory = self.perception_agent.update_working_memory(memory.working_memory, window, idx)
            episode_raw_window_block.append(deepcopy(memory.working_memory[-1]))
            if len(episode_raw_window_block) > self.episode_block_windows:
                episode_raw_window_block = episode_raw_window_block[-self.episode_block_windows :]

            observation_output = self.observation_agent.analyze(window)
            self.total_observation_runs += 1
            trend_entry = self._trend_entry_from_detector_output(
                detector_output=observation_output,
                window_index=idx,
                start_hour=hours,
                end_hour=end_hour,
            )
            self._append_trend_entry(memory, trend_entry)

            run_episode_block = len(memory.trend_memory) >= self.episode_block_windows and (
                len(memory.trend_memory) % self.episode_block_windows == 0
            )

            if run_episode_block:
                trend_block = memory.trend_memory[-self.episode_block_windows :]
                raw_window_block = episode_raw_window_block[-self.episode_block_windows :]
                prior_episode_summary = _format_prior_episode_summary_text(memory.trajectory_memory[-1:])
                episode_parsed, episode_raw, episode_usage, episode_prompt, episode_parse_error = (
                    self.assessment_agent.analyze(
                        raw_window_block=raw_window_block,
                        trend_block=trend_block,
                        patient_metadata=memory.patient_metadata,
                        prior_episode_summary=prior_episode_summary,
                    )
                )
                self.total_episode_calls += 1

                if episode_parse_error is not None:
                    episode_parsed = self._build_default_episode_payload(raw_window_block, trend_block)

                grounded_episode, grounded_critical_events = self._ground_episode(
                    episode_parsed,
                    raw_window_block,
                    trend_block,
                )
                latest_episode = grounded_episode
                latest_critical_events_entry = grounded_critical_events
                latest_episode_trend_block = deepcopy(trend_block)
                memory.trajectory_memory.append(grounded_episode.to_dict())
                memory.critical_events_memory.append(grounded_critical_events)
                self._prune_episode_level_memory(memory)

                self._log_call(
                    step_type="assessment_agent",
                    window_index=idx,
                    hours_since_admission=hours,
                    prompt=episode_prompt,
                    response=episode_raw,
                    usage=episode_usage,
                    parsed_response=episode_parsed,
                    metadata={
                        "episode_parse_error": episode_parse_error,
                        "episode_id": grounded_episode.episode_id,
                        "episode_start_window": grounded_episode.start_window,
                        "episode_end_window": grounded_episode.end_window,
                        "episode_support_count": len(grounded_episode.supporting_event_ids),
                        "episode_critical_event_count": len(grounded_critical_events.critical_events),
                    },
                )

            run_insight_block = len(memory.trend_memory) >= self.insight_block_windows and (
                len(memory.trend_memory) % self.insight_block_windows == 0
            )

            if run_insight_block and latest_episode is not None:
                insight_parsed, insight_raw, insight_usage, insight_prompt, insight_parse_error = (
                    self.insight_agent.analyze(
                        current_insights=memory.insights,
                        episode=latest_episode,
                        critical_events_entry=latest_critical_events_entry,
                        trend_block=latest_episode_trend_block,
                        patient_metadata=memory.patient_metadata,
                    )
                )
                self.total_insight_calls += 1

                if insight_parse_error is not None:
                    insight_parsed = self._build_default_insight_payload(latest_episode)

                memory.insights = self._apply_insight_updates(memory.insights, insight_parsed)

                self._log_call(
                    step_type="insight_agent",
                    window_index=idx,
                    hours_since_admission=hours,
                    prompt=insight_prompt,
                    response=insight_raw,
                    usage=insight_usage,
                    parsed_response=insight_parsed,
                    metadata={
                        "insight_parse_error": insight_parse_error,
                        "num_insights": len(memory.insights),
                        "episode_id": latest_episode.episode_id,
                    },
                )
            else:
                self._recompute_insight_scores(memory.insights)

            memory_db.add_snapshot(memory)

            if verbose:
                print(
                    f"trends={len(memory.trend_memory)} "
                    f"episodes={len(memory.trajectory_memory)} "
                    f"critical_events={len(memory.critical_events_memory)} "
                    f"insights={len(memory.insights)}"
                )

        return memory, memory_db

    def predict_from_memory(
        self,
        memory: ICUEvoMemory,
        *,
        last_window_events: Optional[List[Dict[str, Any]]] = None,
        hours_since_admission: float = 0.0,
        window_index: int = -1,
    ) -> Dict[str, Any]:
        if last_window_events is None:
            last_window_events = []

        prediction, pred_raw, pred_usage, pred_prompt, pred_parse_error = self.predictor_agent.predict(
            memory=memory,
            last_window_events=last_window_events,
        )
        self.total_predictor_calls += 1

        self._log_call(
            step_type="icu_evo_predictor",
            window_index=int(window_index),
            hours_since_admission=float(hours_since_admission),
            prompt=pred_prompt,
            response=pred_raw,
            usage=pred_usage,
            parsed_response=prediction,
            metadata={"predictor_parse_error": pred_parse_error},
        )

        return prediction

    def run_patient_trajectory(
        self,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], ICUEvoMemory, ICUEvoMemoryDatabase]:
        memory, memory_db = self.create_memory_snapshots(
            windows=windows,
            patient_metadata=patient_metadata,
            verbose=verbose,
        )

        last_window = windows[-1] if windows else {}
        last_window_events = last_window.get("current_events", []) if isinstance(last_window, dict) else []
        if not isinstance(last_window_events, list):
            last_window_events = []
        last_hours = float(last_window.get("hours_since_admission", 0.0)) if isinstance(last_window, dict) else 0.0

        prediction = self.predict_from_memory(
            memory=memory,
            last_window_events=last_window_events,
            hours_since_admission=last_hours,
            window_index=memory.last_processed_window_index,
        )

        return prediction, memory, memory_db
