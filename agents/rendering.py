"""Render observed events, prompts, and ICU-Evo memory."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from utils.event_format import format_event_line as format_shared_event_line
from utils.event_format import format_event_lines as format_shared_event_lines
from agents.validation import _safe_text, _normalize_token_count
from agents.validation import (
    _normalize_int_list,
    _to_optional_float,
)
if TYPE_CHECKING:
    from agents.memory import (
        TrendMemoryEntry,
        WorkingWindow,
    )

_PROMPT_PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}|\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


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


def _fill_prompt_placeholder(template: str, key: str, value: str) -> str:
    return template.replace(f"{{{{{key}}}}}", value).replace(f"{{{key}}}", value)


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


def render_snapshot(snapshot: Dict[str, Any]) -> str:
    """Render an ICU-Evo memory snapshot dict into predictor context text."""
    patient_metadata = snapshot.get("patient_metadata", {})
    trajectory_memory = snapshot.get("trajectory_memory", [])
    trend_memory = snapshot.get("trend_memory", [])
    critical_events_memory = snapshot.get("critical_events_memory", [])
    insights = snapshot.get("insights", [])
    working_memory = snapshot.get("working_memory", [])

    parts: List[str] = ["## Patient Metadata"]
    if isinstance(patient_metadata, dict) and patient_metadata:
        for key, value in patient_metadata.items():
            if key in {"subject_id", "icu_stay_id"}:
                continue
            parts.append(f"{key}: {value}")
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Trajectory of the ICU Stay")
    if isinstance(trajectory_memory, list) and trajectory_memory:
        for item in trajectory_memory:
            if not isinstance(item, dict):
                parts.append(f"- {item}")
                continue
            item_type = item.get("type")
            if item_type == "episode":
                parts.append(
                    f"- Window {item.get('start_window')}-{item.get('end_window')} "
                    f"(hour {_to_optional_float(item.get('start_hour')):.1f}-{_to_optional_float(item.get('end_hour')):.1f}): "
                    f"{item.get('episode_summary', '')}"
                )
            else:
                parts.append(f"- {item}")
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Trend Memory")
    if isinstance(trend_memory, list) and trend_memory:
        for item in trend_memory:
            if isinstance(item, dict):
                parts.extend(render_trend(item))
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Critical Events Memory")
    if isinstance(critical_events_memory, list) and critical_events_memory:
        for item in critical_events_memory:
            if not isinstance(item, dict):
                continue
            parts.append(
                f"Window {item.get('start_window')}-{item.get('end_window')} "
                f"(hour {_to_optional_float(item.get('start_hour')):.1f}-{_to_optional_float(item.get('end_hour')):.1f})"
            )
            critical_events = item.get("critical_events", [])
            if isinstance(critical_events, list) and critical_events:
                for event in critical_events:
                    event_text = str(event).strip()
                    if event_text:
                        parts.append(event_text)
            else:
                parts.append("- None")
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Patient Specific Insights")
    if isinstance(insights, list) and insights:
        for item in insights:
            if not isinstance(item, dict):
                parts.append(f"- {item}")
                continue
            insight_id = item.get("insight_id")
            score = item.get("score")
            hypothesis = item.get("hypothesis", "")
            if isinstance(score, (int, float)):
                parts.append(f"- I{insight_id} score={float(score):.3f}: {hypothesis}")
            else:
                parts.append(f"- I{insight_id}: {hypothesis}")
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Current Window Observation")
    if isinstance(working_memory, list) and working_memory:
        current_window = working_memory[-1]
        if isinstance(current_window, dict):
            start_hour = float(current_window["start_hour"])
            end_hour = float(current_window["end_hour"])
            parts.append(f"Window {current_window.get('window_id')} (Hour {start_hour:.1f}-{end_hour:.1f})")
            current_events = current_window.get("events", [])
            if isinstance(current_events, list) and current_events:
                for event in current_events:
                    event_text = str(event).strip()
                    if event_text:
                        parts.append(event_text)
            else:
                parts.append("- (No events)")
        else:
            parts.append("- None")
    else:
        parts.append("- None")

    return "\n".join(parts)


def render_trend(entry: dict) -> list[str]:
    start, end = entry["start_hour"], entry["end_hour"]
    if entry["trend_scope"] == "global":
        title = f"Global Trend (window {entry['start_window']}-{entry['end_window']}, hour {start:.1f}-{end:.1f})"
    else:
        title = f"Current Window Trend (window {entry['window_index']}, hour {start:.1f}-{end:.1f})"
    lines = [f"### {title}", "Vital trends:"]
    for name, stats in sorted(entry["vital_trends"].items()):
        lines.append(f"- {name}: mean={stats['mean']}, min={stats['min']}, max={stats['max']}, count={stats['count']}")
    return lines if entry["vital_trends"] else lines + ["- None"]
