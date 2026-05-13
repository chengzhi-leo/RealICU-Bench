"""Shared event-line formatting helpers for Oracle prompts and parser outputs."""

from __future__ import annotations

import json
import math
from typing import Any, List, Mapping, Sequence

from utils.time_format import format_timestamp_minute


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "null", "none"}
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def format_numeric_value(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_event_line(
    event: Mapping[str, Any],
    *,
    time_keys: Sequence[str] = ("time", "start_time"),
    missing_time_text: str | None = None,
    missing_code_text: str | None = None,
    empty_as_json: bool = True,
) -> str:
    """
    Format one event into a compact stable line.

    The default behavior matches Oracle prompt context rendering.
    """
    parts: list[str] = []

    # Prefer ICU-local stable IDs across windows.
    event_id = event.get("icu_event_index")
    if _is_missing_value(event_id):
        event_id = event.get("event_index")
    if _is_missing_value(event_id):
        event_id = event.get("event_id")
    if _is_missing_value(event_id):
        event_id = event.get("event_idx")
    if _is_missing_value(event_id):
        event_id = event.get("prompt_event_id")
    if _is_missing_value(event_id):
        event_id = event.get("relative_event_id")
    if not _is_missing_value(event_id):
        try:
            parts.append(f"[{int(event_id)}]")
        except (TypeError, ValueError):
            parts.append(f"[{event_id}]")

    time_value: Any = None
    for key in time_keys:
        candidate = event.get(key)
        if not _is_missing_value(candidate):
            time_value = candidate
            break
    if time_value is not None:
        parts.append(format_timestamp_minute(time_value))
    elif missing_time_text is not None:
        parts.append(missing_time_text)

    code = event.get("code")
    if not _is_missing_value(code):
        parts.append(str(code))
    elif missing_code_text is not None:
        parts.append(missing_code_text)

    details = event.get("code_specifics")
    if not _is_missing_value(details):
        parts.append(str(details))

    numeric_value = event.get("numeric_value")
    if not _is_missing_value(numeric_value):
        parts.append(f"={format_numeric_value(numeric_value)}")

    text_value = event.get("text_value")
    if not _is_missing_value(text_value):
        parts.append(str(text_value))

    if parts:
        return " ".join(parts)

    if empty_as_json:
        return json.dumps(dict(event), ensure_ascii=False, default=str)

    return ""


def format_event_lines(
    events: Sequence[Mapping[str, Any]],
    *,
    empty_text: str = "(No events)",
) -> List[str]:
    """
    Format a sequence of events using the shared single-event formatter.

    Returns at least one line: either formatted events or ``empty_text``.
    """
    lines: List[str] = []
    for event in events:
        line = format_event_line(event)
        if isinstance(line, str) and line.strip():
            lines.append(line)

    return lines if lines else [empty_text]
