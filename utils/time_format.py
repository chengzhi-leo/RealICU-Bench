"""Shared timestamp formatting helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def format_timestamp_minute(value: Any) -> str:
    """Format timestamp-like input to minute precision: YYYY-MM-DD HH:MM."""
    if value is None:
        return "Unknown"

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")

    text = str(value).strip()
    if not text:
        return "Unknown"

    normalized = text
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        pass

    if len(text) >= 16:
        return text[:16]
    return text
