"""Shared helpers for parsing dict-shaped JSON from model outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def parse_json_dict_best_effort(content: Any) -> Optional[Dict[str, Any]]:
    """Parse a dict JSON payload from raw text, fenced blocks, or embedded snippets."""
    if isinstance(content, dict):
        return content
    if content is None:
        return None

    text = str(content).strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    for block in fenced_blocks:
        block_text = block.strip()
        if not block_text:
            continue
        try:
            parsed = json.loads(block_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[i:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def parse_json_dict_or_raise(content: Any, *, error_prefix: str = "Could not parse JSON from response") -> Dict[str, Any]:
    """Parse dict JSON payload and raise ValueError when parsing fails."""
    parsed = parse_json_dict_best_effort(content)
    if parsed is not None:
        return parsed

    if content is None:
        preview = ""
    else:
        preview = str(content).strip()[:200]
    raise ValueError(f"{error_prefix}: {preview}...")
