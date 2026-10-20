"""Validate ICU-Evo model outputs."""

from __future__ import annotations

import re

from typing import Any, Callable, Dict, List, Optional, Tuple

from model.llms import LLMClient
from utils.json_parse import parse_json_dict_or_raise


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
    attempts = max_attempts
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
