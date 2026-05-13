"""LLM provider error classification helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

_CONTEXT_LENGTH_ERROR_CODES = {
    "context_length_exceeded",
    "context_window_exceeded",
    "prompt_too_long",
    "input_too_long",
    "too_many_tokens",
    "request_too_large",
}

_CONTEXT_LENGTH_ERROR_MARKERS = (
    "context_length_exceeded",
    "context window exceeded",
    "maximum context length",
    "context length",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "input too long",
    "token limit",
    "request too large",
    "status code: 413",
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _extract_mapping_error_code(payload: Mapping[str, Any]) -> str:
    direct_code = _normalize_text(payload.get("code"))
    if direct_code:
        return direct_code
    nested_error = payload.get("error")
    if isinstance(nested_error, Mapping):
        nested_code = _normalize_text(nested_error.get("code"))
        if nested_code:
            return nested_code
        nested_type = _normalize_text(nested_error.get("type"))
        if nested_type:
            return nested_type
    return ""


def _extract_error_code(error: BaseException) -> str:
    direct_code = _normalize_text(getattr(error, "code", None))
    if direct_code:
        return direct_code

    error_type = _normalize_text(getattr(error, "type", None))
    if error_type:
        return error_type

    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        body_code = _extract_mapping_error_code(body)
        if body_code:
            return body_code

    response = getattr(error, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code == 413:
            return "request_too_large"
        response_json = getattr(response, "json", None)
        if callable(response_json):
            try:
                payload = response_json()
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                response_code = _extract_mapping_error_code(payload)
                if response_code:
                    return response_code

    return ""


def is_context_length_exceeded_error(error: BaseException, status_code: Optional[int] = None) -> bool:
    if isinstance(status_code, int) and status_code == 413:
        return True

    code = _extract_error_code(error)
    if code in _CONTEXT_LENGTH_ERROR_CODES:
        return True

    message = " ".join(
        text
        for text in (
            _normalize_text(getattr(error, "message", None)),
            _normalize_text(error),
            _normalize_text(repr(error)),
        )
        if text
    )
    if not message:
        return False

    return any(marker in message for marker in _CONTEXT_LENGTH_ERROR_MARKERS)
