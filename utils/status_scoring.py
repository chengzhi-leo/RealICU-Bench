"""Canonical status label scoring utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

# Canonical mapping requested for Oracle experiments.
STATUS_SCORE_MAP: Dict[str, float] = {
    "deteriorating": -1.0,
    "fluctuating": -0.5,
    "stable": 0.5,
    "improving": 1.0,
    "insufficient_data": 0.0,
}

PRIMARY_STATUS_LABELS: Tuple[str, ...] = ("deteriorating", "fluctuating", "stable", "improving")


def normalize_status_label(value: Any) -> str:
    """Normalize a raw status label into lower snake-like text."""
    if value is None:
        return "insufficient_data"
    text = str(value).strip().lower()
    if not text:
        return "insufficient_data"
    return text


def status_to_score(value: Any, default: float | None = None) -> float:
    """Map a status label to its canonical numeric score."""
    if default is None:
        default = STATUS_SCORE_MAP["insufficient_data"]
    label = normalize_status_label(value)
    return float(STATUS_SCORE_MAP.get(label, default))


def nearest_primary_status(score: float, labels: Iterable[str] = PRIMARY_STATUS_LABELS) -> str:
    """Map a numeric score to the nearest primary status label anchor."""
    candidate_labels = list(labels)
    if not candidate_labels:
        return "insufficient_data"

    best_label = candidate_labels[0]
    best_distance = abs(float(score) - float(STATUS_SCORE_MAP.get(best_label, 0.0)))
    for label in candidate_labels[1:]:
        anchor = float(STATUS_SCORE_MAP.get(label, 0.0))
        distance = abs(float(score) - anchor)
        # Keep deterministic tie-breaking by retaining the first-seen label.
        if distance < best_distance:
            best_distance = distance
            best_label = label
    return best_label
