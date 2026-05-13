"""Shared utilities for Oracle experiment condition running and analysis."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from utils.status_scoring import (
    PRIMARY_STATUS_LABELS,
    STATUS_SCORE_MAP,
    nearest_primary_status,
    normalize_status_label,
    status_to_score,
)

DOMAIN_KEYS: Tuple[str, ...] = ("hemodynamics", "respiratory", "renal_metabolic", "neurology")
DEFAULT_DOMAIN_WEIGHTS: Dict[str, float] = {
    "hemodynamics": 0.25,
    "respiratory": 0.25,
    "renal_metabolic": 0.25,
    "neurology": 0.25,
}
POSITIVE_STATUS_LABELS = {"stable", "improving"}
NEGATIVE_STATUS_LABELS = {"deteriorating", "fluctuating"}

OUTCOME_MASK_TOKEN = "[OUTCOME_MASKED]"
OUTCOME_LEAK_TERMS_PATTERN = re.compile(
    r"(?i)\b(expired|deceased|dead|died|passed\s+away|death|hospice|comfort\s+measures)\b"
)
DISCHARGE_SUMMARY_SECTION_HEADERS = (
    "Discharge Disposition:",
    "Discharge Diagnosis:",
    "Discharge Condition:",
    "Discharge Instructions:",
    "Followup Instructions:",
    "Medications on Admission:",
    "Discharge Medications:",
)


@dataclass(frozen=True)
class ConditionSpec:
    """Definition of one Oracle condition setup."""

    name: str
    include_icu_outcome_in_prompt: bool
    mask_discharge_summary_outcome_terms: bool
    reverse_prompt_outcome: bool


def build_default_condition_specs() -> Dict[str, ConditionSpec]:
    """Return the default condition set requested for Oracle experiments."""
    specs = [
        ConditionSpec(
            name="full_visible",
            include_icu_outcome_in_prompt=True,
            mask_discharge_summary_outcome_terms=False,
            reverse_prompt_outcome=False,
        ),
        ConditionSpec(
            name="masked_outcome",
            include_icu_outcome_in_prompt=False,
            mask_discharge_summary_outcome_terms=True,
            reverse_prompt_outcome=False,
        ),
        ConditionSpec(
            name="reversed_outcome",
            include_icu_outcome_in_prompt=True,
            mask_discharge_summary_outcome_terms=True,
            reverse_prompt_outcome=True,
        ),
    ]
    return {spec.name: spec for spec in specs}


def reverse_prompt_outcome_flag(value: Any) -> bool:
    """Flip a survived/died style value for prompt-only outcome reversal."""
    if isinstance(value, bool):
        return not value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "survived", "alive"}:
        return False
    if text in {"false", "0", "no", "n", "died", "dead", "deceased"}:
        return True
    # Fall back to boolean semantics for unknown variants.
    return not bool(value)


def apply_prompt_outcome_mode(trajectory: Dict[str, Any], reverse_prompt_outcome: bool) -> Dict[str, Any]:
    """Create a prompt-view copy of trajectory with optional outcome reversal."""
    payload = copy.deepcopy(trajectory)
    if reverse_prompt_outcome:
        payload["survived"] = reverse_prompt_outcome_flag(trajectory.get("survived"))
    return payload


def mask_outcome_terms(text: str) -> str:
    """Mask explicit outcome words while preserving surrounding structure."""
    if not text:
        return ""
    return OUTCOME_LEAK_TERMS_PATTERN.sub(OUTCOME_MASK_TOKEN, text)


def _remove_summary_section(text: str, section_header: str) -> str:
    if not text:
        return ""

    section_start = re.escape(section_header)
    section_end_candidates = "|".join(
        re.escape(header)
        for header in DISCHARGE_SUMMARY_SECTION_HEADERS
        if header.lower() != section_header.lower()
    )
    pattern = re.compile(rf"(?is){section_start}\s*.*?(?=(?:{section_end_candidates})|$)")
    return pattern.sub("", text)


def sanitize_discharge_summary_text(text: str) -> str:
    """Mask outcome leakage in discharge summary free text."""
    if not text:
        return ""

    sanitized = _remove_summary_section(text, "Discharge Disposition:")
    sanitized = _remove_summary_section(sanitized, "Discharge Condition:")
    sanitized = mask_outcome_terms(sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
    return sanitized


def _mask_event_if_discharge_summary(event: MutableMapping[str, Any]) -> None:
    code = str(event.get("code") or "").strip()
    if code != "NOTE_DISCHARGESUMMARY":
        return

    if "code_specifics" in event and event.get("code_specifics") is not None:
        event["code_specifics"] = mask_outcome_terms(str(event.get("code_specifics") or ""))
    if "text_value" in event and event.get("text_value") is not None:
        event["text_value"] = sanitize_discharge_summary_text(str(event.get("text_value") or ""))


def mask_window_outcome_leakage(window: Dict[str, Any]) -> Dict[str, Any]:
    """Mask discharge-summary outcome leakage in a generated time window payload."""
    masked = copy.deepcopy(window)

    for events_key in (
        "history_events",
        "current_events",
        "future_events",
        "source_window_history_events",
        "oracle_context_history_events",
        "oracle_context_current_events",
        "oracle_context_future_events",
    ):
        events = masked.get(events_key)
        if not isinstance(events, list):
            continue
        for item in events:
            if not isinstance(item, MutableMapping):
                continue
            _mask_event_if_discharge_summary(item)
            if str(item.get("type") or "").strip() == "pre_icu_reports":
                content = str(item.get("content") or "")
                item["content"] = sanitize_discharge_summary_text(content)

    pre_icu_history = masked.get("pre_icu_history")
    if isinstance(pre_icu_history, MutableMapping):
        content = pre_icu_history.get("content")
        if content is not None:
            pre_icu_history["content"] = sanitize_discharge_summary_text(str(content))

    current_discharge_summary = masked.get("current_discharge_summary")
    if isinstance(current_discharge_summary, MutableMapping):
        details = current_discharge_summary.get("code_specifics")
        if details is not None:
            current_discharge_summary["code_specifics"] = mask_outcome_terms(str(details))
        text_value = current_discharge_summary.get("text_value")
        if text_value is not None:
            current_discharge_summary["text_value"] = sanitize_discharge_summary_text(str(text_value))

    return masked


def extract_patient_status_payload(oracle_output: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract normalized patient_assessment payload from one Oracle output object."""
    patient_assessment = oracle_output.get("patient_assessment")
    if isinstance(patient_assessment, dict):
        return patient_assessment
    return {}


def extract_overall_label(oracle_output: Mapping[str, Any]) -> str:
    """Extract normalized overall status label from an Oracle output object."""
    patient_assessment = extract_patient_status_payload(oracle_output)
    overall = patient_assessment.get("overall")
    if isinstance(overall, Mapping):
        label = normalize_status_label(overall.get("label"))
        if label:
            return label
    return "insufficient_data"


def extract_domain_labels(oracle_output: Mapping[str, Any]) -> Dict[str, str]:
    """Extract normalized domain-level status labels.

    The current Oracle schema does not emit domain labels; this returns
    `insufficient_data` unless optional domain labels are present.
    """
    patient_assessment = extract_patient_status_payload(oracle_output)
    domains = patient_assessment.get("domains") if isinstance(patient_assessment, Mapping) else {}
    if not isinstance(domains, Mapping):
        domains = {}

    labels: Dict[str, str] = {}
    for key in DOMAIN_KEYS:
        item = domains.get(key)
        if isinstance(item, Mapping):
            label = normalize_status_label(item.get("label"))
        else:
            label = "insufficient_data"
        if label not in STATUS_SCORE_MAP:
            label = "insufficient_data"
        labels[key] = label
    return labels


def compute_weighted_domain_score(
    domain_labels: Mapping[str, str],
    domain_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Compute a weighted average score from domain labels."""
    if domain_weights is None:
        domain_weights = DEFAULT_DOMAIN_WEIGHTS

    weight_total = 0.0
    weighted_sum = 0.0
    for key in DOMAIN_KEYS:
        weight = float(domain_weights.get(key, 0.0))
        label = normalize_status_label(domain_labels.get(key))
        weighted_sum += weight * status_to_score(label)
        weight_total += weight

    if weight_total <= 0:
        return 0.0
    return weighted_sum / weight_total


def compute_domain_consistency(
    oracle_output: Mapping[str, Any],
    domain_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Compute weighted-domain score and category consistency against overall label."""
    overall_label = extract_overall_label(oracle_output)
    overall_score = status_to_score(overall_label)
    domain_labels = extract_domain_labels(oracle_output)
    weighted_domain_score = compute_weighted_domain_score(domain_labels, domain_weights=domain_weights)
    nearest_label = nearest_primary_status(weighted_domain_score)
    is_evaluable = overall_label in set(PRIMARY_STATUS_LABELS)
    is_match = bool(is_evaluable and nearest_label == overall_label)
    return {
        "overall_label": overall_label,
        "overall_score": overall_score,
        "domain_labels": domain_labels,
        "weighted_domain_score": weighted_domain_score,
        "weighted_domain_label": nearest_label,
        "is_evaluable": is_evaluable,
        "is_match": is_match,
        "score_abs_gap": abs(weighted_domain_score - overall_score),
    }


def normalize_time_position(hours_since_admission: float, max_hours: float) -> float:
    """Normalize ICU timeline to [0, 1]."""
    if max_hours <= 0:
        return 0.0
    value = float(hours_since_admission) / float(max_hours)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def assign_normalized_time_bin(normalized_time: float, num_bins: int = 10) -> int:
    """Assign normalized trajectory time into integer bins [0, num_bins-1]."""
    if num_bins <= 1:
        return 0
    clipped = min(max(float(normalized_time), 0.0), 1.0)
    if math.isclose(clipped, 1.0):
        return num_bins - 1
    return int(clipped * num_bins)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman rho without SciPy dependency."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_rank = pd.Series(list(x), dtype="float64").rank(method="average")
    y_rank = pd.Series(list(y), dtype="float64").rank(method="average")
    rho = x_rank.corr(y_rank, method="pearson")
    if rho is None:
        return float("nan")
    return float(rho)


def auc_from_scores(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute AUROC from binary labels and continuous scores."""
    if len(y_true) != len(y_score) or not y_true:
        return float("nan")

    df = pd.DataFrame({"y_true": list(y_true), "y_score": list(y_score)})
    positives = int((df["y_true"] == 1).sum())
    negatives = int((df["y_true"] == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    df["rank"] = df["y_score"].rank(method="average")
    positive_rank_sum = float(df.loc[df["y_true"] == 1, "rank"].sum())
    auc = (positive_rank_sum - (positives * (positives + 1) / 2.0)) / float(positives * negatives)
    return float(auc)
