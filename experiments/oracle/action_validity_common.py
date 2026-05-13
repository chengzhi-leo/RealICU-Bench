"""Shared utilities for Oracle Q3/Q4 action-validity experiments."""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

ACTIONABLE_EVENT_CODES: Tuple[str, ...] = (
    "DRUG_START",
    "DRUG_PRESCRIPTION",
    "PROCEDURE",
    "BODY_INPUT",
    "TRANSFER",
)

ACTION_SCORE_MAP: Dict[str, float] = {
    "potentially_harmful": -1.0,
    "insufficient_data": 0.0,
    "acceptable": 0.5,
    "best_practice": 1.0,
}

NEGATIVE_ACTION_LABELS: Set[str] = {"potentially_harmful"}

STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
    "during",
    "over",
    "patient",
    "window",
    "current",
    "action",
    "clinical",
    "team",
}

TOKEN_SYNONYMS: Dict[str, str] = {
    "noradrenaline": "norepinephrine",
    "levophed": "norepinephrine",
    "pressor": "vasopressor",
    "pressors": "vasopressor",
    "abx": "antibiotics",
    "antimicrobial": "antibiotics",
    "fio2": "oxygen",
    "o2": "oxygen",
    "spo2": "oxygenation",
    "desat": "desaturation",
    "desats": "desaturation",
    "uop": "urine",
    "dialysis": "rrt",
}

DEFAULT_JACCARD_THRESHOLD = 0.30
DEFAULT_MIN_SHARED_TOKENS = 2


def normalize_action_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "best": "best_practice",
        "good": "acceptable",
        "harmful": "potentially_harmful",
        "potential_harm": "potentially_harmful",
        "insufficient": "insufficient_data",
    }
    return aliases.get(text, text)


def extract_action_label(action_eval: Any) -> str:
    if not isinstance(action_eval, Mapping):
        return ""

    for key in ("label", "overall", "contextual_appropriateness", "guideline_adherence", "quality_rating"):
        candidate = action_eval.get(key)
        if isinstance(candidate, Mapping):
            for field in ("label", "status", "value"):
                label = normalize_action_label(candidate.get(field))
                if label:
                    return label
        else:
            label = normalize_action_label(candidate)
            if label:
                return label
    return ""


def action_label_to_score(label: Any) -> float:
    normalized = normalize_action_label(label)
    if normalized in ACTION_SCORE_MAP:
        return float(ACTION_SCORE_MAP[normalized])
    return float("nan")


def _event_code(event: Mapping[str, Any]) -> str:
    return str(event.get("code") or "").strip().upper()


def extract_actionable_events(
    events: Sequence[Mapping[str, Any]],
    whitelist: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    if whitelist is None:
        whitelist_set = set(ACTIONABLE_EVENT_CODES)
    else:
        whitelist_set = {str(code).strip().upper() for code in whitelist if str(code).strip()}

    extracted: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if _event_code(event) not in whitelist_set:
            continue
        extracted.append(dict(event))
    return extracted


def count_actionable_events(events: Sequence[Mapping[str, Any]], whitelist: Optional[Iterable[str]] = None) -> int:
    return len(extract_actionable_events(events, whitelist=whitelist))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _event_text(event: Mapping[str, Any]) -> str:
    parts = [
        str(event.get("code") or ""),
        str(event.get("code_specifics") or ""),
        str(event.get("text_value") or ""),
    ]
    return " ".join(parts).lower()


def detect_wrong_action_signals(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    signals = {
        "hypotension": False,
        "hypoxemia": False,
        "renal_failure": False,
        "trigger": "fallback",
    }

    for event in events:
        if not isinstance(event, Mapping):
            continue
        text = _event_text(event)
        numeric = _safe_float(event.get("numeric_value"))

        if (
            ("map" in text or "mean arterial pressure" in text)
            and numeric is not None
            and numeric < 65.0
        ) or ("hypotension" in text):
            signals["hypotension"] = True

        if (
            ("spo2" in text or "oxygen saturation" in text or "sat" in text)
            and numeric is not None
            and numeric < 90.0
        ) or any(keyword in text for keyword in ("hypoxemia", "desaturation", "desat")):
            signals["hypoxemia"] = True

        if (
            "creatinine" in text and numeric is not None and numeric >= 2.0
        ) or any(keyword in text for keyword in ("oliguria", "anuria")):
            signals["renal_failure"] = True

    if signals["hypotension"]:
        signals["trigger"] = "hypotension"
    elif signals["hypoxemia"]:
        signals["trigger"] = "hypoxemia"
    elif signals["renal_failure"]:
        signals["trigger"] = "renal_failure"

    return signals


def select_wrong_action_template(events: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    signals = detect_wrong_action_signals(events)
    trigger = str(signals.get("trigger") or "fallback")

    templates = {
        "hypotension": {
            "template_id": "wrong_pressor_deescalation",
            "action_text": "Decrease norepinephrine despite persistent hypotension (MAP < 65).",
            "trigger": "hypotension",
        },
        "hypoxemia": {
            "template_id": "wrong_oxygen_deescalation",
            "action_text": "Reduce FiO2 to room air despite ongoing hypoxemia (SpO2 < 90).",
            "trigger": "hypoxemia",
        },
        "renal_failure": {
            "template_id": "wrong_nephrotoxic_exposure",
            "action_text": "Administer nephrotoxic contrast despite rising creatinine and oliguria/anuria.",
            "trigger": "renal_failure",
        },
        "fallback": {
            "template_id": "wrong_septic_shock_contradiction",
            "action_text": "Stop antibiotics and vasopressor support despite ongoing septic-shock concern.",
            "trigger": "fallback",
        },
    }

    return dict(templates.get(trigger, templates["fallback"]))


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None

    for parser in (datetime.fromisoformat,):
        try:
            return parser(text)
        except ValueError:
            pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def inject_counterfactual_current_event(
    window: Mapping[str, Any],
    *,
    marker_token: str,
    wrong_action_text: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Append one synthetic wrong action event to current events.

    Returns:
        mutated_window, injected_event, expected_action_id
    """
    payload = copy.deepcopy(dict(window))
    current_events = payload.get("current_events")
    if not isinstance(current_events, list):
        current_events = []
    else:
        current_events = list(current_events)

    end_time = _coerce_datetime(payload.get("current_window_end"))
    start_time = _coerce_datetime(payload.get("current_window_start"))

    if end_time is not None:
        injected_time = end_time - timedelta(seconds=1)
    elif start_time is not None:
        injected_time = start_time
    else:
        injected_time = datetime.utcnow()

    injected_event = {
        "time": injected_time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": "PROCEDURE",
        "code_specifics": f"COUNTERFACTUAL_ACTION {marker_token}",
        "text_value": f"{marker_token} {wrong_action_text}",
        "numeric_value": None,
    }

    existing_event_ids: List[int] = []
    for event in current_events:
        if not isinstance(event, Mapping):
            continue
        for key in ("event_id", "event_idx"):
            raw_id = event.get(key)
            try:
                parsed_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            existing_event_ids.append(parsed_id)
            break

    next_event_id = (max(existing_event_ids) + 1) if existing_event_ids else (len(current_events) + 1)
    injected_event["event_id"] = int(next_event_id)

    current_events.append(injected_event)
    payload["current_events"] = current_events
    payload["num_current_events"] = len(current_events)

    expected_action_id = str(next_event_id)
    return payload, injected_event, expected_action_id


def _extract_action_id_number(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None

    match = re.search(r"\bevent_id\s*[:=]\s*(\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    bracket_match = re.fullmatch(r"\[(\d+)\]", text)
    if bracket_match:
        return int(bracket_match.group(1))

    if re.fullmatch(r"\d+", text):
        return int(text)
    return None


def identify_action_evaluation(
    action_evaluations: Any,
    *,
    expected_action_id: Optional[str],
    marker_token: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(action_evaluations, list):
        return None

    normalized_expected = str(expected_action_id or "").strip().lower()
    expected_numeric = _extract_action_id_number(expected_action_id)
    normalized_marker = str(marker_token or "").strip().lower()

    if normalized_expected:
        for item in action_evaluations:
            if not isinstance(item, Mapping):
                continue
            action_id_raw = item.get("action_id")
            action_id = str(action_id_raw or "").strip().lower()
            if action_id == normalized_expected:
                return dict(item)
            action_numeric = _extract_action_id_number(action_id_raw)
            if expected_numeric is not None and action_numeric == expected_numeric:
                return dict(item)

    if normalized_marker:
        for item in action_evaluations:
            if not isinstance(item, Mapping):
                continue
            try:
                blob = json.dumps(item, ensure_ascii=False).lower()
            except TypeError:
                blob = str(item).lower()
            if normalized_marker in blob:
                return dict(item)

    return None


def recommendation_to_text(recommendation: Any) -> str:
    if recommendation is None:
        return ""
    if isinstance(recommendation, str):
        return recommendation.strip()
    if not isinstance(recommendation, Mapping):
        return ""

    action = str(
        recommendation.get("action_name")
        or recommendation.get("action")
        or recommendation.get("contraindicated_action")
        or ""
    ).strip()
    desc = str(recommendation.get("action_description") or "").strip()
    if not desc:
        desc = str(recommendation.get("reason") or recommendation.get("rationale") or "").strip()
    if action and desc and action.lower() not in desc.lower():
        return f"{action}. {desc}".strip()
    return (desc or action).strip()


def event_to_action_text(event: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in ("code_specifics", "text_value", "code"):
        value = event.get(key)
        text = str(value).strip() if value is not None else ""
        if text:
            parts.append(text)
    numeric = _safe_float(event.get("numeric_value"))
    if numeric is not None:
        parts.append(f"value {numeric:.2f}")
    return " ".join(parts).strip()


def build_doctor_action_texts_from_events(events: Sequence[Mapping[str, Any]]) -> List[str]:
    action_events = extract_actionable_events(events)
    texts = [event_to_action_text(event) for event in action_events]
    return [text for text in texts if text]


def build_recommendation_texts(recommendations: Any) -> List[str]:
    if not isinstance(recommendations, list):
        return []
    texts = [recommendation_to_text(item) for item in recommendations]
    return [text for text in texts if text]


def normalize_text_tokens(text: Any, synonyms: Optional[Mapping[str, str]] = None) -> Set[str]:
    if text is None:
        return set()
    if synonyms is None:
        synonyms = TOKEN_SYNONYMS

    raw = str(text).lower()
    raw = re.sub(r"[^a-z0-9_]+", " ", raw)
    tokens = []
    for token in raw.split():
        canonical = synonyms.get(token, token)
        if len(canonical) <= 1:
            continue
        if canonical in STOPWORDS:
            continue
        tokens.append(canonical)
    return set(tokens)


def pair_similarity(recommendation_text: Any, doctor_action_text: Any) -> Tuple[float, int]:
    rec_tokens = normalize_text_tokens(recommendation_text)
    action_tokens = normalize_text_tokens(doctor_action_text)
    if not rec_tokens or not action_tokens:
        return 0.0, 0

    intersection = rec_tokens.intersection(action_tokens)
    union = rec_tokens.union(action_tokens)
    if not union:
        return 0.0, 0

    return float(len(intersection) / len(union)), int(len(intersection))


def match_recommendations_to_actions(
    recommendation_texts: Sequence[str],
    doctor_action_texts: Sequence[str],
    *,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    min_shared_tokens: int = DEFAULT_MIN_SHARED_TOKENS,
) -> Dict[str, Any]:
    candidates: List[Tuple[float, int, int, int]] = []

    for rec_idx, rec_text in enumerate(recommendation_texts):
        for action_idx, action_text in enumerate(doctor_action_texts):
            jaccard, shared = pair_similarity(rec_text, action_text)
            if shared < int(min_shared_tokens):
                continue
            if jaccard < float(jaccard_threshold):
                continue
            candidates.append((jaccard, shared, rec_idx, action_idx))

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

    used_recommendations: Set[int] = set()
    used_actions: Set[int] = set()
    matches: List[Dict[str, Any]] = []

    for jaccard, shared, rec_idx, action_idx in candidates:
        if rec_idx in used_recommendations or action_idx in used_actions:
            continue
        used_recommendations.add(rec_idx)
        used_actions.add(action_idx)
        matches.append(
            {
                "recommendation_index": rec_idx,
                "doctor_action_index": action_idx,
                "jaccard": float(jaccard),
                "shared_tokens": int(shared),
                "recommendation_text": recommendation_texts[rec_idx],
                "doctor_action_text": doctor_action_texts[action_idx],
            }
        )

    unmatched_recommendations = [
        idx for idx in range(len(recommendation_texts)) if idx not in used_recommendations
    ]
    unmatched_actions = [idx for idx in range(len(doctor_action_texts)) if idx not in used_actions]

    return {
        "matches": matches,
        "unmatched_recommendation_indices": unmatched_recommendations,
        "unmatched_doctor_action_indices": unmatched_actions,
        "num_matches": len(matches),
        "num_recommendations": len(recommendation_texts),
        "num_doctor_actions": len(doctor_action_texts),
    }


def compute_precision_recall_f1(
    *,
    num_matches: int,
    num_recommendations: int,
    num_doctor_actions: int,
) -> Tuple[float, float, float]:
    precision = float("nan")
    recall = float("nan")

    if num_recommendations > 0:
        precision = float(num_matches) / float(num_recommendations)
    if num_doctor_actions > 0:
        recall = float(num_matches) / float(num_doctor_actions)

    if math.isnan(precision) or math.isnan(recall):
        return precision, recall, float("nan")

    if precision + recall == 0:
        return precision, recall, 0.0

    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, float(f1)


def is_finite_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(parsed) and math.isfinite(parsed)
