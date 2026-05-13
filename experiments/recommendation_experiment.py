"""Recommendation prediction from precomputed ICU-Evo memory runs and event baselines."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.icu_evo_agent import ICUEvoAgent
from config.config import get_config
from experiments.create_memory import (
    collect_windowed_snapshots,
    extract_snapshot_window_features,
    filter_memory_patient_records_by_stay_ids,
    infer_snapshot_window_index,
    load_memory_patient_records,
    load_memory_run_config,
    load_patient_memory_payload,
    render_snapshot_to_text,
    resolve_memory_run_dir,
)
from experiments.oracle.action_validity_common import ACTIONABLE_EVENT_CODES, normalize_action_label
from experiments.window_rag import (
    IncrementalWindowRAGRetriever,
    RAGRetrievedWindow,
    SentenceTransformerWindowEncoder,
    build_rag_window_record,
    render_rag_event_context,
)
from prompts.predictor_prompts import get_recommendation_action_prompt
from utils.json_parse import parse_json_dict_best_effort
from utils.llm_errors import is_context_length_exceeded_error
from utils.llm_log_viewer import save_llm_calls_html

RUN_CONFIG_FILENAME = "run_config.json"
AGGREGATE_FILENAME = "aggregate_results.json"
CONTEXT_MODE_MEMORY = "memory"
CONTEXT_MODE_FULL_HISTORY_EVENTS = "full_history_events"
CONTEXT_MODE_LOCAL_EVENTS_ONLY = "local_events_only"
CONTEXT_MODE_RAG_HISTORY_EVENTS = "rag_history_events"
CONTEXT_MODE_ALL = "all"
SUPPORTED_CONTEXT_MODES = (
    CONTEXT_MODE_MEMORY,
    CONTEXT_MODE_FULL_HISTORY_EVENTS,
    CONTEXT_MODE_LOCAL_EVENTS_ONLY,
    CONTEXT_MODE_RAG_HISTORY_EVENTS,
)
SUPPORTED_CONTEXT_MODE_CHOICES = SUPPORTED_CONTEXT_MODES + (CONTEXT_MODE_ALL,)
ORACLE_POSITIVE_ACTION_LABELS = frozenset({"best_practice", "acceptable"})
EVENT_LINE_CODE_PATTERN = re.compile(
    r"^\s*(?:\[[^\]]+\]\s+)?(?:\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s+)?([A-Z][A-Z0-9_]+)\b"
)
ORACLE_INTERVAL_DECIMALS = 6
ORACLE_TIME_MATCH_EPSILON = 1e-9
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
RAG_TOP_K_WINDOWS = 3
RAG_EMBEDDING_BATCH_SIZE = 64
RAG_RETRIEVAL_SCOPE = "past_only"


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


def _sum_numeric_stats(items: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _filter_completed_patient_records(
    records: Sequence[Dict[str, Any]],
    results_dir: Path,
) -> Tuple[List[Dict[str, Any]], int]:
    pending_records: List[Dict[str, Any]] = []
    skipped = 0
    for record in records:
        subject_id = int(record["subject_id"])
        icu_stay_id = int(record["icu_stay_id"])
        patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
        if (patient_dir / "prediction.json").exists() and (patient_dir / "recommendation_predictions.json").exists():
            skipped += 1
            continue
        pending_records.append(record)
    return pending_records, skipped


def _normalize_confidence_label(value: Any) -> str:
    if value is None:
        return "Unknown"
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "low": "Low",
        "moderate": "Moderate",
        "medium": "Moderate",
        "high": "High",
    }
    return aliases.get(normalized, "Unknown")


def _coerce_recommended_actions(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _coerce_red_flags(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _rank_sort_key(action: Mapping[str, Any], index: int) -> Tuple[int, int]:
    rank_raw = action.get("rank")
    try:
        rank = int(rank_raw)
    except (TypeError, ValueError):
        rank = sys.maxsize
    return rank, index


def _extract_recommendation_summary_fields(
    parsed_prediction: Mapping[str, Any],
    recommended_actions: Sequence[Mapping[str, Any]],
) -> Tuple[str, str]:
    summary_confidence = _normalize_confidence_label(parsed_prediction.get("confidence"))
    summary_rationale = str(parsed_prediction.get("rationale") or "").strip()

    if not recommended_actions:
        return summary_confidence, summary_rationale

    ranked_actions = sorted(
        enumerate(recommended_actions),
        key=lambda item: _rank_sort_key(item[1], item[0]),
    )
    primary_action = ranked_actions[0][1]

    if summary_confidence == "Unknown":
        summary_confidence = _normalize_confidence_label(primary_action.get("confidence"))
    if not summary_rationale:
        summary_rationale = str(primary_action.get("rationale") or "").strip()

    return summary_confidence, summary_rationale


def _extract_current_window(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    working_memory = snapshot.get("working_memory")
    if not isinstance(working_memory, list) or not working_memory:
        raise ValueError("Snapshot missing non-empty working_memory")
    current_window = working_memory[-1]
    if not isinstance(current_window, Mapping):
        raise ValueError("Snapshot current working_memory entry must be a mapping")
    return current_window


def _event_code(event_line: str) -> str:
    match = EVENT_LINE_CODE_PATTERN.match(event_line)
    if match is None:
        return ""
    return str(match.group(1)).strip().upper()


def _filter_non_action_events(
    events: Sequence[str],
    actionable_codes: Sequence[str],
) -> Tuple[List[str], int]:
    action_code_set = {str(code).strip().upper() for code in actionable_codes if str(code).strip()}
    kept: List[str] = []
    masked_count = 0
    for event in events:
        event_line = str(event).strip()
        if not event_line:
            continue
        if _event_code(event_line) in action_code_set:
            masked_count += 1
            continue
        kept.append(event_line)
    return kept, masked_count


def _window_events(window: Mapping[str, Any]) -> List[str]:
    events = window.get("events")
    if not isinstance(events, list):
        return []
    return [str(event).strip() for event in events if str(event).strip()]


def _render_flat_raw_events(events: Sequence[str]) -> str:
    event_rows = [str(event).strip() for event in events if str(event).strip()]
    return "\n".join(event_rows) if event_rows else "(No events)"


def build_full_history_event_context(
    snapshot_by_window: Mapping[int, Mapping[str, Any]],
    current_window_index: int,
    current_snapshot: Mapping[str, Any],
    actionable_codes: Sequence[str],
) -> Tuple[str, int, int]:
    current_window = _extract_current_window(current_snapshot)
    current_events = _window_events(current_window)
    masked_events, masked_action_count = _filter_non_action_events(current_events, actionable_codes=actionable_codes)

    candidate_windows = {int(index) for index in snapshot_by_window.keys() if int(index) <= int(current_window_index)}
    candidate_windows.add(int(current_window_index))
    ordered_windows = sorted(candidate_windows)
    merged_events: List[Dict[str, Any]] = []
    for window_index in ordered_windows:
        if int(window_index) == int(current_window_index):
            merged_events.extend(masked_events)
        else:
            snapshot = snapshot_by_window.get(int(window_index))
            if not isinstance(snapshot, Mapping):
                continue
            window = _extract_current_window(snapshot)
            merged_events.extend(_window_events(window))

    context = _render_flat_raw_events(merged_events)
    return context, int(masked_action_count), int(len(masked_events))


def build_local_events_only_context(
    current_snapshot: Mapping[str, Any],
    actionable_codes: Sequence[str],
) -> Tuple[str, int, int]:
    current_window = _extract_current_window(current_snapshot)
    current_events = _window_events(current_window)
    masked_events, masked_action_count = _filter_non_action_events(current_events, actionable_codes=actionable_codes)
    context = _render_flat_raw_events(masked_events)
    return context, int(masked_action_count), int(len(masked_events))


def _build_rag_record_from_snapshot(
    *,
    window_index: int,
    start_hour: float,
    end_hour: float,
    snapshot: Mapping[str, Any],
):
    current_window = _extract_current_window(snapshot)
    return build_rag_window_record(
        window_index=int(window_index),
        start_hour=float(start_hour),
        end_hour=float(end_hour),
        event_lines=_window_events(current_window),
    )


def _build_masked_current_rag_record(
    *,
    window_index: int,
    start_hour: float,
    end_hour: float,
    snapshot: Mapping[str, Any],
    actionable_codes: Sequence[str],
):
    current_window = _extract_current_window(snapshot)
    current_events = _window_events(current_window)
    masked_events, masked_action_count = _filter_non_action_events(current_events, actionable_codes=actionable_codes)
    record = build_rag_window_record(
        window_index=int(window_index),
        start_hour=float(start_hour),
        end_hour=float(end_hour),
        event_lines=masked_events,
    )
    return record, int(masked_action_count), int(len(masked_events))


def _retrieved_windows_payload(items: Sequence[RAGRetrievedWindow]) -> List[Dict[str, Any]]:
    return [dict(item.to_payload()) for item in items]


def _normalize_context_mode(context_mode: str) -> str:
    normalized = str(context_mode).strip()
    if normalized not in SUPPORTED_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported context_mode={context_mode}. " f"Supported modes: {', '.join(SUPPORTED_CONTEXT_MODES)}"
        )
    return normalized


def _results_dir_name_for_context_mode(context_mode: str) -> str:
    if context_mode == CONTEXT_MODE_MEMORY:
        return "recommendation_experiment/memory"
    return f"recommendation_experiment/{context_mode}"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _interval_key(start_hour: float, end_hour: float) -> Tuple[float, float]:
    return (round(float(start_hour), ORACLE_INTERVAL_DECIMALS), round(float(end_hour), ORACLE_INTERVAL_DECIMALS))


def _interval_marker_id(interval_key: Tuple[float, float]) -> str:
    return f"{interval_key[0]:.{ORACLE_INTERVAL_DECIMALS}f}|{interval_key[1]:.{ORACLE_INTERVAL_DECIMALS}f}"


def _safe_optional_int(value: Any, *, field_name: str, source_path: Path) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}={value!r} in {source_path}") from exc


def _snapshot_interval(snapshot: Mapping[str, Any], *, source_label: str) -> Tuple[float, float]:
    start_raw = snapshot.get("last_processed_start_hour")
    end_raw = snapshot.get("last_processed_end_hour")
    try:
        start_hour = float(start_raw)
        end_hour = float(end_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing/invalid snapshot hour bounds in {source_label}: "
            f"last_processed_start_hour={start_raw!r}, last_processed_end_hour={end_raw!r}"
        ) from exc
    if end_hour <= start_hour:
        raise ValueError(f"Invalid snapshot hour bounds in {source_label}: start={start_hour}, end={end_hour}")
    return start_hour, end_hour


def _oracle_interval_from_row(row: Mapping[str, Any], oracle_prediction_path: Path) -> Tuple[float, float]:
    metadata = row.get("window_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Missing window_metadata in row in {oracle_prediction_path}")
    start_raw = metadata.get("hours_since_admission")
    duration_raw = metadata.get("current_window_hours")
    try:
        start_hour = float(start_raw)
        duration_hours = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid oracle window hours in {oracle_prediction_path}: "
            f"hours_since_admission={start_raw!r}, current_window_hours={duration_raw!r}"
        ) from exc
    if duration_hours <= 0:
        raise ValueError(
            f"Oracle current_window_hours must be > 0 in {oracle_prediction_path}, got {duration_hours!r}"
        )
    return start_hour, start_hour + duration_hours


def _build_oracle_marker(
    *,
    row: Mapping[str, Any],
    interval_key: Tuple[float, float],
    oracle_prediction_path: Path,
) -> Dict[str, Any]:
    metadata = row.get("window_metadata")
    metadata_payload = metadata if isinstance(metadata, Mapping) else {}
    marker = {
        "marker_id": _interval_marker_id(interval_key),
        "interval_start_hour": float(interval_key[0]),
        "interval_end_hour": float(interval_key[1]),
        "window_index": _safe_optional_int(
            row.get("window_index"),
            field_name="window_index",
            source_path=oracle_prediction_path,
        ),
        "source_window_index": _safe_optional_int(
            row.get("source_window_index", metadata_payload.get("source_window_index")),
            field_name="source_window_index",
            source_path=oracle_prediction_path,
        ),
        "stride_source_window_index": _safe_optional_int(
            row.get("stride_source_window_index", metadata_payload.get("stride_source_window_index")),
            field_name="stride_source_window_index",
            source_path=oracle_prediction_path,
        ),
    }
    return marker


def _load_oracle_targets_by_interval(
    oracle_prediction_path: Path,
) -> Tuple[
    Dict[Tuple[float, float], List[Dict[str, Any]]],
    Set[Tuple[float, float]],
    Dict[Tuple[float, float], Dict[str, Any]],
    List[Tuple[float, float]],
]:
    payload = _read_json(oracle_prediction_path)
    window_outputs = payload.get("window_outputs")
    if not isinstance(window_outputs, list):
        raise ValueError(f"Invalid window_outputs list in {oracle_prediction_path}")

    actions_by_interval: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    labeled_intervals: Set[Tuple[float, float]] = set()
    markers_by_interval: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for row in window_outputs:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid row in window_outputs in {oracle_prediction_path}")
        interval_start, interval_end = _oracle_interval_from_row(
            row=row, oracle_prediction_path=oracle_prediction_path
        )
        interval_key = _interval_key(interval_start, interval_end)
        if interval_key in markers_by_interval:
            raise ValueError(
                f"Duplicate oracle interval marker for interval={interval_key} in {oracle_prediction_path}"
            )
        marker = _build_oracle_marker(
            row=row, interval_key=interval_key, oracle_prediction_path=oracle_prediction_path
        )
        markers_by_interval[interval_key] = marker

        oracle_output = row.get("oracle_output")
        if not isinstance(oracle_output, Mapping):
            continue
        action_review = oracle_output.get("action_review")
        if not isinstance(action_review, Mapping):
            continue
        evaluations = action_review.get("evaluations")
        if evaluations is None:
            continue
        if not isinstance(evaluations, list):
            raise ValueError(
                f"Invalid action_review.evaluations for interval={interval_key} in {oracle_prediction_path}"
            )
        if evaluations:
            labeled_intervals.add(interval_key)

        selected_actions: List[Dict[str, Any]] = []
        for evaluation in evaluations:
            if not isinstance(evaluation, Mapping):
                raise ValueError(f"Invalid evaluation row for interval={interval_key} in {oracle_prediction_path}")
            label = normalize_action_label(evaluation.get("label"))
            if label not in ORACLE_POSITIVE_ACTION_LABELS:
                continue
            action_name = str(evaluation.get("action_name") or "").strip()
            if not action_name:
                raise ValueError(
                    f"Missing action_name for interval={interval_key} with label={label} in {oracle_prediction_path}"
                )
            rationale = str(evaluation.get("rationale") or evaluation.get("reason") or "").strip()
            gt_text = f"{action_name}. {rationale}".strip() if rationale else action_name
            selected_actions.append(
                {
                    "action_name": action_name,
                    "action_description": rationale,
                    "gt_action_text": gt_text,
                    "oracle_label": label,
                    "oracle_action_id": str(evaluation.get("action_id") or "").strip(),
                    "oracle_interval_marker_id": str(marker["marker_id"]),
                    "oracle_interval_start_hour": float(marker["interval_start_hour"]),
                    "oracle_interval_end_hour": float(marker["interval_end_hour"]),
                    "oracle_source_window_index": marker.get("source_window_index"),
                    "oracle_stride_source_window_index": marker.get("stride_source_window_index"),
                    "oracle_window_index": marker.get("window_index"),
                }
            )

        if not selected_actions:
            continue
        if interval_key in actions_by_interval:
            raise ValueError(
                f"Duplicate oracle action reviews for interval={interval_key} in {oracle_prediction_path}"
            )
        actions_by_interval[interval_key] = selected_actions

    ordered_intervals = sorted(markers_by_interval.keys(), key=lambda item: (float(item[0]), float(item[1])))
    return actions_by_interval, labeled_intervals, markers_by_interval, ordered_intervals


def _select_records_with_stride(records: List[Dict[str, Any]], stride: int) -> List[Dict[str, Any]]:
    if not records:
        return []
    normalized_stride = max(1, int(stride))
    selected = [record for idx, record in enumerate(records) if idx % normalized_stride == 0]
    if not selected or selected[-1] is not records[-1]:
        selected.append(records[-1])
    return selected


def _marker_best_window_index(marker: Mapping[str, Any]) -> Optional[int]:
    for key in ("stride_source_window_index", "source_window_index", "window_index"):
        value = marker.get(key)
        if value is None:
            continue
        return int(value)
    return None


def build_masked_recommendation_snapshot(
    previous_snapshot: Mapping[str, Any],
    current_snapshot: Mapping[str, Any],
    actionable_codes: Sequence[str],
) -> Tuple[Dict[str, Any], int, int]:
    previous_working_memory = previous_snapshot.get("working_memory")
    if not isinstance(previous_working_memory, list):
        raise ValueError("Previous snapshot missing working_memory list")

    current_window = _extract_current_window(current_snapshot)
    current_events = current_window.get("events")
    if not isinstance(current_events, list):
        current_events = []

    masked_events, masked_action_count = _filter_non_action_events(
        [str(event).strip() for event in current_events if str(event).strip()],
        actionable_codes=actionable_codes,
    )

    context_snapshot = copy.deepcopy(dict(previous_snapshot))
    masked_window = copy.deepcopy(dict(current_window))
    masked_window["events"] = masked_events

    merged_working_memory = copy.deepcopy(previous_working_memory)
    merged_working_memory.append(masked_window)
    context_snapshot["working_memory"] = merged_working_memory
    context_snapshot["last_processed_window_index"] = current_window.get("window_id")
    context_snapshot["last_processed_start_hour"] = current_window.get("start_hour")
    context_snapshot["last_processed_end_hour"] = current_window.get("end_hour")

    return context_snapshot, masked_action_count, len(masked_events)


def collect_union_ground_truth_action_events(
    snapshot_by_window: Mapping[int, Mapping[str, Any]],
    current_window_index: int,
    future_window_count: int,
    actionable_codes: Sequence[str],
) -> Tuple[List[str], List[int]]:
    if future_window_count < 0:
        raise ValueError(f"future_window_count must be >= 0, got {future_window_count}")

    action_code_set = {str(code).strip().upper() for code in actionable_codes if str(code).strip()}
    events: List[str] = []
    included_windows: List[int] = []
    for offset in range(0, int(future_window_count) + 1):
        window_index = int(current_window_index) + int(offset)
        snapshot = snapshot_by_window.get(window_index)
        if not isinstance(snapshot, Mapping):
            continue
        current_window = _extract_current_window(snapshot)
        action_events = [event for event in _window_events(current_window) if _event_code(event) in action_code_set]
        if not action_events:
            continue
        included_windows.append(window_index)
        events.extend(action_events)
    return events, included_windows


def collect_union_ground_truth_actions_by_time(
    *,
    oracle_actions_by_interval: Mapping[Tuple[float, float], Sequence[Mapping[str, Any]]],
    oracle_markers_by_interval: Mapping[Tuple[float, float], Mapping[str, Any]],
    ordered_oracle_intervals: Sequence[Tuple[float, float]],
    current_window_start_hour: float,
    prediction_horizon_hours: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        start_hour = float(current_window_start_hour)
        horizon_hours = float(prediction_horizon_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "current_window_start_hour and prediction_horizon_hours must be numeric "
            f"(got {current_window_start_hour!r}, {prediction_horizon_hours!r})"
        ) from exc
    if horizon_hours <= 0:
        raise ValueError(f"prediction_horizon_hours must be > 0, got {horizon_hours}")
    horizon_end = start_hour + horizon_hours
    events: List[Dict[str, Any]] = []
    included_markers: List[Dict[str, Any]] = []
    for interval in ordered_oracle_intervals:
        interval_start = float(interval[0])
        if interval_start < start_hour - ORACLE_TIME_MATCH_EPSILON:
            continue
        if interval_start > horizon_end + ORACLE_TIME_MATCH_EPSILON:
            break
        window_events = oracle_actions_by_interval.get(interval)
        if not isinstance(window_events, Sequence) or not window_events:
            continue
        marker = oracle_markers_by_interval.get(interval)
        if marker is None:
            raise ValueError(f"Missing oracle marker for interval={interval}")
        included_markers.append(dict(marker))
        for event in window_events:
            if not isinstance(event, Mapping):
                continue
            events.append(dict(event))
    return events, included_markers


def infer_window_duration_hours(snapshot: Mapping[str, Any]) -> float:
    current_window = _extract_current_window(snapshot)
    start_hour_raw = current_window.get("start_hour")
    end_hour_raw = current_window.get("end_hour")

    try:
        start_hour = float(start_hour_raw)
        end_hour = float(end_hour_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing/invalid window duration fields: start_hour={start_hour_raw}, end_hour={end_hour_raw}"
        ) from exc

    duration_hours = end_hour - start_hour
    if not math.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError(f"Invalid non-positive window duration: {duration_hours}")

    return duration_hours


def compute_future_window_count_for_horizon(
    prediction_horizon_hours: float,
    window_duration_hours: float,
) -> int:
    try:
        horizon = float(prediction_horizon_hours)
        duration = float(window_duration_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction_horizon_hours and window_duration_hours must be numeric") from exc

    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError(f"prediction_horizon_hours must be > 0, got {horizon}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"window_duration_hours must be > 0, got {duration}")

    return int(math.ceil(horizon / duration))


def process_single_patient(
    patient_record: Dict[str, Any],
    agent: ICUEvoAgent,
    memory_run_config: Mapping[str, Any],
    snapshot_stride: int,
    top_k_actions: int,
    prediction_horizon_hours: float,
    context_mode: str,
    oracle_results_dir: Path,
    memory_run_dir: Path,
    results_dir: Path,
    patient_idx: int,
    total_patients: int,
    enable_logging: bool,
    actionable_codes: Sequence[str],
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    subject_id = int(patient_record["subject_id"])
    icu_stay_id = int(patient_record["icu_stay_id"])
    actual_outcome = str(patient_record.get("actual_outcome") or "unknown")
    source_patient_dir = Path(str(patient_record["patient_dir"]))

    if verbose:
        print(f"\n[Patient {patient_idx}/{total_patients}] Subject: {subject_id}, ICU Stay: {icu_stay_id}")
        print(f"   Source memory: {source_patient_dir}")
        print(f"   Context mode: {context_mode}")

    try:
        payload = load_patient_memory_payload(source_patient_dir, run_config=memory_run_config)
        memory_snapshots = payload.get("memory_snapshots", [])
        if not isinstance(memory_snapshots, list):
            memory_snapshots = []
        final_memory = payload.get("final_memory", {})
        if not isinstance(final_memory, dict):
            final_memory = {}

        windowed_snapshots = collect_windowed_snapshots(memory_snapshots)
        if not windowed_snapshots and final_memory:
            final_window_index = infer_snapshot_window_index(final_memory)
            normalized_window_index = final_window_index if final_window_index is not None else -1
            windowed_snapshots = [(int(normalized_window_index), final_memory)]

        if not windowed_snapshots:
            print("   WARNING: No snapshots found in source memory, skipping...")
            return None

        snapshot_by_window = {int(window_idx): snapshot for window_idx, snapshot in windowed_snapshots}
        oracle_prediction_path = (
            oracle_results_dir / "patients" / f"{subject_id}_{icu_stay_id}" / "oracle_predictions.json"
        )
        (
            oracle_actions_by_interval,
            oracle_labeled_intervals,
            oracle_markers_by_interval,
            ordered_oracle_intervals,
        ) = _load_oracle_targets_by_interval(oracle_prediction_path)
        snapshot_records: List[Dict[str, Any]] = []
        for window_idx, snapshot in windowed_snapshots:
            start_hour, end_hour = _snapshot_interval(
                snapshot,
                source_label=f"{source_patient_dir}/memory_database.json window_index={window_idx}",
            )
            snapshot_records.append(
                {
                    "window_index": int(window_idx),
                    "snapshot": snapshot,
                    "start_hour": float(start_hour),
                    "end_hour": float(end_hour),
                    "interval_key": _interval_key(start_hour, end_hour),
                }
            )
        labeled_snapshot_records = [
            record for record in snapshot_records if record["interval_key"] in oracle_labeled_intervals
        ]
        selected_snapshot_records = _select_records_with_stride(labeled_snapshot_records, snapshot_stride)

        if verbose:
            print(
                f"   Memory snapshots: total={len(windowed_snapshots)}, "
                f"oracle_labeled={len(labeled_snapshot_records)}, "
                f"selected={len(selected_snapshot_records)}, stride={snapshot_stride}"
            )

        recommendation_predictions: List[Dict[str, Any]] = []
        recommendation_call_logs: List[Dict[str, Any]] = []
        rag_retriever: Optional[IncrementalWindowRAGRetriever] = None
        if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS:
            rag_window_records = [
                _build_rag_record_from_snapshot(
                    window_index=int(record["window_index"]),
                    start_hour=float(record["start_hour"]),
                    end_hour=float(record["end_hour"]),
                    snapshot=record["snapshot"],
                )
                for record in snapshot_records
            ]
            rag_retriever = IncrementalWindowRAGRetriever(
                window_records=rag_window_records,
                encoder=SentenceTransformerWindowEncoder(
                    model_name=RAG_EMBEDDING_MODEL,
                    device="cuda",
                    batch_size=RAG_EMBEDDING_BATCH_SIZE,
                ),
                top_k=RAG_TOP_K_WINDOWS,
            )

        total_input_tokens = 0
        total_output_tokens = 0
        total_masked_action_events = 0
        num_windows_skipped_no_ground_truth = 0
        full_history_context_limit_reached = False
        full_history_context_limit_window_index: Optional[int] = None
        full_history_context_limit_message = ""

        for sequence_idx, selected_record in enumerate(selected_snapshot_records, start=1):
            selected_window_index = int(selected_record["window_index"])
            selected_snapshot = selected_record["snapshot"]
            current_start_hour = float(selected_record["start_hour"])
            current_end_hour = float(selected_record["end_hour"])
            current_interval_key = selected_record["interval_key"]
            lookup_window_index = int(selected_window_index)
            if lookup_window_index <= 0:
                continue
            current_oracle_marker = oracle_markers_by_interval.get(current_interval_key)
            if not isinstance(current_oracle_marker, Mapping):
                raise ValueError(
                    f"Missing oracle marker for current interval={current_interval_key} "
                    f"for patient={subject_id}_{icu_stay_id}"
                )

            skip_due_to_prior_context_limit = (
                context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS and full_history_context_limit_reached
            )
            snapshot_source = context_mode
            rag_retrieved_windows: List[RAGRetrievedWindow] = []
            if context_mode == CONTEXT_MODE_MEMORY:
                previous_snapshot = snapshot_by_window.get(lookup_window_index - 1)
                if not isinstance(previous_snapshot, Mapping):
                    if verbose:
                        print(
                            f"   Skipping Window {lookup_window_index}: "
                            f"missing prior snapshot {lookup_window_index - 1}"
                        )
                    continue
                context_snapshot, masked_action_count, non_action_events = build_masked_recommendation_snapshot(
                    previous_snapshot=previous_snapshot,
                    current_snapshot=selected_snapshot,
                    actionable_codes=actionable_codes,
                )
                context = render_snapshot_to_text(context_snapshot)
                snapshot_source = "precomputed_icu_evo_memory"
            elif context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS:
                if skip_due_to_prior_context_limit:
                    current_window = _extract_current_window(selected_snapshot)
                    current_events = _window_events(current_window)
                    masked_events, masked_action_count = _filter_non_action_events(
                        current_events,
                        actionable_codes=actionable_codes,
                    )
                    non_action_events = int(len(masked_events))
                    context = ""
                else:
                    context, masked_action_count, non_action_events = build_full_history_event_context(
                        snapshot_by_window=snapshot_by_window,
                        current_window_index=lookup_window_index,
                        current_snapshot=selected_snapshot,
                        actionable_codes=actionable_codes,
                    )
            elif context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY:
                context, masked_action_count, non_action_events = build_local_events_only_context(
                    current_snapshot=selected_snapshot,
                    actionable_codes=actionable_codes,
                )
            elif context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS:
                if rag_retriever is None:
                    raise ValueError("rag_retriever must be initialized for rag_history_events mode")
                current_record, masked_action_count, non_action_events = _build_masked_current_rag_record(
                    window_index=lookup_window_index,
                    start_hour=current_start_hour,
                    end_hour=current_end_hour,
                    snapshot=selected_snapshot,
                    actionable_codes=actionable_codes,
                )
                rag_retrieved_windows = rag_retriever.retrieve(
                    current_window_index=lookup_window_index,
                    query_text=current_record.event_text,
                )
                context = render_rag_event_context(
                    current_window=current_record,
                    retrieved_windows=rag_retrieved_windows,
                    retrieval_top_k=RAG_TOP_K_WINDOWS,
                )
                snapshot_source = CONTEXT_MODE_RAG_HISTORY_EVENTS
            else:
                raise ValueError(f"Unsupported context_mode={context_mode}")

            inferred_window_index, _, _ = extract_snapshot_window_features(selected_snapshot)
            window_index = int(lookup_window_index)
            if inferred_window_index >= 0:
                window_index = inferred_window_index

            window_duration_hours = float(current_end_hour - current_start_hour)
            if window_duration_hours <= 0:
                raise ValueError(
                    f"Invalid snapshot interval for patient={subject_id}_{icu_stay_id} window={lookup_window_index}: "
                    f"start={current_start_hour}, end={current_end_hour}"
                )
            future_window_count = compute_future_window_count_for_horizon(
                prediction_horizon_hours=prediction_horizon_hours,
                window_duration_hours=window_duration_hours,
            )
            ground_truth_events, included_horizon_markers = collect_union_ground_truth_actions_by_time(
                oracle_actions_by_interval=oracle_actions_by_interval,
                oracle_markers_by_interval=oracle_markers_by_interval,
                ordered_oracle_intervals=ordered_oracle_intervals,
                current_window_start_hour=float(current_start_hour),
                prediction_horizon_hours=float(prediction_horizon_hours),
            )
            if not ground_truth_events:
                num_windows_skipped_no_ground_truth += 1
                if verbose:
                    print(f"   Skipping Window {lookup_window_index}: no GT actions in horizon")
                continue

            prompt = ""
            raw_response = ""
            usage: Dict[str, Any] = {}
            prediction_error: Optional[Dict[str, str]] = None
            if skip_due_to_prior_context_limit:
                prediction_error = {
                    "type": "llm_context_length_exceeded",
                    "message": (
                        f"Skipped inference after prior context-length failure at window "
                        f"{full_history_context_limit_window_index}: {full_history_context_limit_message}"
                    ).strip(),
                }
                if verbose:
                    print(
                        f"   WARNING: Window {lookup_window_index} skipped after prior full-history token limit at "
                        f"window {full_history_context_limit_window_index}; counted as prediction error."
                    )
            else:
                prompt = get_recommendation_action_prompt(
                    top_k_actions=top_k_actions,
                    prediction_horizon_hours=prediction_horizon_hours,
                ).replace("{context}", context)
                estimated_input_tokens = agent.llm_client.estimate_input_tokens(prompt=prompt)
                usage = {"input_tokens": int(estimated_input_tokens), "output_tokens": 0}
                if estimated_input_tokens > int(agent.llm_client.max_input_tokens):
                    message = (
                        f"Input prompt token limit exceeded: {estimated_input_tokens} > "
                        f"{agent.llm_client.max_input_tokens} "
                        f"(provider={agent.llm_client.provider}, model={agent.llm_client.model})"
                    )
                    prediction_error = {
                        "type": "llm_context_length_exceeded",
                        "message": message,
                    }
                    if context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS:
                        full_history_context_limit_reached = True
                        full_history_context_limit_window_index = int(lookup_window_index)
                        full_history_context_limit_message = message
                    if verbose:
                        print(
                            f"   WARNING: Window {lookup_window_index} prompt exceeds token limit; "
                            "counted as prediction error without API call."
                        )
                else:
                    try:
                        response = agent.llm_client.chat(prompt=prompt, response_format="text")
                        raw_response = str(response.get("content", ""))
                        usage_obj = response.get("usage", {})
                        if isinstance(usage_obj, dict):
                            usage = usage_obj
                    except Exception as e:
                        if is_context_length_exceeded_error(e):
                            if context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS:
                                full_history_context_limit_reached = True
                                full_history_context_limit_window_index = int(lookup_window_index)
                                full_history_context_limit_message = str(e)
                            prediction_error = {
                                "type": "llm_context_length_exceeded",
                                "message": str(e),
                            }
                            if verbose:
                                print(
                                    f"   WARNING: Window {lookup_window_index} hit token/context limit; "
                                    "counted as prediction error."
                                )
                        else:
                            raise

            input_tokens = _normalize_token_count(usage.get("input_tokens"))
            output_tokens = _normalize_token_count(usage.get("output_tokens"))
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            if prediction_error is not None:
                parsed_prediction = {"recommended_actions": [], "red_flags": []}
                recommended_actions: List[Dict[str, Any]] = []
                red_flags: List[Dict[str, Any]] = []
                confidence = "Unknown"
                rationale = ""
            else:
                parsed_prediction = parse_json_dict_best_effort(raw_response)
                if parsed_prediction is None:
                    parsed_prediction = {}
                recommended_actions = _coerce_recommended_actions(parsed_prediction.get("recommended_actions"))[
                    : int(top_k_actions)
                ]
                red_flags = _coerce_red_flags(parsed_prediction.get("red_flags"))
                confidence, rationale = _extract_recommendation_summary_fields(parsed_prediction, recommended_actions)

            total_masked_action_events += int(masked_action_count)
            ground_truth_window_indices: List[int] = []
            for marker in included_horizon_markers:
                marker_index = _marker_best_window_index(marker)
                if marker_index is None:
                    continue
                ground_truth_window_indices.append(int(marker_index))

            prediction_item = {
                "window_index": int(window_index),
                "snapshot_sequence": int(sequence_idx),
                "hours_since_admission": float(current_start_hour),
                "window_start_hour": float(current_start_hour),
                "window_end_hour": float(current_end_hour),
                "snapshot_source": str(snapshot_source),
                "context_mode": str(context_mode),
                "oracle_window_marker_id": str(current_oracle_marker.get("marker_id") or ""),
                "oracle_window_marker": dict(current_oracle_marker),
                "num_masked_action_events": int(masked_action_count),
                "num_non_action_events_in_context_window": int(non_action_events),
                "num_rag_retrieved_windows": int(len(rag_retrieved_windows)),
                "rag_retrieved_window_indices": [int(item.window_index) for item in rag_retrieved_windows],
                "rag_retrieved_windows": _retrieved_windows_payload(rag_retrieved_windows),
                "rag_embedding_model": (
                    RAG_EMBEDDING_MODEL if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
                ),
                "rag_top_k_windows": (
                    int(RAG_TOP_K_WINDOWS) if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
                ),
                "rag_retrieval_scope": (
                    RAG_RETRIEVAL_SCOPE if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
                ),
                "num_ground_truth_action_events": int(len(ground_truth_events)),
                "prediction_horizon_hours": float(prediction_horizon_hours),
                "top_k_actions": int(top_k_actions),
                "window_duration_hours": float(window_duration_hours),
                "num_future_windows_for_horizon": int(future_window_count),
                "ground_truth_window_indices": ground_truth_window_indices,
                "ground_truth_window_marker_ids": [
                    str(marker.get("marker_id") or "") for marker in included_horizon_markers
                ],
                "ground_truth_window_markers": [dict(marker) for marker in included_horizon_markers],
                "confidence": confidence,
                "rationale": rationale,
                "recommended_actions": recommended_actions,
                "num_recommendations": int(len(recommended_actions)),
                "red_flags": red_flags,
                "num_red_flags": int(len(red_flags)),
                "ground_truth_action_events": ground_truth_events,
                "parsed_prediction": parsed_prediction,
                "prediction_error": prediction_error,
            }
            recommendation_predictions.append(prediction_item)

            recommendation_call_logs.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "patient_id": f"{subject_id}_{icu_stay_id}",
                    "window_index": int(window_index),
                    "hours_since_admission": float(current_start_hour),
                    "prompt": prompt,
                    "response": raw_response,
                    "parsed_response": parsed_prediction,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "metadata": {
                        "step_type": "recommendation_predictor",
                        "llm_provider": agent.llm_client.provider,
                        "llm_model": agent.llm_client.model,
                        "snapshot_source": str(snapshot_source),
                        "context_mode": str(context_mode),
                        "snapshot_sequence": int(sequence_idx),
                        "memory_run": str(memory_run_dir),
                        "num_masked_action_events": int(masked_action_count),
                        "num_rag_retrieved_windows": int(len(rag_retrieved_windows)),
                        "top_k_actions": int(top_k_actions),
                        "prediction_horizon_hours": float(prediction_horizon_hours),
                        "oracle_window_marker_id": str(current_oracle_marker.get("marker_id") or ""),
                        "prediction_error": prediction_error,
                    },
                }
            )

            if verbose:
                print(
                    f"   Recommendation Window {window_index}: "
                    f"rec={prediction_item['num_recommendations']} "
                    f"gt_actions={prediction_item['num_ground_truth_action_events']} "
                    f"horizon_windows={prediction_item['num_future_windows_for_horizon']} "
                    f"top_k={top_k_actions}"
                )

        if not recommendation_predictions:
            print("   WARNING: No eligible windows after GT-aware filtering, skipping...")
            return None

        total_recommendations = sum(int(item.get("num_recommendations", 0)) for item in recommendation_predictions)
        total_red_flags = sum(int(item.get("num_red_flags", 0)) for item in recommendation_predictions)
        total_ground_truth_action_events = sum(
            int(item.get("num_ground_truth_action_events", 0)) for item in recommendation_predictions
        )
        total_rag_retrieved_windows = sum(
            int(item.get("num_rag_retrieved_windows", 0)) for item in recommendation_predictions
        )
        num_failed_windows = sum(
            1 for item in recommendation_predictions if isinstance(item.get("prediction_error"), dict)
        )

        patient_metrics = {
            "num_windows_predicted": len(recommendation_predictions),
            "num_failed_windows": int(num_failed_windows),
            "num_windows_skipped_no_ground_truth": int(num_windows_skipped_no_ground_truth),
            "num_recommendations": total_recommendations,
            "num_red_flags": total_red_flags,
            "num_ground_truth_action_events": total_ground_truth_action_events,
            "num_rag_retrieved_windows": int(total_rag_retrieved_windows),
        }

        patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        prediction_payload = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "memory_run": str(memory_run_dir),
            "source_patient_dir": str(source_patient_dir),
            "context_mode": str(context_mode),
            "oracle_results_dir": str(oracle_results_dir),
            "num_memory_snapshots": len(windowed_snapshots),
            "num_oracle_labeled_snapshots": len(labeled_snapshot_records),
            "num_selected_snapshots_for_prediction": len(selected_snapshot_records),
            "snapshot_stride": snapshot_stride,
            "window_alignment_mode": "oracle_interval_time_match",
            "action_horizon": "current_plus_dynamic_future_windows",
            "prediction_horizon_hours": float(prediction_horizon_hours),
            "top_k_actions": int(top_k_actions),
            "actionable_event_codes": [str(code) for code in actionable_codes],
            "rag_embedding_model": RAG_EMBEDDING_MODEL if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_top_k_windows": int(RAG_TOP_K_WINDOWS) if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_retrieval_scope": RAG_RETRIEVAL_SCOPE if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "recommendation_predictions": recommendation_predictions,
            "recommendation_metrics": patient_metrics,
            "recommendation_prediction_tokens": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
            "total_masked_action_events": int(total_masked_action_events),
        }

        with open(patient_dir / "recommendation_predictions.json", "w") as f:
            json.dump(prediction_payload, f, indent=2)

        summary_payload = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "memory_run": str(memory_run_dir),
            "context_mode": str(context_mode),
            "num_memory_snapshots": len(windowed_snapshots),
            "num_oracle_labeled_snapshots": len(labeled_snapshot_records),
            "num_selected_snapshots_for_prediction": len(selected_snapshot_records),
            "snapshot_stride": snapshot_stride,
            "num_windows_predicted": patient_metrics["num_windows_predicted"],
            "num_failed_windows": patient_metrics["num_failed_windows"],
            "num_windows_skipped_no_ground_truth": patient_metrics["num_windows_skipped_no_ground_truth"],
            "num_recommendations": patient_metrics["num_recommendations"],
            "num_red_flags": patient_metrics["num_red_flags"],
            "num_ground_truth_action_events": patient_metrics["num_ground_truth_action_events"],
            "num_rag_retrieved_windows": patient_metrics["num_rag_retrieved_windows"],
            "rag_embedding_model": RAG_EMBEDDING_MODEL if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_top_k_windows": int(RAG_TOP_K_WINDOWS) if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_retrieval_scope": RAG_RETRIEVAL_SCOPE if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
        }
        with open(patient_dir / "prediction.json", "w") as f:
            json.dump(summary_payload, f, indent=2)

        if enable_logging:
            patient_logs = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "agent_type": f"recommendation_{context_mode}",
                "llm_provider": getattr(agent.llm_client, "provider", None),
                "llm_model": getattr(agent.llm_client, "model", None),
                "context_mode": str(context_mode),
                "pipeline_agents": [{"name": "recommendation_predictor", "used": True}],
                "total_calls": len(recommendation_call_logs),
                "calls": recommendation_call_logs,
            }
            with open(patient_dir / "llm_calls.json", "w") as f:
                json.dump(patient_logs, f, indent=2)
            save_llm_calls_html(patient_logs, patient_dir / "llm_calls.html")
            if verbose:
                print("   Saved log viewer: llm_calls.html")

        return {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "context_mode": str(context_mode),
            "num_memory_snapshots": len(windowed_snapshots),
            "num_oracle_labeled_snapshots": len(labeled_snapshot_records),
            "num_selected_snapshots_for_prediction": len(selected_snapshot_records),
            "snapshot_stride": snapshot_stride,
            "num_windows_predicted": patient_metrics["num_windows_predicted"],
            "num_failed_windows": patient_metrics["num_failed_windows"],
            "num_windows_skipped_no_ground_truth": patient_metrics["num_windows_skipped_no_ground_truth"],
            "num_recommendations": patient_metrics["num_recommendations"],
            "num_red_flags": patient_metrics["num_red_flags"],
            "num_ground_truth_action_events": patient_metrics["num_ground_truth_action_events"],
            "num_rag_retrieved_windows": patient_metrics["num_rag_retrieved_windows"],
            "rag_embedding_model": RAG_EMBEDDING_MODEL if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_top_k_windows": int(RAG_TOP_K_WINDOWS) if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_retrieval_scope": RAG_RETRIEVAL_SCOPE if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "total_masked_action_events": int(total_masked_action_events),
            "recommendation_prediction_tokens": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
        }

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def _save_run_config(results_dir: Path, payload: Dict[str, Any]) -> None:
    with open(results_dir / RUN_CONFIG_FILENAME, "w") as f:
        json.dump(payload, f, indent=2)


def run_experiment(
    memory_run: str,
    snapshot_stride: int,
    top_k_actions: int,
    prediction_horizon_hours: float,
    context_mode: str,
    oracle_results_dir: str,
    verbose: bool,
    enable_logging: bool,
    patient_stay_ids_path: Optional[str],
    num_workers: int = 4,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    requested_context_mode = str(context_mode).strip()
    if requested_context_mode == CONTEXT_MODE_ALL:
        mode_results: Dict[str, Any] = {}
        for mode in SUPPORTED_CONTEXT_MODES:
            print("\n" + "=" * 80)
            print(f"RUNNING CONTEXT MODE: {mode}")
            print("=" * 80)
            mode_results[mode] = run_experiment(
                memory_run=memory_run,
                snapshot_stride=snapshot_stride,
                top_k_actions=top_k_actions,
                prediction_horizon_hours=prediction_horizon_hours,
                context_mode=mode,
                oracle_results_dir=oracle_results_dir,
                verbose=verbose,
                enable_logging=enable_logging,
                patient_stay_ids_path=patient_stay_ids_path,
                num_workers=num_workers,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        return {
            "context_mode": CONTEXT_MODE_ALL,
            "executed_context_modes": list(SUPPORTED_CONTEXT_MODES),
            "mode_results": mode_results,
        }

    try:
        normalized_stride = int(snapshot_stride)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid snapshot_stride value: {snapshot_stride}") from exc
    if normalized_stride < 1:
        raise ValueError(f"snapshot_stride must be >= 1, got {normalized_stride}")
    try:
        normalized_top_k_actions = int(top_k_actions)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid top_k_actions value: {top_k_actions}") from exc
    if normalized_top_k_actions < 1:
        raise ValueError(f"top_k_actions must be >= 1, got {normalized_top_k_actions}")
    try:
        normalized_prediction_horizon_hours = float(prediction_horizon_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prediction_horizon_hours value: {prediction_horizon_hours}") from exc
    if not math.isfinite(normalized_prediction_horizon_hours) or normalized_prediction_horizon_hours <= 0:
        raise ValueError(f"prediction_horizon_hours must be > 0, got {normalized_prediction_horizon_hours}")
    try:
        normalized_num_workers = int(num_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid num_workers value: {num_workers}") from exc
    if normalized_num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {normalized_num_workers}")
    normalized_context_mode = _normalize_context_mode(context_mode)
    normalized_oracle_results_dir = Path(str(oracle_results_dir))
    if not normalized_oracle_results_dir.exists() or not normalized_oracle_results_dir.is_dir():
        raise FileNotFoundError(f"Oracle results directory not found: {normalized_oracle_results_dir}")
    oracle_patients_dir = normalized_oracle_results_dir / "patients"
    if not oracle_patients_dir.exists() or not oracle_patients_dir.is_dir():
        raise FileNotFoundError(f"Oracle patients directory not found: {oracle_patients_dir}")

    config = get_config()
    effective_llm_provider = str(llm_provider if llm_provider is not None else config.llm_provider).strip()
    if not effective_llm_provider:
        raise ValueError("LLM provider must be set in config.llm.provider or via --llm-provider.")
    effective_llm_model = str(llm_model if llm_model is not None else config.llm_model).strip()
    if not effective_llm_model:
        raise ValueError("LLM model must be set in config.llm.model or via --llm-model.")
    memory_run_dir = resolve_memory_run_dir(memory_run)
    source_run_config = load_memory_run_config(memory_run_dir)

    print("=" * 80)
    print("RECOMMENDATION EXPERIMENT")
    print("=" * 80)
    print(f"Memory Run: {memory_run_dir}")
    print(f"Snapshot Stride (k): {normalized_stride}")
    print(f"Top-K Actions: {normalized_top_k_actions}")
    print(f"Prediction Horizon (hours): {normalized_prediction_horizon_hours}")
    print(f"Context Mode: {normalized_context_mode}")
    print(f"Oracle Results Dir: {normalized_oracle_results_dir}")
    print(f"Num Workers: {normalized_num_workers}")
    print(f"LLM Provider: {effective_llm_provider}")
    print(f"LLM Model: {effective_llm_model}")

    results_dir = memory_run_dir / _results_dir_name_for_context_mode(normalized_context_mode)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {results_dir}")

    all_records = load_memory_patient_records(memory_run_dir)
    selected_records = filter_memory_patient_records_by_stay_ids(
        records=all_records,
        patient_stay_ids_path=patient_stay_ids_path,
    )

    print(f"Patients in memory run: {len(all_records)}")
    if patient_stay_ids_path:
        print(f"Patients after --patient-stay-ids filter: {len(selected_records)}")
    selected_records, skipped_completed = _filter_completed_patient_records(selected_records, results_dir)
    if skipped_completed:
        print(f"Skipping {skipped_completed} already-completed patients")

    run_config_payload = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "recommendation_experiment",
        "memory_input": {
            "memory_run": str(memory_run_dir),
            "source_generated_at": source_run_config.get("generated_at"),
            "source_experiment": source_run_config.get("experiment"),
            "selected_patients": len(selected_records),
        },
        "logging": {
            "enable_logging": bool(enable_logging),
        },
        "execution": {
            "num_workers": normalized_num_workers,
        },
        "recommendation_prediction": {
            "snapshot_stride": normalized_stride,
            "selection_mode": "oracle_labeled_window_time_match_then_stride",
            "window_alignment_mode": "oracle_interval_time_match",
            "top_k_actions": int(normalized_top_k_actions),
            "action_horizon": "current_plus_dynamic_future_windows",
            "prediction_horizon_hours": float(normalized_prediction_horizon_hours),
            "context_mode": str(normalized_context_mode),
            "oracle_results_dir": str(normalized_oracle_results_dir),
            "local_events_only_scope": (
                "working_memory_last_window_only"
                if normalized_context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY
                else None
            ),
            "rag_embedding_model": (
                RAG_EMBEDDING_MODEL if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
            ),
            "rag_top_k_windows": (
                int(RAG_TOP_K_WINDOWS) if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
            ),
            "rag_retrieval_scope": (
                RAG_RETRIEVAL_SCOPE if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
            ),
            "future_window_count_strategy": "ceil(prediction_horizon_hours / window_duration_hours)",
            "target_output": f"top_{int(normalized_top_k_actions)}",
            "skip_window_index": 0,
            "actionable_event_codes": [str(code) for code in ACTIONABLE_EVENT_CODES],
        },
        "llm": {
            "provider": effective_llm_provider,
            "model": effective_llm_model,
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_tokens,
        },
    }
    _save_run_config(results_dir, run_config_payload)

    if not selected_records:
        print("No patients available for prediction.")
        return {}

    all_results: List[Dict[str, Any]] = []
    patient_data = [(idx, record) for idx, record in enumerate(selected_records, 1)]

    def process_patient_wrapper(args):
        idx, patient_record = args
        patient_agent = ICUEvoAgent(
            provider=effective_llm_provider,
            observation_config_path=config.icu_evo_observation_config_path,
            episode_block_windows=config.icu_evo_episode_block_windows,
            insight_block_windows=config.icu_evo_insight_block_windows,
            model=effective_llm_model,
            enable_logging=False,
            window_duration_hours=config.agent_current_window_hours,
            max_working_windows=config.icu_evo_max_working_windows,
            max_insights=config.icu_evo_max_insights,
            max_trajectory_entries=config.icu_evo_max_trajectory_entries,
        )
        return process_single_patient(
            patient_record=patient_record,
            agent=patient_agent,
            memory_run_config=source_run_config,
            snapshot_stride=normalized_stride,
            top_k_actions=normalized_top_k_actions,
            prediction_horizon_hours=normalized_prediction_horizon_hours,
            context_mode=normalized_context_mode,
            oracle_results_dir=normalized_oracle_results_dir,
            memory_run_dir=memory_run_dir,
            results_dir=results_dir,
            patient_idx=idx,
            total_patients=len(selected_records),
            enable_logging=enable_logging,
            actionable_codes=ACTIONABLE_EVENT_CODES,
            verbose=verbose,
        )

    max_workers = min(normalized_num_workers, len(patient_data))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_patient_wrapper, item) for item in patient_data]
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)

    if not all_results:
        print("No patient results generated.")
        return {}

    total_patients = len(all_results)
    total_window_predictions = sum(int(item.get("num_windows_predicted", 0)) for item in all_results)
    total_failed_windows = sum(int(item.get("num_failed_windows", 0)) for item in all_results)
    total_oracle_labeled_snapshots = sum(int(item.get("num_oracle_labeled_snapshots", 0)) for item in all_results)
    total_selected_snapshots_for_prediction = sum(
        int(item.get("num_selected_snapshots_for_prediction", 0)) for item in all_results
    )
    total_recommendations = sum(int(item.get("num_recommendations", 0)) for item in all_results)
    total_red_flags = sum(int(item.get("num_red_flags", 0)) for item in all_results)
    total_ground_truth_action_events = sum(int(item.get("num_ground_truth_action_events", 0)) for item in all_results)
    total_rag_retrieved_windows = sum(int(item.get("num_rag_retrieved_windows", 0)) for item in all_results)
    total_masked_action_events = sum(int(item.get("total_masked_action_events", 0)) for item in all_results)
    total_windows_skipped_no_ground_truth = sum(
        int(item.get("num_windows_skipped_no_ground_truth", 0)) for item in all_results
    )

    token_totals = _sum_numeric_stats([item.get("recommendation_prediction_tokens", {}) for item in all_results])

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Total Patients: {total_patients}")
    print(f"Total Oracle-Labeled Snapshots: {total_oracle_labeled_snapshots}")
    print(f"Total Selected Snapshots for Prediction: {total_selected_snapshots_for_prediction}")
    print(f"Total Predicted Windows: {total_window_predictions}")
    print(f"Failed Predicted Windows: {total_failed_windows}")
    print(f"Total Recommendations: {total_recommendations}")
    print(f"Total Red-Flag Actions: {total_red_flags}")
    print(f"Total Ground-Truth Action Events: {total_ground_truth_action_events}")
    print(f"Total RAG Retrieved Windows: {total_rag_retrieved_windows}")
    print(f"Total Masked Action Events: {total_masked_action_events}")
    print(f"Skipped Windows (No GT in Horizon): {total_windows_skipped_no_ground_truth}")
    print("\nRecommendation Predictor Tokens:")
    print(f"  Input: {int(token_totals.get('input_tokens', 0))}")
    print(f"  Output: {int(token_totals.get('output_tokens', 0))}")
    print(f"  Total: {int(token_totals.get('total_tokens', 0))}")

    aggregate = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "recommendation_experiment",
        "memory_run": str(memory_run_dir),
        "context_mode": str(normalized_context_mode),
        "oracle_results_dir": str(normalized_oracle_results_dir),
        "snapshot_stride": normalized_stride,
        "window_alignment_mode": "oracle_interval_time_match",
        "selection_mode": "oracle_labeled_window_time_match_then_stride",
        "num_workers": normalized_num_workers,
        "top_k_actions": int(normalized_top_k_actions),
        "action_horizon": "current_plus_dynamic_future_windows",
        "prediction_horizon_hours": float(normalized_prediction_horizon_hours),
        "local_events_only_scope": (
            "working_memory_last_window_only" if normalized_context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY else None
        ),
        "rag_embedding_model": (
            RAG_EMBEDDING_MODEL if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
        ),
        "rag_top_k_windows": (
            int(RAG_TOP_K_WINDOWS) if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
        ),
        "rag_retrieval_scope": (
            RAG_RETRIEVAL_SCOPE if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
        ),
        "target_output": f"top_{int(normalized_top_k_actions)}",
        "total_patients": total_patients,
        "total_oracle_labeled_snapshots": int(total_oracle_labeled_snapshots),
        "total_selected_snapshots_for_prediction": int(total_selected_snapshots_for_prediction),
        "total_window_predictions": total_window_predictions,
        "total_failed_windows": total_failed_windows,
        "total_recommendations": total_recommendations,
        "total_red_flags": total_red_flags,
        "total_ground_truth_action_events": total_ground_truth_action_events,
        "total_rag_retrieved_windows": int(total_rag_retrieved_windows),
        "total_masked_action_events": total_masked_action_events,
        "total_windows_skipped_no_ground_truth": total_windows_skipped_no_ground_truth,
        "recommendation_prediction_tokens": {
            "input_tokens": int(token_totals.get("input_tokens", 0)),
            "output_tokens": int(token_totals.get("output_tokens", 0)),
            "total_tokens": int(token_totals.get("total_tokens", 0)),
        },
        "individual_results": sorted(all_results, key=lambda item: (item["subject_id"], item["icu_stay_id"])),
    }

    with open(results_dir / AGGREGATE_FILENAME, "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Recommendation prediction from precomputed ICU-Evo memory and event baselines"
    )
    parser.add_argument(
        "--memory-run",
        type=str,
        required=True,
        help="Path to memory run directory produced by experiments/create_memory.py.",
    )
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=8,
        help="Use every k-th oracle-labeled, time-aligned memory snapshot for recommendation prediction.",
    )
    parser.add_argument(
        "--top-k-actions",
        type=int,
        default=5,
        help="Number of actions to request in the recommendation prompt.",
    )
    parser.add_argument(
        "--prediction-horizon-hours",
        type=float,
        default=1,
        help="Prediction horizon in hours used for dynamic future-window GT collection.",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        required=True,
        choices=list(SUPPORTED_CONTEXT_MODE_CHOICES),
        help=(
            "Context construction mode: "
            "memory | full_history_events | local_events_only | rag_history_events | all."
        ),
    )
    parser.add_argument(
        "--oracle-results-dir",
        type=str,
        required=True,
        help="Oracle run directory containing patients/*/oracle_predictions.json; GT uses acceptable/best_practice.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers for patient processing (default: 2).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-logging", action="store_true", help="Disable detailed LLM call logging")
    parser.add_argument(
        "--patient-stay-ids",
        type=str,
        default=None,
        help="Optional CSV with columns subject_id,icu_stay_id to filter patients from memory run.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="Optional LLM provider override for this run.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Optional LLM model override for this run.",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("RUNNING: RECOMMENDATION EXPERIMENT")
    print(f"{'='*80}\n")

    run_experiment(
        memory_run=args.memory_run,
        snapshot_stride=args.snapshot_stride,
        top_k_actions=args.top_k_actions,
        prediction_horizon_hours=args.prediction_horizon_hours,
        context_mode=args.context_mode,
        oracle_results_dir=args.oracle_results_dir,
        verbose=not args.quiet,
        enable_logging=not args.no_logging,
        patient_stay_ids_path=args.patient_stay_ids,
        num_workers=args.num_workers,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
