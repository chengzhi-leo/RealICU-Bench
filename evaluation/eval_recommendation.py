"""Evaluate recommendation predictions with configurable action matching."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.plot_style import (
    METRIC_BAND_ALPHA,
    MODE_TRACE_ALPHA,
    PATIENT_TRACE_ALPHA,
    configure_plot_style,
    heatmap_cmap,
    metric_color,
    mode_color_map,
    plt,
    smooth_series,
)
from evaluation.pubmedbert_matcher import PubMedBERTMatcher
from experiments.oracle.action_validity_common import (
    event_to_action_text,
    normalize_action_label,
    recommendation_to_text,
)
from experiments.oracle.common import assign_normalized_time_bin
from model.llms import LLMClient
from prompts.action_matcher_prompts import get_action_matcher_prompt
from utils.json_parse import parse_json_dict_best_effort

configure_plot_style()

ORACLE_POSITIVE_LABELS = frozenset({"best_practice", "acceptable"})
MATCHER_BACKEND_LLM = "llm"
MATCHER_BACKEND_PUBMEDBERT = "pubmedbert"
SUPPORTED_MATCHER_BACKENDS = (MATCHER_BACKEND_PUBMEDBERT, MATCHER_BACKEND_LLM)
KNOWN_RECOMMENDATION_MODES = ("memory", "full_history_events", "local_events_only", "rag_history_events")
LLM_MATCHER_PROVIDER = "google"
LLM_MATCHER_MODEL = "qwen/qwen3-235b-a22b-instruct-2507-maas"
LLM_MATCHER_MAX_TOKENS = 12800
PUBMEDBERT_MODEL = "NeuML/pubmedbert-base-embeddings"
PUBMEDBERT_SIMILARITY_THRESHOLD = 0.5
PUBMEDBERT_BATCH_SIZE = 64

NUM_TIME_BINS = 10
MEMORY_DEPTH_NUM_BINS = 6
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260409
FIXED_EVAL_K = 5
HIT_MATCH_COUNT_THRESHOLD = 1
ABSOLUTE_TIME_BIN_HOURS = 24.0
CURVE_SMOOTHING_WINDOW = 2
_THREAD_LOCAL = threading.local()
_PUBMEDBERT_LOCK = threading.Lock()
_PUBMEDBERT_MATCHER: Optional[PubMedBERTMatcher] = None
_PUBMEDBERT_MATCHER_KEY: Optional[Tuple[str, Optional[str], int]] = None
ORACLE_INTERVAL_DECIMALS = 6

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


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


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _json_dump(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe_json_value(payload), f, indent=2, ensure_ascii=False)


def _parse_json_list_cell(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list cell, got: {text[:100]}")
    return parsed


def _ensure_predictions_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {path}")
    direct_patients = path / "patients"
    nested_patients = path / "recommendation_experiment" / "patients"
    if direct_patients.exists():
        matches = list(direct_patients.glob("*/recommendation_predictions.json"))
        if matches:
            return path
    if nested_patients.exists():
        matches = list(nested_patients.glob("*/recommendation_predictions.json"))
        if matches:
            return path / "recommendation_experiment"
    raise FileNotFoundError(
        f"No recommendation_predictions.json found under {path}. "
        "Expected either <dir>/patients/*/recommendation_predictions.json or "
        "<dir>/recommendation_experiment/patients/*/recommendation_predictions.json."
    )


def _ensure_gt_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"GT directory not found: {path}")
    patients_dir = path / "patients"
    if not patients_dir.exists() or not patients_dir.is_dir():
        raise FileNotFoundError(f"GT patients directory not found: {patients_dir}")
    matches = list(patients_dir.glob("*/oracle_predictions.json"))
    if not matches:
        raise FileNotFoundError(f"No oracle_predictions.json files found under {patients_dir}")
    return path


def _has_recommendation_predictions(path: Path) -> bool:
    patients_dir = path / "patients"
    if not patients_dir.exists() or not patients_dir.is_dir():
        return False
    return any(patients_dir.glob("*/recommendation_predictions.json"))


def _discover_recommendation_mode_dirs(path: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for mode in KNOWN_RECOMMENDATION_MODES:
        candidate = path / mode
        if candidate.exists() and candidate.is_dir() and _has_recommendation_predictions(candidate):
            discovered[str(mode)] = candidate
    return discovered


def _has_recommendation_eval_outputs(path: Path) -> bool:
    return (path / "window_level_windows.csv").exists()


def _discover_recommendation_eval_mode_dirs(path: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for mode in KNOWN_RECOMMENDATION_MODES:
        candidate = path / mode
        if candidate.exists() and candidate.is_dir() and _has_recommendation_eval_outputs(candidate):
            discovered[str(mode)] = candidate
    return discovered


def _ensure_recommendation_eval_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Reuse eval directory not found: {path}")
    if _has_recommendation_eval_outputs(path) or _discover_recommendation_eval_mode_dirs(path):
        return path
    raise FileNotFoundError(
        f"No window_level_windows.csv found under {path}. "
        "Expected either <dir>/window_level_windows.csv or <dir>/<mode>/window_level_windows.csv."
    )


def _default_output_dir_from_recommendation_root(recommendation_root: Path) -> Path:
    experiment_name = recommendation_root.name
    memory_run_name = recommendation_root.parent.name if recommendation_root.parent != recommendation_root else ""
    if not memory_run_name:
        raise ValueError(f"Cannot infer memory run name from recommendation_root={recommendation_root}")
    if not experiment_name:
        raise ValueError(f"Cannot infer experiment name from recommendation_root={recommendation_root}")
    return Path("evaluation_results") / memory_run_name / experiment_name


def _default_output_root_from_recommendation_input(recommendation_input: Path) -> Path:
    if recommendation_input.name == "recommendation_experiment":
        memory_run_name = recommendation_input.parent.name
        if not memory_run_name:
            raise ValueError(f"Cannot infer memory run name from recommendation_input={recommendation_input}")
        return Path("evaluation_results") / memory_run_name / "recommendation_experiment"
    if recommendation_input.parent.name == "recommendation_experiment":
        memory_run_name = recommendation_input.parent.parent.name
        if not memory_run_name:
            raise ValueError(f"Cannot infer memory run name from recommendation_input={recommendation_input}")
        return Path("evaluation_results") / memory_run_name / "recommendation_experiment"
    return _default_output_dir_from_recommendation_root(recommendation_input)


def _infer_snapshot_window_index(snapshot: Mapping[str, Any]) -> Optional[int]:
    window_id = snapshot.get("last_processed_window_index")
    if window_id is None:
        return None
    return int(window_id)


def _collect_windowed_snapshots(memory_snapshots: Sequence[Mapping[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    snapshots_by_window: Dict[int, Dict[str, Any]] = {}
    for snapshot in memory_snapshots:
        if not isinstance(snapshot, Mapping):
            raise ValueError("Invalid memory snapshot row; expected mapping.")
        window_index = _infer_snapshot_window_index(snapshot)
        if window_index is None:
            continue
        snapshots_by_window[int(window_index)] = dict(snapshot)
    return sorted(snapshots_by_window.items(), key=lambda item: item[0])


def _load_memory_snapshots(patient_dir: Path) -> List[Mapping[str, Any]]:
    memory_db_path = patient_dir / "memory_database.json"
    payload = _load_json(memory_db_path)
    memory_snapshots = payload.get("memory_snapshots")
    if not isinstance(memory_snapshots, list):
        raise ValueError(f"Invalid memory_snapshots list in {memory_db_path}")
    return memory_snapshots


def _memory_depth_by_window(source_patient_dir: Path) -> Dict[int, int]:
    memory_snapshots = _load_memory_snapshots(source_patient_dir)
    windowed = _collect_windowed_snapshots(memory_snapshots)
    depths: Dict[int, int] = {}
    for window_index, snapshot in windowed:
        trajectory_memory = snapshot.get("trajectory_memory")
        trend_memory = snapshot.get("trend_memory")
        critical_events_memory = snapshot.get("critical_events_memory")
        insights = snapshot.get("insights")
        if not isinstance(trajectory_memory, list):
            trajectory_memory = []
        if not isinstance(trend_memory, list):
            trend_memory = []
        if not isinstance(critical_events_memory, list):
            critical_events_memory = []
        if not isinstance(insights, list):
            insights = []
        depth = len(trajectory_memory) + len(trend_memory) + len(critical_events_memory) + len(insights)
        depths[int(window_index)] = int(depth)
    return depths


def _ground_truth_action_text(item: Mapping[str, Any]) -> str:
    text = str(item.get("gt_action_text") or "").strip()
    if text:
        return text
    text = recommendation_to_text(item)
    if text:
        return text
    return event_to_action_text(item)


def _extract_icu_stay_id(payload: Mapping[str, Any], source_path: Path) -> int:
    icu_stay_id_raw = payload.get("icu_stay_id")
    if icu_stay_id_raw is None:
        raise ValueError(f"Missing icu_stay_id in {source_path}")
    icu_stay_id = int(icu_stay_id_raw)
    if icu_stay_id <= 0:
        raise ValueError(f"Invalid icu_stay_id={icu_stay_id} in {source_path}")
    return icu_stay_id


def _interval_key(start_hour: float, end_hour: float) -> Tuple[float, float]:
    return (round(float(start_hour), ORACLE_INTERVAL_DECIMALS), round(float(end_hour), ORACLE_INTERVAL_DECIMALS))


def _interval_marker_id(interval_key: Tuple[float, float]) -> str:
    return f"{interval_key[0]:.{ORACLE_INTERVAL_DECIMALS}f}|{interval_key[1]:.{ORACLE_INTERVAL_DECIMALS}f}"


def _parse_window_interval(
    *,
    hours_since_admission: Any,
    current_window_hours: Any,
    source_path: Path,
    context_label: str,
) -> Tuple[float, float]:
    try:
        start_hour = float(hours_since_admission)
        duration_hours = float(current_window_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid hour fields in {source_path} ({context_label}): "
            f"hours_since_admission={hours_since_admission!r}, current_window_hours={current_window_hours!r}"
        ) from exc
    if duration_hours <= 0:
        raise ValueError(
            f"current_window_hours must be > 0 in {source_path} ({context_label}), got {duration_hours!r}"
        )
    return start_hour, start_hour + duration_hours


def _resolve_gt_window_interval(
    row: Mapping[str, Any],
    gt_prediction_path: Path,
) -> Tuple[float, float]:
    metadata = row.get("window_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Missing window_metadata in row in {gt_prediction_path}")
    start_hour, end_hour = _parse_window_interval(
        hours_since_admission=metadata.get("hours_since_admission"),
        current_window_hours=metadata.get("current_window_hours"),
        source_path=gt_prediction_path,
        context_label="gt_window",
    )
    return _interval_key(start_hour, end_hour)


def _load_gt_targets_by_interval(
    gt_prediction_path: Path,
) -> Tuple[Dict[Tuple[float, float], List[Dict[str, Any]]], Dict[Tuple[float, float], List[Dict[str, Any]]]]:
    payload = _load_json(gt_prediction_path)
    window_outputs = payload.get("window_outputs")
    if not isinstance(window_outputs, list):
        raise ValueError(f"Invalid window_outputs list in {gt_prediction_path}")

    actions_by_interval: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    red_flags_by_interval: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for row in window_outputs:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid row in window_outputs in {gt_prediction_path}")
        interval_key = _resolve_gt_window_interval(row=row, gt_prediction_path=gt_prediction_path)
        marker_id = _interval_marker_id(interval_key)

        oracle_output = row.get("oracle_output")
        if not isinstance(oracle_output, Mapping):
            continue
        action_review = oracle_output.get("action_review")
        if not isinstance(action_review, Mapping):
            continue
        evaluations = action_review.get("evaluations")
        if evaluations is not None and not isinstance(evaluations, list):
            raise ValueError(f"Invalid action_review.evaluations for interval={marker_id} in {gt_prediction_path}")
        selected_actions: List[Dict[str, Any]] = []
        if evaluations is not None:
            for evaluation in evaluations:
                if not isinstance(evaluation, Mapping):
                    raise ValueError(
                        f"Invalid evaluation row for interval={marker_id} in {gt_prediction_path}: {evaluation}"
                    )
                label = normalize_action_label(evaluation.get("label"))
                if label not in ORACLE_POSITIVE_LABELS:
                    continue
                action_name = str(evaluation.get("action_name") or "").strip()
                if not action_name:
                    raise ValueError(
                        f"Missing action_name for interval={marker_id} with label={label} in {gt_prediction_path}"
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
                        "oracle_interval_marker_id": marker_id,
                        "oracle_interval_start_hour": float(interval_key[0]),
                        "oracle_interval_end_hour": float(interval_key[1]),
                    }
                )

        if selected_actions:
            if interval_key in actions_by_interval:
                raise ValueError(f"Duplicate oracle action reviews for interval={marker_id} in {gt_prediction_path}")
            actions_by_interval[interval_key] = selected_actions

        red_flags = action_review.get("red_flags")
        if red_flags is not None and not isinstance(red_flags, list):
            raise ValueError(f"Invalid action_review.red_flags for interval={marker_id} in {gt_prediction_path}")
        selected_red_flags: List[Dict[str, Any]] = []
        if red_flags is not None:
            for red_flag in red_flags:
                if not isinstance(red_flag, Mapping):
                    raise ValueError(
                        f"Invalid red_flag row for interval={marker_id} in {gt_prediction_path}: {red_flag}"
                    )
                contraindicated_action = str(red_flag.get("contraindicated_action") or "").strip()
                if not contraindicated_action:
                    raise ValueError(
                        f"Missing contraindicated_action for interval={marker_id} in {gt_prediction_path}"
                    )
                reason = str(red_flag.get("reason") or "").strip()
                gt_text = f"{contraindicated_action}. {reason}".strip() if reason else contraindicated_action
                selected_red_flags.append(
                    {
                        "contraindicated_action": contraindicated_action,
                        "reason": reason,
                        "gt_action_text": gt_text,
                        "oracle_interval_marker_id": marker_id,
                        "oracle_interval_start_hour": float(interval_key[0]),
                        "oracle_interval_end_hour": float(interval_key[1]),
                    }
                )

        if selected_red_flags:
            if interval_key in red_flags_by_interval:
                raise ValueError(f"Duplicate oracle red_flags for interval={marker_id} in {gt_prediction_path}")
            red_flags_by_interval[interval_key] = selected_red_flags

    return actions_by_interval, red_flags_by_interval


def _load_gt_window_intervals(gt_prediction_path: Path) -> Set[Tuple[float, float]]:
    payload = _load_json(gt_prediction_path)
    window_outputs = payload.get("window_outputs")
    if not isinstance(window_outputs, list):
        raise ValueError(f"Invalid window_outputs list in {gt_prediction_path}")
    intervals: Set[Tuple[float, float]] = set()
    for row in window_outputs:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid row in window_outputs in {gt_prediction_path}")
        intervals.add(_resolve_gt_window_interval(row=row, gt_prediction_path=gt_prediction_path))
    return intervals


def _resolve_prediction_interval(item: Mapping[str, Any], source_path: Path) -> Tuple[float, float]:
    marker = item.get("oracle_window_marker")
    if isinstance(marker, Mapping):
        start_raw = marker.get("interval_start_hour")
        end_raw = marker.get("interval_end_hour")
        if start_raw is not None and end_raw is not None:
            try:
                return _interval_key(float(start_raw), float(end_raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid oracle_window_marker in {source_path}: {marker}") from exc
    start_raw = item.get("window_start_hour")
    end_raw = item.get("window_end_hour")
    if start_raw is not None and end_raw is not None:
        try:
            return _interval_key(float(start_raw), float(end_raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid window_start_hour/window_end_hour in prediction row in {source_path}: "
                f"{start_raw!r}, {end_raw!r}"
            ) from exc
    hours_raw = item.get("hours_since_admission")
    duration_raw = item.get("window_duration_hours")
    if hours_raw is not None and duration_raw is not None:
        start_hour, end_hour = _parse_window_interval(
            hours_since_admission=hours_raw,
            current_window_hours=duration_raw,
            source_path=source_path,
            context_label="prediction_window",
        )
        return _interval_key(start_hour, end_hour)
    raise ValueError(
        f"Missing oracle_window_marker/window_start_hour/window_end_hour/window_duration_hours in prediction row in "
        f"{source_path}"
    )


def _build_window_matching_prompt(
    prediction_items: Sequence[Mapping[str, Any]],
    oracle_recommended_gt_items: Sequence[Mapping[str, Any]],
    oracle_red_flag_gt_items: Sequence[Mapping[str, Any]],
) -> str:
    prediction_payload = []
    for idx, item in enumerate(prediction_items):
        prediction_payload.append(
            {
                "prediction_index": int(idx),
                "action_name": str(item.get("action_name") or "").strip(),
                "action_description": str(item.get("action_description") or "").strip(),
            }
        )

    recommended_gt_payload = []
    for idx, item in enumerate(oracle_recommended_gt_items):
        recommended_gt_payload.append(
            {
                "gt_index": int(idx),
                "action_description": _ground_truth_action_text(item),
            }
        )
    red_flag_gt_payload = []
    for idx, item in enumerate(oracle_red_flag_gt_items):
        red_flag_gt_payload.append(
            {
                "gt_index": int(idx),
                "action_description": _ground_truth_action_text(item),
            }
        )

    prompt_template = get_action_matcher_prompt()
    predicted_actions = json.dumps(prediction_payload, ensure_ascii=False, indent=2)
    oracle_recommended_actions = json.dumps(recommended_gt_payload, ensure_ascii=False, indent=2)
    oracle_red_flag_actions = json.dumps(red_flag_gt_payload, ensure_ascii=False, indent=2)
    return (
        prompt_template.replace("{predicted_actions}", predicted_actions)
        .replace("{oracle_recommended_actions}", oracle_recommended_actions)
        .replace("{oracle_red_flag_actions}", oracle_red_flag_actions)
    )


def _extract_match_pairs(
    parsed_payload: Dict[str, Any],
    *,
    num_predictions: int,
    num_ground_truth: int,
) -> List[Dict[str, int]]:
    matched_pairs_raw = parsed_payload.get("matched_pairs")
    if not isinstance(matched_pairs_raw, list):
        matches_raw = parsed_payload.get("matches")
        if isinstance(matches_raw, list):
            matched_pairs = []
            seen_predictions = set()
            for row in matches_raw:
                if not isinstance(row, Mapping):
                    raise ValueError(f"Invalid matches row: {row}")
                pred_raw = row.get("pred_idx")
                gt_indices_raw = row.get("gt_indices")
                try:
                    prediction_index = int(pred_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid pred_idx in matches row: {row}") from exc
                if prediction_index < 0 or prediction_index >= num_predictions:
                    raise ValueError(f"pred_idx out of range: {prediction_index}")
                if prediction_index in seen_predictions:
                    continue
                seen_predictions.add(prediction_index)

                if not isinstance(gt_indices_raw, list):
                    raise ValueError(f"gt_indices must be a list in matches row: {row}")
                parsed_gt_indices: List[int] = []
                for value in gt_indices_raw:
                    try:
                        gt_index = int(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid gt index in gt_indices: {value}") from exc
                    if gt_index < 0 or gt_index >= num_ground_truth:
                        raise ValueError(f"gt index out of range: {gt_index}")
                    parsed_gt_indices.append(gt_index)

                if not parsed_gt_indices:
                    continue
                matched_pairs.append(
                    {
                        "prediction_index": int(prediction_index),
                        "gt_index": int(parsed_gt_indices[0]),
                    }
                )
            return matched_pairs

    if not isinstance(matched_pairs_raw, list):
        matched_prediction_indices = parsed_payload.get("matched_prediction_indices")
        if not isinstance(matched_prediction_indices, list):
            raise ValueError("LLM output must include matched_pairs list.")
        matched_pairs: List[Dict[str, int]] = []
        seen_predictions = set()
        for value in matched_prediction_indices:
            try:
                prediction_index = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prediction index in matched_prediction_indices: {value}") from exc
            if prediction_index < 0 or prediction_index >= num_predictions:
                raise ValueError(f"Prediction index out of range: {prediction_index}")
            if prediction_index in seen_predictions:
                continue
            seen_predictions.add(prediction_index)
            matched_pairs.append({"prediction_index": int(prediction_index), "gt_index": -1})
        return matched_pairs

    matched_pairs = []
    seen_predictions = set()
    for row in matched_pairs_raw:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid matched pair row: {row}")

        prediction_raw = row.get("prediction_index")
        gt_raw = row.get("gt_index")
        try:
            prediction_index = int(prediction_raw)
            gt_index = int(gt_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pair indices: {row}") from exc

        if prediction_index < 0 or prediction_index >= num_predictions:
            raise ValueError(f"prediction_index out of range: {prediction_index}")
        if gt_index < 0 or gt_index >= num_ground_truth:
            raise ValueError(f"gt_index out of range: {gt_index}")
        if prediction_index in seen_predictions:
            continue

        seen_predictions.add(prediction_index)
        matched_pairs.append({"prediction_index": int(prediction_index), "gt_index": int(gt_index)})

    return matched_pairs


def _extract_match_pairs_from_section(
    section_payload: Any,
    *,
    num_predictions: int,
    num_ground_truth: int,
) -> List[Dict[str, int]]:
    if num_ground_truth == 0:
        return []
    if section_payload is None:
        return []
    if isinstance(section_payload, Mapping):
        return _extract_match_pairs(
            dict(section_payload),
            num_predictions=num_predictions,
            num_ground_truth=num_ground_truth,
        )
    if isinstance(section_payload, list):
        return _extract_match_pairs(
            {"matches": list(section_payload)},
            num_predictions=num_predictions,
            num_ground_truth=num_ground_truth,
        )
    raise ValueError(f"Invalid matcher section payload: {section_payload}")


def _match_window_with_llm(
    llm_client: LLMClient,
    *,
    prediction_items: Sequence[Mapping[str, Any]],
    oracle_recommended_gt_items: Sequence[Mapping[str, Any]],
    oracle_red_flag_gt_items: Sequence[Mapping[str, Any]],
    patient_id: str,
    window_index: int,
) -> Dict[str, Any]:
    if not prediction_items or (not oracle_recommended_gt_items and not oracle_red_flag_gt_items):
        return {
            "recommended_matched_pairs": [],
            "recommended_matched_prediction_indices": [],
            "red_flag_matched_pairs": [],
            "red_flag_matched_prediction_indices": [],
            "raw_response": "",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    prompt = _build_window_matching_prompt(
        prediction_items=prediction_items,
        oracle_recommended_gt_items=oracle_recommended_gt_items,
        oracle_red_flag_gt_items=oracle_red_flag_gt_items,
    )
    response = llm_client.chat(prompt=prompt, response_format="text", temperature=0)
    raw_response = response.get("content", "")
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}

    parsed = parse_json_dict_best_effort(raw_response)

    if parsed is None:
        preview = str(raw_response).strip()[:500]
        raise ValueError(
            f"Failed to parse LLM JSON for patient={patient_id}, window_index={window_index}. "
            f"Response preview: {preview}"
        )

    recommended_pairs: List[Dict[str, int]]
    red_flag_pairs: List[Dict[str, int]]
    if "recommended_action_matches" in parsed or "red_flag_matches" in parsed:
        recommended_pairs = _extract_match_pairs_from_section(
            parsed.get("recommended_action_matches"),
            num_predictions=len(prediction_items),
            num_ground_truth=len(oracle_recommended_gt_items),
        )
        red_flag_pairs = _extract_match_pairs_from_section(
            parsed.get("red_flag_matches"),
            num_predictions=len(prediction_items),
            num_ground_truth=len(oracle_red_flag_gt_items),
        )
    else:
        recommended_pairs = _extract_match_pairs(
            parsed,
            num_predictions=len(prediction_items),
            num_ground_truth=len(oracle_recommended_gt_items),
        )
        red_flag_pairs = []
    recommended_indices = sorted({int(item["prediction_index"]) for item in recommended_pairs})
    red_flag_indices = sorted({int(item["prediction_index"]) for item in red_flag_pairs})

    return {
        "recommended_matched_pairs": recommended_pairs,
        "recommended_matched_prediction_indices": recommended_indices,
        "red_flag_matched_pairs": red_flag_pairs,
        "red_flag_matched_prediction_indices": red_flag_indices,
        "raw_response": raw_response,
        "input_tokens": _normalize_token_count(usage.get("input_tokens")),
        "output_tokens": _normalize_token_count(usage.get("output_tokens")),
    }


def _get_thread_llm_client() -> LLMClient:
    client = getattr(_THREAD_LOCAL, "llm_client", None)
    client_key = getattr(_THREAD_LOCAL, "llm_client_key", None)
    next_key = (LLM_MATCHER_PROVIDER, LLM_MATCHER_MODEL, int(LLM_MATCHER_MAX_TOKENS))
    if client is None or client_key != next_key:
        client = LLMClient(
            provider=LLM_MATCHER_PROVIDER,
            model=LLM_MATCHER_MODEL,
            max_tokens=int(LLM_MATCHER_MAX_TOKENS),
        )
        _THREAD_LOCAL.llm_client = client
        _THREAD_LOCAL.llm_client_key = next_key
    return client


def _get_pubmedbert_matcher(
    *,
    model_name: str,
    device: Optional[str],
    batch_size: int,
) -> PubMedBERTMatcher:
    global _PUBMEDBERT_MATCHER, _PUBMEDBERT_MATCHER_KEY
    key = (str(model_name), str(device) if device is not None else None, int(batch_size))
    with _PUBMEDBERT_LOCK:
        if _PUBMEDBERT_MATCHER is None or _PUBMEDBERT_MATCHER_KEY != key:
            _PUBMEDBERT_MATCHER = PubMedBERTMatcher(
                model_name=str(model_name),
                device=device,
                batch_size=int(batch_size),
            )
            _PUBMEDBERT_MATCHER_KEY = key
    if _PUBMEDBERT_MATCHER is None:
        raise RuntimeError("Failed to initialize PubMedBERT matcher.")
    return _PUBMEDBERT_MATCHER


def _match_window_with_pubmedbert(
    matcher: PubMedBERTMatcher,
    *,
    prediction_items: Sequence[Mapping[str, Any]],
    oracle_recommended_gt_items: Sequence[Mapping[str, Any]],
    oracle_red_flag_gt_items: Sequence[Mapping[str, Any]],
    similarity_threshold: float,
) -> Dict[str, Any]:
    if not prediction_items or (not oracle_recommended_gt_items and not oracle_red_flag_gt_items):
        return {
            "recommended_matched_pairs": [],
            "recommended_matched_prediction_indices": [],
            "red_flag_matched_pairs": [],
            "red_flag_matched_prediction_indices": [],
            "raw_response": "",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    prediction_texts = [recommendation_to_text(item) for item in prediction_items]
    recommended_gt_texts = [_ground_truth_action_text(item) for item in oracle_recommended_gt_items]
    red_flag_gt_texts = [_ground_truth_action_text(item) for item in oracle_red_flag_gt_items]

    recommended_result = matcher.match_pairs(
        prediction_texts=prediction_texts,
        ground_truth_texts=recommended_gt_texts,
        similarity_threshold=float(similarity_threshold),
    )
    red_flag_result = matcher.match_pairs(
        prediction_texts=prediction_texts,
        ground_truth_texts=red_flag_gt_texts,
        similarity_threshold=float(similarity_threshold),
    )
    raw_payload = {
        "similarity_threshold": float(similarity_threshold),
        "recommended_action_matches": recommended_result,
        "red_flag_matches": red_flag_result,
    }
    return {
        "recommended_matched_pairs": list(recommended_result["matched_pairs"]),
        "recommended_matched_prediction_indices": list(recommended_result["matched_prediction_indices"]),
        "red_flag_matched_pairs": list(red_flag_result["matched_pairs"]),
        "red_flag_matched_prediction_indices": list(red_flag_result["matched_prediction_indices"]),
        "raw_response": json.dumps(raw_payload, ensure_ascii=False),
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _run_window_match_task(
    task: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    backend = str(task["matcher_backend"])
    try:
        if backend == MATCHER_BACKEND_LLM:
            llm_client = _get_thread_llm_client()
            match_result = _match_window_with_llm(
                llm_client,
                prediction_items=task["prediction_items"],
                oracle_recommended_gt_items=task["oracle_recommended_gt_items"],
                oracle_red_flag_gt_items=task["oracle_red_flag_gt_items"],
                patient_id=str(task["patient_id"]),
                window_index=int(task["window_index"]),
            )
        elif backend == MATCHER_BACKEND_PUBMEDBERT:
            matcher = _get_pubmedbert_matcher(
                model_name=str(task["pubmedbert_model"]),
                device=task["pubmedbert_device"],
                batch_size=int(task["pubmedbert_batch_size"]),
            )
            match_result = _match_window_with_pubmedbert(
                matcher,
                prediction_items=task["prediction_items"],
                oracle_recommended_gt_items=task["oracle_recommended_gt_items"],
                oracle_red_flag_gt_items=task["oracle_red_flag_gt_items"],
                similarity_threshold=float(task["pubmedbert_similarity_threshold"]),
            )
        else:
            raise ValueError(f"Unsupported matcher_backend: {backend}")
    except Exception as exc:
        raise RuntimeError(
            f"Matching failed for patient={task['patient_id']} window={task['window_index']} "
            f"backend={backend}: {exc}"
        ) from exc
    return task, match_result


def _build_rows_from_match(
    *,
    task: Dict[str, Any],
    match_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prediction_items = task["prediction_items"]
    recommended_gt_items = task["oracle_recommended_gt_items"]
    red_flag_gt_items = task["oracle_red_flag_gt_items"]
    recommended_pairs = match_result["recommended_matched_pairs"]
    red_flag_pairs = match_result["red_flag_matched_pairs"]
    recommended_prediction_indices = match_result["recommended_matched_prediction_indices"]
    red_flag_prediction_indices = match_result["red_flag_matched_prediction_indices"]
    recommended_match_set = set(int(value) for value in recommended_prediction_indices)
    red_flag_match_set = set(int(value) for value in red_flag_prediction_indices)

    window_row = {
        "patient_id": task["patient_id"],
        "subject_id": int(task["subject_id"]),
        "icu_stay_id": int(task["icu_stay_id"]),
        "window_index": int(task["window_index"]),
        "oracle_window_marker_id": str(task["oracle_window_marker_id"]),
        "oracle_interval_start_hour": float(task["oracle_interval_start_hour"]),
        "oracle_interval_end_hour": float(task["oracle_interval_end_hour"]),
        "num_windows": int(task["num_windows"]),
        "relative_time": float(task["relative_time"]),
        "time_bin": int(task["time_bin"]),
        "hours_since_admission": float(task["hours_since_admission"]),
        "top_k_actions": int(task["top_k_actions"]),
        "num_recommendations": int(task["num_recommendations"]),
        "num_ground_truth_actions": int(task["num_ground_truth_actions"]),
        "num_ground_truth_red_flags": int(task["num_ground_truth_red_flags"]),
        "matcher_backend": str(task["matcher_backend"]),
        "memory_depth": task["memory_depth"],
        "matched_prediction_indices": list(recommended_prediction_indices),
        "matched_pairs": list(recommended_pairs),
        "matched_red_flag_prediction_indices": list(red_flag_prediction_indices),
        "matched_red_flag_pairs": list(red_flag_pairs),
        "matcher_input_tokens": int(match_result["input_tokens"]),
        "matcher_output_tokens": int(match_result["output_tokens"]),
        "matcher_raw_response": str(match_result["raw_response"]),
        "llm_input_tokens": int(match_result["input_tokens"]),
        "llm_output_tokens": int(match_result["output_tokens"]),
        "llm_raw_response": str(match_result["raw_response"]),
    }

    prediction_rows: List[Dict[str, Any]] = []
    recommended_gt_texts = [_ground_truth_action_text(item) for item in recommended_gt_items]
    red_flag_gt_texts = [_ground_truth_action_text(item) for item in red_flag_gt_items]
    recommended_pair_by_prediction = {
        int(pair["prediction_index"]): int(pair["gt_index"]) for pair in recommended_pairs
    }
    red_flag_pair_by_prediction = {int(pair["prediction_index"]): int(pair["gt_index"]) for pair in red_flag_pairs}
    for prediction_index, prediction_item in enumerate(prediction_items):
        prediction_rows.append(
            {
                "patient_id": task["patient_id"],
                "subject_id": int(task["subject_id"]),
                "icu_stay_id": int(task["icu_stay_id"]),
                "window_index": int(task["window_index"]),
                "oracle_window_marker_id": str(task["oracle_window_marker_id"]),
                "oracle_interval_start_hour": float(task["oracle_interval_start_hour"]),
                "oracle_interval_end_hour": float(task["oracle_interval_end_hour"]),
                "prediction_index": int(prediction_index),
                "is_matched": int(prediction_index in recommended_match_set),
                "is_matched_recommended": int(prediction_index in recommended_match_set),
                "is_matched_red_flag": int(prediction_index in red_flag_match_set),
                "matched_gt_index": recommended_pair_by_prediction.get(prediction_index),
                "matched_recommended_gt_index": recommended_pair_by_prediction.get(prediction_index),
                "matched_red_flag_gt_index": red_flag_pair_by_prediction.get(prediction_index),
                "matcher_backend": str(task["matcher_backend"]),
                "recommended_action_name": str(prediction_item.get("action_name") or "").strip(),
                "recommended_action_description": str(prediction_item.get("action_description") or "").strip(),
                "recommended_action_text": recommendation_to_text(prediction_item),
                "matched_gt_action_text": (
                    recommended_gt_texts[recommended_pair_by_prediction[prediction_index]]
                    if prediction_index in recommended_pair_by_prediction
                    and recommended_pair_by_prediction[prediction_index] >= 0
                    and recommended_pair_by_prediction[prediction_index] < len(recommended_gt_texts)
                    else ""
                ),
                "matched_recommended_gt_action_text": (
                    recommended_gt_texts[recommended_pair_by_prediction[prediction_index]]
                    if prediction_index in recommended_pair_by_prediction
                    and recommended_pair_by_prediction[prediction_index] >= 0
                    and recommended_pair_by_prediction[prediction_index] < len(recommended_gt_texts)
                    else ""
                ),
                "matched_red_flag_action_text": (
                    red_flag_gt_texts[red_flag_pair_by_prediction[prediction_index]]
                    if prediction_index in red_flag_pair_by_prediction
                    and red_flag_pair_by_prediction[prediction_index] >= 0
                    and red_flag_pair_by_prediction[prediction_index] < len(red_flag_gt_texts)
                    else ""
                ),
                "num_ground_truth_actions": int(task["num_ground_truth_actions"]),
                "num_ground_truth_red_flags": int(task["num_ground_truth_red_flags"]),
            }
        )
    return window_row, prediction_rows


def _load_window_and_prediction_rows(
    prediction_dir: Path,
    *,
    gt_dir: Path,
    num_workers: int,
    matcher_backend: str,
    pubmedbert_model: str,
    pubmedbert_similarity_threshold: float,
    pubmedbert_batch_size: int,
    pubmedbert_device: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if int(num_workers) < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    backend = str(matcher_backend).strip().lower()
    if backend not in SUPPORTED_MATCHER_BACKENDS:
        raise ValueError(
            f"Unsupported matcher_backend={matcher_backend}. Expected one of {SUPPORTED_MATCHER_BACKENDS}"
        )
    if float(pubmedbert_similarity_threshold) < 0.0 or float(pubmedbert_similarity_threshold) > 1.0:
        raise ValueError(f"pubmedbert_similarity_threshold must be in [0, 1], got {pubmedbert_similarity_threshold}")
    if int(pubmedbert_batch_size) < 1:
        raise ValueError(f"pubmedbert_batch_size must be >= 1, got {pubmedbert_batch_size}")

    prediction_files = sorted((prediction_dir / "patients").glob("*/recommendation_predictions.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No recommendation_predictions.json files found under {prediction_dir / 'patients'}")

    prediction_records: List[Dict[str, Any]] = []
    prediction_intervals_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    for pred_path in prediction_files:
        payload = _load_json(pred_path)
        predictions = payload.get("recommendation_predictions")
        if not isinstance(predictions, list):
            raise ValueError(f"Missing recommendation_predictions list in {pred_path}")
        icu_stay_id = _extract_icu_stay_id(payload, pred_path)
        prediction_intervals: Set[Tuple[float, float]] = set()
        for item in predictions:
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid prediction row in {pred_path}")
            prediction_intervals.add(_resolve_prediction_interval(item=item, source_path=pred_path))
        if int(icu_stay_id) in prediction_intervals_by_stay:
            raise ValueError(f"Duplicate prediction entries for icu_stay_id={icu_stay_id} in {prediction_dir}")
        prediction_intervals_by_stay[int(icu_stay_id)] = set(prediction_intervals)
        prediction_records.append(
            {
                "pred_path": pred_path,
                "payload": payload,
                "predictions": predictions,
                "icu_stay_id": int(icu_stay_id),
            }
        )

    gt_actions_by_stay: Dict[int, Dict[Tuple[float, float], List[Dict[str, Any]]]] = {}
    gt_red_flags_by_stay: Dict[int, Dict[Tuple[float, float], List[Dict[str, Any]]]] = {}
    gt_intervals_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    for gt_prediction_path in sorted((gt_dir / "patients").glob("*/oracle_predictions.json")):
        gt_payload = _load_json(gt_prediction_path)
        icu_stay_id = _extract_icu_stay_id(gt_payload, gt_prediction_path)
        if int(icu_stay_id) in gt_actions_by_stay:
            raise ValueError(f"Duplicate GT entries for icu_stay_id={icu_stay_id} in {gt_dir}")
        gt_actions, gt_red_flags = _load_gt_targets_by_interval(gt_prediction_path)
        gt_actions_by_stay[int(icu_stay_id)] = gt_actions
        gt_red_flags_by_stay[int(icu_stay_id)] = gt_red_flags
        gt_intervals_by_stay[int(icu_stay_id)] = _load_gt_window_intervals(gt_prediction_path)

    prediction_stays = int(len(prediction_intervals_by_stay))
    gt_stays = int(len(gt_intervals_by_stay))
    prediction_windows_total = int(sum(len(v) for v in prediction_intervals_by_stay.values()))
    gt_windows_total = int(sum(len(v) for v in gt_intervals_by_stay.values()))

    matched_intervals_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    skipped_missing_gt_stays = 0
    skipped_unmatched_prediction_windows = 0
    matched_windows_total = 0
    for icu_stay_id, pred_intervals in prediction_intervals_by_stay.items():
        gt_intervals = gt_intervals_by_stay.get(int(icu_stay_id))
        if gt_intervals is None:
            skipped_missing_gt_stays += 1
            skipped_unmatched_prediction_windows += int(len(pred_intervals))
            continue
        shared_intervals = pred_intervals.intersection(gt_intervals)
        skipped_unmatched_prediction_windows += int(len(pred_intervals) - len(shared_intervals))
        if shared_intervals:
            matched_intervals_by_stay[int(icu_stay_id)] = set(shared_intervals)
            matched_windows_total += int(len(shared_intervals))
    matched_stays = int(len(matched_intervals_by_stay))
    window_match_stats = {
        "prediction_stays": prediction_stays,
        "gt_stays": gt_stays,
        "matched_stays": matched_stays,
        "prediction_windows": prediction_windows_total,
        "gt_windows": gt_windows_total,
        "matched_windows": int(matched_windows_total),
        "skipped_prediction_stays_missing_gt": int(skipped_missing_gt_stays),
        "skipped_prediction_windows_missing_gt": int(skipped_unmatched_prediction_windows),
    }

    tasks: List[Dict[str, Any]] = []
    skipped_no_gt_action_windows = 0
    windows_without_gt_red_flags = 0
    for record in prediction_records:
        pred_path = record["pred_path"]
        payload = record["payload"]
        predictions = record["predictions"]
        patient_id = pred_path.parent.name
        icu_stay_id = int(record["icu_stay_id"])
        matched_intervals = matched_intervals_by_stay.get(int(icu_stay_id))
        if matched_intervals is None:
            continue
        gt_actions_by_interval = gt_actions_by_stay.get(int(icu_stay_id))
        gt_red_flags_by_interval = gt_red_flags_by_stay.get(int(icu_stay_id))
        if gt_actions_by_interval is None or gt_red_flags_by_interval is None:
            continue

        subject_id_raw = payload.get("subject_id")
        if subject_id_raw is None:
            raise ValueError(f"Missing subject_id in {pred_path}")
        subject_id = int(subject_id_raw)

        num_memory_snapshots_raw = payload.get("num_memory_snapshots")
        if num_memory_snapshots_raw is None:
            raise ValueError(f"Missing num_memory_snapshots in {pred_path}")
        num_memory_snapshots = int(num_memory_snapshots_raw)
        if num_memory_snapshots <= 0:
            raise ValueError(f"num_memory_snapshots must be > 0 in {pred_path}")

        top_k_actions_raw = payload.get("top_k_actions")
        if top_k_actions_raw is None:
            raise ValueError(f"Missing top_k_actions in {pred_path}")
        top_k_actions = int(top_k_actions_raw)
        if top_k_actions < 1:
            raise ValueError(f"top_k_actions must be >= 1 in {pred_path}")

        source_patient_dir_raw = payload.get("source_patient_dir")
        if source_patient_dir_raw is None:
            raise ValueError(f"Missing source_patient_dir in {pred_path}")
        source_patient_dir = Path(str(source_patient_dir_raw))
        memory_depth_map = _memory_depth_by_window(source_patient_dir)

        for item in predictions:
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid prediction row in {pred_path}")
            interval_key = _resolve_prediction_interval(item=item, source_path=pred_path)
            if interval_key not in matched_intervals:
                continue

            window_raw = item.get("window_index")
            if window_raw is None:
                raise ValueError(f"Missing window_index in prediction row in {pred_path}")
            window_index = int(window_raw)
            if window_index < 0:
                raise ValueError(f"Negative window_index={window_index} in {pred_path}")

            recommended_actions = item.get("recommended_actions")
            if not isinstance(recommended_actions, list):
                raise ValueError(f"Missing recommended_actions list in {pred_path} window={window_index}")
            prediction_items = [dict(action) for action in recommended_actions if isinstance(action, Mapping)]
            red_flag_gt_items: List[Dict[str, Any]] = []
            if interval_key not in gt_actions_by_interval:
                skipped_no_gt_action_windows += 1
                continue
            gt_items = [dict(action) for action in gt_actions_by_interval[interval_key]]
            red_flag_gt_items = [dict(action) for action in gt_red_flags_by_interval.get(interval_key, [])]
            if not red_flag_gt_items:
                windows_without_gt_red_flags += 1

            num_windows = int(num_memory_snapshots)
            if num_windows > 1:
                relative_time = float(window_index) / float(num_windows - 1)
            else:
                relative_time = 0.0
            relative_time = min(max(relative_time, 0.0), 1.0)

            time_bin = assign_normalized_time_bin(relative_time, num_bins=NUM_TIME_BINS)
            hours_since_admission_raw = item.get("hours_since_admission")
            if hours_since_admission_raw is None:
                raise ValueError(f"Missing hours_since_admission in {pred_path} window={window_index}")
            hours_since_admission = float(hours_since_admission_raw)
            memory_depth = memory_depth_map.get(window_index)

            tasks.append(
                {
                    "patient_id": patient_id,
                    "subject_id": int(subject_id),
                    "icu_stay_id": int(icu_stay_id),
                    "window_index": int(window_index),
                    "num_windows": int(num_windows),
                    "relative_time": float(relative_time),
                    "time_bin": int(time_bin),
                    "hours_since_admission": float(hours_since_admission),
                    "top_k_actions": int(top_k_actions),
                    "num_recommendations": int(len(prediction_items)),
                    "num_ground_truth_actions": int(len(gt_items)),
                    "num_ground_truth_red_flags": int(len(red_flag_gt_items)),
                    "memory_depth": memory_depth,
                    "matcher_backend": str(backend),
                    "oracle_window_marker_id": _interval_marker_id(interval_key),
                    "oracle_interval_start_hour": float(interval_key[0]),
                    "oracle_interval_end_hour": float(interval_key[1]),
                    "prediction_items": prediction_items,
                    "oracle_recommended_gt_items": gt_items,
                    "oracle_red_flag_gt_items": red_flag_gt_items,
                    "pubmedbert_model": str(pubmedbert_model),
                    "pubmedbert_similarity_threshold": float(pubmedbert_similarity_threshold),
                    "pubmedbert_batch_size": int(pubmedbert_batch_size),
                    "pubmedbert_device": str(pubmedbert_device) if pubmedbert_device is not None else None,
                }
            )

    total_tasks = int(len(tasks))
    if total_tasks == 0:
        if skipped_missing_gt_stays > 0:
            raise ValueError(
                "No recommendation prediction windows found for evaluation after skipping "
                f"{skipped_missing_gt_stays} ICU stays without matching GT."
            )
        raise ValueError("No recommendation prediction windows found for evaluation.")
    print(
        f"Loaded recommendation predictions: patients={len(prediction_files)}, "
        f"windows={total_tasks}, workers={int(num_workers)}"
    )
    print(
        "Window matching summary: "
        f"prediction_stays={window_match_stats['prediction_stays']}, "
        f"gt_stays={window_match_stats['gt_stays']}, "
        f"matched_stays={window_match_stats['matched_stays']}, "
        f"matched_windows={window_match_stats['matched_windows']}"
    )
    print(f"Skipped prediction stays with missing GT: {skipped_missing_gt_stays}")
    print(f"Skipped prediction windows not matched in GT: {skipped_unmatched_prediction_windows}")
    print(f"Skipped windows with no GT best_practice/acceptable actions: {skipped_no_gt_action_windows}")
    print(f"Windows with no GT red_flags: {windows_without_gt_red_flags}")

    window_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    matcher_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def _should_log_progress(completed: int, total: int) -> bool:
        return completed <= 5 or completed == total or completed % 100 == 0

    def _finalize_window(
        *,
        completed: int,
        task: Dict[str, Any],
        match_result: Dict[str, Any],
    ) -> None:
        nonlocal matcher_calls, total_input_tokens, total_output_tokens
        if task["prediction_items"] and (task["oracle_recommended_gt_items"] or task["oracle_red_flag_gt_items"]):
            matcher_calls += 1
            total_input_tokens += int(match_result["input_tokens"])
            total_output_tokens += int(match_result["output_tokens"])
        window_row, per_prediction_rows = _build_rows_from_match(task=task, match_result=match_result)
        window_rows.append(window_row)
        prediction_rows.extend(per_prediction_rows)
        if _should_log_progress(completed, total_tasks):
            print(
                f"[{completed}/{total_tasks}] matched patient={task['patient_id']} "
                f"window={task['window_index']} rec={task['num_recommendations']} "
                f"gt={task['num_ground_truth_actions']} matched={len(match_result['recommended_matched_prediction_indices'])} "
                f"red_flags={task['num_ground_truth_red_flags']} "
                f"matched_red_flags={len(match_result['red_flag_matched_prediction_indices'])}"
            )

    if int(num_workers) == 1:
        for completed, task in enumerate(tasks, start=1):
            _, match_result = _run_window_match_task(
                task,
            )
            _finalize_window(completed=completed, task=task, match_result=match_result)
    else:
        with ThreadPoolExecutor(max_workers=min(int(num_workers), total_tasks)) as executor:
            futures = [
                executor.submit(
                    _run_window_match_task,
                    task,
                )
                for task in tasks
            ]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                task, match_result = future.result()
                _finalize_window(completed=completed, task=task, match_result=match_result)

    window_frame = pd.DataFrame(window_rows)
    if window_frame.empty:
        raise ValueError("No recommendation prediction windows found for evaluation.")
    window_frame = window_frame.sort_values(["patient_id", "window_index"]).reset_index(drop=True)
    prediction_frame = pd.DataFrame(prediction_rows)
    if not prediction_frame.empty:
        prediction_frame = prediction_frame.sort_values(
            ["patient_id", "window_index", "prediction_index"]
        ).reset_index(drop=True)

    matcher_usage = {
        "matcher_backend": str(backend),
        "pubmedbert_model": str(pubmedbert_model),
        "pubmedbert_similarity_threshold": float(pubmedbert_similarity_threshold),
        "pubmedbert_batch_size": int(pubmedbert_batch_size),
        "pubmedbert_device": str(pubmedbert_device) if pubmedbert_device is not None else None,
        "matcher_calls": int(matcher_calls),
        "input_tokens": int(total_input_tokens),
        "output_tokens": int(total_output_tokens),
        "total_tokens": int(total_input_tokens + total_output_tokens),
        "window_match_stats": {
            "prediction_stays": int(window_match_stats["prediction_stays"]),
            "gt_stays": int(window_match_stats["gt_stays"]),
            "matched_stays": int(window_match_stats["matched_stays"]),
            "prediction_windows": int(window_match_stats["prediction_windows"]),
            "gt_windows": int(window_match_stats["gt_windows"]),
            "matched_windows": int(window_match_stats["matched_windows"]),
            "skipped_prediction_stays_missing_gt": int(window_match_stats["skipped_prediction_stays_missing_gt"]),
            "skipped_prediction_windows_missing_gt": int(window_match_stats["skipped_prediction_windows_missing_gt"]),
        },
        "skipped_missing_gt_stays": int(skipped_missing_gt_stays),
        "skipped_unmatched_prediction_windows": int(skipped_unmatched_prediction_windows),
        "skipped_no_gt_action_windows": int(skipped_no_gt_action_windows),
        "windows_without_gt_red_flags": int(windows_without_gt_red_flags),
    }
    return window_frame, prediction_frame, matcher_usage


def _load_reused_recommendation_frames(reuse_eval_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    window_path = reuse_eval_dir / "window_level_windows.csv"
    if not window_path.exists():
        raise FileNotFoundError(f"Missing reused matcher window file: {window_path}")
    window_frame = pd.read_csv(window_path)
    if window_frame.empty:
        raise ValueError(f"Reused matcher window file is empty: {window_path}")

    list_columns = (
        "matched_prediction_indices",
        "matched_pairs",
        "matched_red_flag_prediction_indices",
        "matched_red_flag_pairs",
    )
    for column in list_columns:
        if column not in window_frame.columns:
            window_frame[column] = [[] for _ in range(len(window_frame))]
        else:
            window_frame[column] = window_frame[column].map(_parse_json_list_cell)

    prediction_path = reuse_eval_dir / "window_level_predictions.csv"
    prediction_frame = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()

    metrics_path = reuse_eval_dir / "metrics.json"
    metrics_payload = _load_json(metrics_path) if metrics_path.exists() else {}
    matcher_usage = metrics_payload.get("matcher_usage")
    if not isinstance(matcher_usage, dict):
        matcher_usage = {
            "matcher_backend": MATCHER_BACKEND_PUBMEDBERT,
            "pubmedbert_model": PUBMEDBERT_MODEL,
            "pubmedbert_similarity_threshold": float(PUBMEDBERT_SIMILARITY_THRESHOLD),
            "pubmedbert_batch_size": int(PUBMEDBERT_BATCH_SIZE),
            "pubmedbert_device": "cuda",
            "matcher_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "window_match_stats": {},
        }
    matcher_usage = dict(matcher_usage)
    matcher_usage["reused_from"] = str(reuse_eval_dir)
    return window_frame, prediction_frame, matcher_usage


def _write_recommendation_analysis_outputs(
    *,
    window_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    matcher_usage: Dict[str, Any],
    prediction_root: Optional[Path],
    gt_root: Optional[Path],
    output_dir: Path,
    reuse_eval_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    llm_usage = {
        "llm_calls": int(matcher_usage.get("matcher_calls", 0)),
        "input_tokens": int(matcher_usage.get("input_tokens", 0)),
        "output_tokens": int(matcher_usage.get("output_tokens", 0)),
        "total_tokens": int(matcher_usage.get("total_tokens", 0)),
    }

    k_values = [int(FIXED_EVAL_K)]
    selected_k = int(FIXED_EVAL_K)

    window_k_frame = _expand_window_rows_by_k(window_frame, k_values=k_values)
    metrics_by_k_df = _compute_metrics_by_k(window_k_frame)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_row = metrics_by_k_df[metrics_by_k_df["k"] == int(selected_k)].iloc[0]
    metrics_payload = {
        "generated_at": datetime.now().isoformat(),
        "prediction_dir": str(prediction_root) if prediction_root is not None else None,
        "gt_dir": str(gt_root) if gt_root is not None else None,
        "reuse_eval_dir": str(reuse_eval_dir) if reuse_eval_dir is not None else None,
        "fixed_k": int(FIXED_EVAL_K),
        "matcher_usage": matcher_usage,
        "llm_usage": llm_usage,
        "summary": {
            "macro_hit_at_k": float(summary_row["macro_hit_at_k"]),
            "macro_recall_at_k": float(summary_row["macro_recall_at_k"]),
            "red_flag_macro_hit_at_k": float(summary_row["red_flag_macro_hit_at_k"]),
            "harmful_recommendation_rate_case_level_at_k": float(
                summary_row["harmful_recommendation_rate_case_level_at_k"]
            ),
        },
        "metrics_by_k": json.loads(metrics_by_k_df.to_json(orient="records")),
    }
    _json_dump(output_dir / "metrics.json", metrics_payload)

    metrics_by_k_df.to_csv(output_dir / "metrics_by_k.csv", index=False)

    window_export_df = window_frame.copy()
    window_export_df["matched_prediction_indices"] = window_export_df["matched_prediction_indices"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    window_export_df["matched_pairs"] = window_export_df["matched_pairs"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    window_export_df["matched_red_flag_prediction_indices"] = window_export_df[
        "matched_red_flag_prediction_indices"
    ].map(lambda value: json.dumps(value, ensure_ascii=False))
    window_export_df["matched_red_flag_pairs"] = window_export_df["matched_red_flag_pairs"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    window_export_df.to_csv(output_dir / "window_level_windows.csv", index=False)

    prediction_frame.to_csv(output_dir / "window_level_predictions.csv", index=False)
    window_k_frame.to_csv(output_dir / "window_level_metrics.csv", index=False)

    absolute_hour_curve_df = _build_absolute_hour_metric_frame(window_k_frame, selected_k=selected_k)
    absolute_hour_curve_df.to_csv(output_dir / "absolute_hour_recommendation_metrics.csv", index=False)
    absolute_hour_curve_df[
        ["k", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_hit_at_k"]
    ].to_csv(output_dir / "absolute_hour_recommendation_macro_hit_at_k.csv", index=False)
    absolute_hour_curve_df[
        ["k", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_recall_at_k"]
    ].to_csv(output_dir / "absolute_hour_recommendation_macro_recall_at_k.csv", index=False)
    absolute_hour_curve_df[
        [
            "k",
            "absolute_hour_bin",
            "hour_start",
            "hour_end",
            "hour_mid",
            "n_windows",
            "num_red_flag_patients",
            "harmful_recommendation_rate_case_level_at_k",
        ]
    ].to_csv(output_dir / "absolute_hour_red_flags_hrr.csv", index=False)

    selected_metrics = metrics_by_k_df[metrics_by_k_df["k"] == int(selected_k)]
    if selected_metrics.empty:
        raise ValueError(f"Missing selected K metrics for k={selected_k}")
    summary_row = selected_metrics.iloc[0]

    print(f"Saved evaluation outputs to: {output_dir}")
    print("Recommendation Evaluation:")
    print(
        f"  K={selected_k} (fixed) "
        f"macro_hit={float(summary_row['macro_hit_at_k']):.4f} "
        f"macro_recall={float(summary_row['macro_recall_at_k']):.4f}"
    )
    if not pd.isna(summary_row["red_flag_macro_hit_at_k"]):
        print(
            f"  Red-Flag macro_hit={float(summary_row['red_flag_macro_hit_at_k']):.4f} "
            f"hrr={float(summary_row['harmful_recommendation_rate_case_level_at_k']):.4f}"
        )
    return {
        "selected_k": int(selected_k),
        "absolute_hour_curve_df": absolute_hour_curve_df,
        "window_k_frame": window_k_frame,
        "output_dir": output_dir,
        "prediction_root": prediction_root,
    }


def _expand_window_rows_by_k(window_frame: pd.DataFrame, *, k_values: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in window_frame.iterrows():
        num_recommendations = int(row["num_recommendations"])
        matched_prediction_indices = row["matched_prediction_indices"]
        if not isinstance(matched_prediction_indices, list):
            matched_prediction_indices = []
        matched_set = {int(value) for value in matched_prediction_indices}
        matched_red_flag_prediction_indices = row.get("matched_red_flag_prediction_indices")
        if not isinstance(matched_red_flag_prediction_indices, list):
            matched_red_flag_prediction_indices = []
        matched_red_flag_set = {int(value) for value in matched_red_flag_prediction_indices}
        num_ground_truth_red_flags = int(row.get("num_ground_truth_red_flags") or 0)

        base = {
            "patient_id": row["patient_id"],
            "subject_id": int(row["subject_id"]),
            "icu_stay_id": int(row["icu_stay_id"]),
            "window_index": int(row["window_index"]),
            "num_windows": int(row["num_windows"]),
            "relative_time": float(row["relative_time"]),
            "time_bin": int(row["time_bin"]),
            "hours_since_admission": float(row["hours_since_admission"]),
            "memory_depth": row["memory_depth"],
            "num_recommendations": int(num_recommendations),
            "num_ground_truth_actions": int(row["num_ground_truth_actions"]),
            "num_ground_truth_red_flags": int(num_ground_truth_red_flags),
            "top_k_actions": int(row["top_k_actions"]),
        }

        for k in k_values:
            num_considered = min(int(k), int(num_recommendations))
            num_matches_at_k = sum(1 for idx in matched_set if idx < num_considered)
            hit_at_k = int(num_matches_at_k >= int(HIT_MATCH_COUNT_THRESHOLD))
            if num_considered > 0:
                precision_at_k = float(num_matches_at_k) / float(num_considered)
            else:
                precision_at_k = 0.0
            num_ground_truth_actions = int(row["num_ground_truth_actions"])
            recall_at_k = (
                float(num_matches_at_k) / float(num_ground_truth_actions) if num_ground_truth_actions > 0 else 0.0
            )
            num_red_flag_matches_at_k = sum(1 for idx in matched_red_flag_set if idx < num_considered)
            red_flag_hit_at_k: float
            red_flag_precision_at_k: float
            if num_ground_truth_red_flags > 0:
                red_flag_hit_at_k = float(int(num_red_flag_matches_at_k > 0))
                red_flag_precision_at_k = (
                    float(num_red_flag_matches_at_k) / float(num_considered) if num_considered > 0 else 0.0
                )
            else:
                red_flag_hit_at_k = float("nan")
                red_flag_precision_at_k = float("nan")

            rows.append(
                {
                    **base,
                    "k": int(k),
                    "num_recommendations_at_k": int(num_considered),
                    "num_matches_at_k": int(num_matches_at_k),
                    "hit_at_k": int(hit_at_k),
                    "precision_at_k": float(precision_at_k),
                    "recall_at_k": float(recall_at_k),
                    "num_red_flag_matches_at_k": int(num_red_flag_matches_at_k),
                    "red_flag_hit_at_k": float(red_flag_hit_at_k),
                    "red_flag_precision_at_k": float(red_flag_precision_at_k),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No K-expanded rows generated.")
    return frame


def _bootstrap_mean_ci(values: Sequence[float], *, n_samples: int, rng: np.random.Generator) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        value = float(arr[0])
        return value, value
    samples = rng.choice(arr, size=(n_samples, len(arr)), replace=True)
    means = samples.mean(axis=1)
    low = float(np.quantile(means, 0.025))
    high = float(np.quantile(means, 0.975))
    return low, high


def _smooth_metric_and_band(
    mean_values: Sequence[float],
    low_values: Sequence[float],
    high_values: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_arr = smooth_series(mean_values, window=CURVE_SMOOTHING_WINDOW)
    low_arr = smooth_series(low_values, window=CURVE_SMOOTHING_WINDOW)
    high_arr = smooth_series(high_values, window=CURVE_SMOOTHING_WINDOW)
    low_safe = np.minimum(low_arr, high_arr)
    high_safe = np.maximum(low_arr, high_arr)
    low_safe = np.clip(low_safe, 0.0, 1.0)
    high_safe = np.clip(high_safe, 0.0, 1.0)
    mean_arr = np.clip(mean_arr, 0.0, 1.0)
    return mean_arr, low_safe, high_safe


def _band_from_mean_std(
    mean_values: Sequence[float],
    std_values: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    mean_arr = np.asarray(mean_values, dtype=float)
    std_arr = np.asarray(std_values, dtype=float)
    low = np.clip(mean_arr - std_arr, 0.0, 1.0)
    high = np.clip(mean_arr + std_arr, 0.0, 1.0)
    return low, high


def _build_absolute_hour_metric_frame(window_k_frame: pd.DataFrame, *, selected_k: int) -> pd.DataFrame:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")
    subset, max_bin, _ = _absolute_hour_bin_frame(subset)
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for hour_bin in range(max_bin + 1):
        bin_rows = subset[subset["absolute_hour_bin"] == int(hour_bin)].copy()
        if bin_rows.empty:
            macro_hit = float("nan")
            macro_recall = float("nan")
            harmful_recommendation_rate_case_level = float("nan")
            harmful_recommendation_rate_case_level_ci_low = float("nan")
            harmful_recommendation_rate_case_level_ci_high = float("nan")
            num_patients = 0
            num_red_flag_patients = 0
        else:
            by_patient = bin_rows.groupby("patient_id").agg(
                hit_at_k=("hit_at_k", "mean"),
                recall_at_k=("recall_at_k", "mean"),
            )
            macro_hit = float(by_patient["hit_at_k"].mean()) if not by_patient.empty else float("nan")
            macro_recall = float(by_patient["recall_at_k"].mean()) if not by_patient.empty else float("nan")
            red_flag_rows = bin_rows.dropna(subset=["red_flag_hit_at_k"])
            if red_flag_rows.empty:
                harmful_recommendation_rate_case_level = float("nan")
                harmful_recommendation_rate_case_level_ci_low = float("nan")
                harmful_recommendation_rate_case_level_ci_high = float("nan")
                num_red_flag_patients = 0
            else:
                red_flag_case_totals = red_flag_rows.groupby("patient_id").agg(
                    total_red_flag_matches_at_k=("num_red_flag_matches_at_k", "sum"),
                    total_red_flag_recommendations_at_k=("num_recommendations_at_k", "sum"),
                )
                valid_red_flag_case_totals = red_flag_case_totals[
                    red_flag_case_totals["total_red_flag_recommendations_at_k"] > 0
                ]
                if valid_red_flag_case_totals.empty:
                    harmful_recommendation_rate_case_level = float("nan")
                    harmful_recommendation_rate_case_level_ci_low = float("nan")
                    harmful_recommendation_rate_case_level_ci_high = float("nan")
                    num_red_flag_patients = 0
                else:
                    red_flag_case_level_ratio = (
                        valid_red_flag_case_totals["total_red_flag_matches_at_k"].astype(float)
                        / valid_red_flag_case_totals["total_red_flag_recommendations_at_k"].astype(float)
                    )
                    harmful_recommendation_rate_case_level = float(red_flag_case_level_ratio.mean())
                    (
                        harmful_recommendation_rate_case_level_ci_low,
                        harmful_recommendation_rate_case_level_ci_high,
                    ) = _bootstrap_mean_ci(
                        red_flag_case_level_ratio.astype(float).tolist(),
                        n_samples=BOOTSTRAP_SAMPLES,
                        rng=rng,
                    )
                    num_red_flag_patients = int(valid_red_flag_case_totals.shape[0])
            num_patients = int(by_patient.shape[0])
        rows.append(
            {
                "k": int(selected_k),
                "absolute_hour_bin": int(hour_bin),
                "hour_start": float(hour_bin * float(ABSOLUTE_TIME_BIN_HOURS)),
                "hour_end": float((hour_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS)),
                "hour_mid": float((hour_bin + 0.5) * float(ABSOLUTE_TIME_BIN_HOURS)),
                "n_windows": int(len(bin_rows)),
                "num_patients": int(num_patients),
                "num_red_flag_patients": int(num_red_flag_patients),
                "macro_hit_at_k": float(macro_hit),
                "macro_recall_at_k": float(macro_recall),
                "harmful_recommendation_rate_case_level_at_k": float(harmful_recommendation_rate_case_level),
                "harmful_recommendation_rate_case_level_at_k_ci_low": float(
                    harmful_recommendation_rate_case_level_ci_low
                ),
                "harmful_recommendation_rate_case_level_at_k_ci_high": float(
                    harmful_recommendation_rate_case_level_ci_high
                ),
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("absolute_hour_bin").reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No absolute-hour recommendation metric rows generated.")
    return plot_df


def _plot_absolute_hour_metric_frame(
    curve_df: pd.DataFrame,
    *,
    metric_column: str,
    metric_label: str,
    metric_key: str,
    title: str,
    output_path: Path,
    group_column: Optional[str] = None,
) -> None:
    if curve_df.empty:
        raise ValueError(f"No rows available for {metric_column} plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    max_hour = 0.0
    if group_column is None:
        ax.plot(
            curve_df["hour_mid"].to_numpy(dtype=float),
            curve_df[metric_column].to_numpy(dtype=float),
            linewidth=2.4,
            color=metric_color(metric_key),
            marker="o",
        )
        max_hour = float(curve_df["hour_end"].max())
    else:
        color_by_group = mode_color_map(sorted(curve_df[group_column].astype(str).unique().tolist()))
        for group_name in sorted(curve_df[group_column].astype(str).unique().tolist()):
            subset = curve_df[curve_df[group_column].astype(str) == str(group_name)].sort_values("absolute_hour_bin")
            if subset.empty:
                continue
            ax.plot(
                subset["hour_mid"].to_numpy(dtype=float),
                subset[metric_column].to_numpy(dtype=float),
                linewidth=2.2,
                color=color_by_group[str(group_name)],
                marker="o",
                alpha=MODE_TRACE_ALPHA,
                label=str(group_name),
            )
            max_hour = max(max_hour, float(subset["hour_end"].max()))
        ax.legend()

    ax.set_xlabel("Hours Since ICU Admission")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.set_xlim(0, max_hour if max_hour > 0 else float(ABSOLUTE_TIME_BIN_HOURS))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_metrics_by_k(window_k_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for k, subset in window_k_frame.groupby("k"):
        num_windows = int(len(subset))
        micro_hit = float(subset["hit_at_k"].mean()) if num_windows > 0 else float("nan")
        micro_precision = float(subset["precision_at_k"].mean()) if num_windows > 0 else float("nan")
        micro_recall = float(subset["recall_at_k"].mean()) if num_windows > 0 else float("nan")

        patient_hit = subset.groupby("patient_id")["hit_at_k"].mean().astype(float)
        patient_precision = subset.groupby("patient_id")["precision_at_k"].mean().astype(float)
        patient_recall = subset.groupby("patient_id")["recall_at_k"].mean().astype(float)
        macro_hit = float(patient_hit.mean()) if len(patient_hit) > 0 else float("nan")
        macro_precision = float(patient_precision.mean()) if len(patient_precision) > 0 else float("nan")
        macro_recall = float(patient_recall.mean()) if len(patient_recall) > 0 else float("nan")
        red_flag_subset = subset.dropna(subset=["red_flag_hit_at_k", "red_flag_precision_at_k"])
        red_flag_windows = int(len(red_flag_subset))
        if red_flag_windows > 0:
            red_flag_patient_hit = red_flag_subset.groupby("patient_id")["red_flag_hit_at_k"].mean().astype(float)
            red_flag_patient_precision = (
                red_flag_subset.groupby("patient_id")["red_flag_precision_at_k"].mean().astype(float)
            )
            red_flag_macro_hit = float(red_flag_patient_hit.mean()) if len(red_flag_patient_hit) > 0 else float("nan")
            red_flag_macro_precision = (
                float(red_flag_patient_precision.mean()) if len(red_flag_patient_precision) > 0 else float("nan")
            )
            red_flag_case_totals = red_flag_subset.groupby("patient_id").agg(
                total_red_flag_matches_at_k=("num_red_flag_matches_at_k", "sum"),
                total_red_flag_recommendations_at_k=("num_recommendations_at_k", "sum"),
            )
            valid_red_flag_case_totals = red_flag_case_totals[
                red_flag_case_totals["total_red_flag_recommendations_at_k"] > 0
            ]
            if len(valid_red_flag_case_totals) > 0:
                red_flag_case_level_ratio = (
                    valid_red_flag_case_totals["total_red_flag_matches_at_k"].astype(float)
                    / valid_red_flag_case_totals["total_red_flag_recommendations_at_k"].astype(float)
                )
                harmful_recommendation_rate_case_level = float(red_flag_case_level_ratio.mean())
            else:
                harmful_recommendation_rate_case_level = float("nan")
        else:
            red_flag_macro_hit = float("nan")
            red_flag_macro_precision = float("nan")
            harmful_recommendation_rate_case_level = float("nan")

        rows.append(
            {
                "k": int(k),
                "num_windows": int(num_windows),
                "num_patients": int(subset["patient_id"].nunique()),
                "micro_hit_at_k": float(micro_hit),
                "macro_hit_at_k": float(macro_hit),
                "micro_precision_at_k": float(micro_precision),
                "macro_precision_at_k": float(macro_precision),
                "micro_recall_at_k": float(micro_recall),
                "macro_recall_at_k": float(macro_recall),
                "total_matches_at_k": int(subset["num_matches_at_k"].sum()),
                "total_recommendations_at_k": int(subset["num_recommendations_at_k"].sum()),
                "num_windows_with_red_flags": int(red_flag_windows),
                "num_patients_with_red_flags": int(red_flag_subset["patient_id"].nunique()),
                "red_flag_macro_hit_at_k": float(red_flag_macro_hit),
                "red_flag_macro_precision_at_k": float(red_flag_macro_precision),
                "harmful_recommendation_rate_case_level_at_k": float(harmful_recommendation_rate_case_level),
                "total_red_flag_matches_at_k": int(red_flag_subset["num_red_flag_matches_at_k"].sum()),
                "total_red_flag_recommendations_at_k": int(red_flag_subset["num_recommendations_at_k"].sum()),
            }
        )
    metrics_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if metrics_df.empty:
        raise ValueError("No K-level metrics computed.")
    return metrics_df


def _memory_depth_bins(values: pd.Series) -> Tuple[pd.Series, List[str]]:
    valid = values.dropna().astype(float)
    if valid.empty:
        raise ValueError("Cannot build memory-depth bins from empty values.")
    unique = np.unique(valid.to_numpy(dtype=float))
    if len(unique) <= MEMORY_DEPTH_NUM_BINS:
        sorted_unique = sorted(int(value) for value in unique.tolist())
        index_map = {value: idx for idx, value in enumerate(sorted_unique)}
        mapped = values.map(lambda x: index_map.get(int(float(x))) if pd.notna(x) else pd.NA)
        labels = [f"{value}" for value in sorted_unique]
        return mapped.astype("Int64"), labels

    quantile_bins = pd.qcut(valid, q=MEMORY_DEPTH_NUM_BINS, labels=False, duplicates="drop")
    if quantile_bins.isna().all():
        raise ValueError("Failed to create quantile memory-depth bins.")

    bin_index_by_row = pd.Series(index=valid.index, data=quantile_bins.astype(int).to_numpy())
    full_codes = pd.Series(pd.NA, index=values.index, dtype="Int64")
    full_codes.loc[bin_index_by_row.index] = bin_index_by_row.astype("Int64")

    labels: List[str] = []
    for bin_id in sorted(bin_index_by_row.unique().tolist()):
        selected = valid[bin_index_by_row == bin_id]
        left = int(math.floor(float(selected.min())))
        right = int(math.ceil(float(selected.max())))
        labels.append(f"{left}-{right}")
    return full_codes.astype("Int64"), labels


def _plot_relative_time_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_path: Path,
) -> pd.DataFrame:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for time_bin in range(NUM_TIME_BINS):
        bin_rows = subset[subset["time_bin"] == int(time_bin)]
        hit_values = bin_rows["hit_at_k"].astype(float).tolist()
        precision_values = bin_rows["precision_at_k"].astype(float).tolist()

        hit_mean = float(np.mean(hit_values)) if hit_values else float("nan")
        precision_mean = float(np.mean(precision_values)) if precision_values else float("nan")
        hit_ci_low, hit_ci_high = _bootstrap_mean_ci(hit_values, n_samples=BOOTSTRAP_SAMPLES, rng=rng)
        precision_ci_low, precision_ci_high = _bootstrap_mean_ci(
            precision_values,
            n_samples=BOOTSTRAP_SAMPLES,
            rng=rng,
        )

        rows.append(
            {
                "k": int(selected_k),
                "time_bin": int(time_bin),
                "time_bin_mid_pct": float((time_bin + 0.5) * (100.0 / NUM_TIME_BINS)),
                "n_windows": int(len(bin_rows)),
                "hit_at_k": float(hit_mean),
                "hit_ci_low": float(hit_ci_low),
                "hit_ci_high": float(hit_ci_high),
                "precision_at_k": float(precision_mean),
                "precision_ci_low": float(precision_ci_low),
                "precision_ci_high": float(precision_ci_high),
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = plot_df["time_bin_mid_pct"].to_numpy(dtype=float)

    y_hit = plot_df["hit_at_k"].to_numpy(dtype=float)
    low_hit = plot_df["hit_ci_low"].to_numpy(dtype=float)
    high_hit = plot_df["hit_ci_high"].to_numpy(dtype=float)
    y_hit_smooth, low_hit_smooth, high_hit_smooth = _smooth_metric_and_band(y_hit, low_hit, high_hit)
    axes[0].plot(x, y_hit_smooth, linewidth=2.4, color=metric_color("hit"))
    axes[0].fill_between(x, low_hit_smooth, high_hit_smooth, alpha=METRIC_BAND_ALPHA, color=metric_color("hit"))
    axes[0].set_xlabel("Relative Window Position (%)")
    axes[0].set_ylabel(f"Hit Rate")
    axes[0].set_title(f"Relative-Time Hit Rate")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    y_precision = plot_df["precision_at_k"].to_numpy(dtype=float)
    low_precision = plot_df["precision_ci_low"].to_numpy(dtype=float)
    high_precision = plot_df["precision_ci_high"].to_numpy(dtype=float)
    y_precision_smooth, low_precision_smooth, high_precision_smooth = _smooth_metric_and_band(
        y_precision,
        low_precision,
        high_precision,
    )
    axes[1].plot(x, y_precision_smooth, linewidth=2.4, color=metric_color("precision"))
    axes[1].fill_between(
        x,
        low_precision_smooth,
        high_precision_smooth,
        alpha=METRIC_BAND_ALPHA,
        color=metric_color("precision"),
    )
    axes[1].set_xlabel("Relative Window Position (%)")
    axes[1].set_ylabel(f"Precision")
    axes[1].set_title(f"Relative-Time Precision")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _plot_prefix_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_path: Path,
) -> pd.DataFrame:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")

    subset = subset.sort_values(["patient_id", "relative_time", "window_index"]).reset_index(drop=True)
    subset["cum_hit_sum"] = subset.groupby("patient_id")["hit_at_k"].cumsum()
    subset["cum_precision_sum"] = subset.groupby("patient_id")["precision_at_k"].cumsum()
    subset["cum_count"] = subset.groupby("patient_id").cumcount() + 1
    subset["prefix_hit_at_k"] = subset["cum_hit_sum"] / subset["cum_count"]
    subset["prefix_precision_at_k"] = subset["cum_precision_sum"] / subset["cum_count"]

    thresholds = np.linspace(0.0, 1.0, NUM_TIME_BINS + 1)
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        patient_hit_values: List[float] = []
        patient_precision_values: List[float] = []

        for _, patient_df in subset.groupby("patient_id"):
            valid = patient_df[patient_df["relative_time"] <= threshold]
            if valid.empty:
                continue
            patient_hit_values.append(float(valid["prefix_hit_at_k"].iloc[-1]))
            patient_precision_values.append(float(valid["prefix_precision_at_k"].iloc[-1]))

        rows.append(
            {
                "k": int(selected_k),
                "relative_time_pct": float(threshold * 100.0),
                "num_patients": int(len(patient_hit_values)),
                "cohort_mean_prefix_hit_at_k": (
                    float(np.mean(patient_hit_values)) if patient_hit_values else float("nan")
                ),
                "cohort_mean_prefix_precision_at_k": (
                    float(np.mean(patient_precision_values)) if patient_precision_values else float("nan")
                ),
                "cohort_std_prefix_hit_at_k": (
                    float(np.std(np.asarray(patient_hit_values, dtype=float), ddof=0))
                    if patient_hit_values
                    else float("nan")
                ),
                "cohort_std_prefix_precision_at_k": (
                    float(np.std(np.asarray(patient_precision_values, dtype=float), ddof=0))
                    if patient_precision_values
                    else float("nan")
                ),
            }
        )

    mean_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for _, patient_df in subset.groupby("patient_id"):
        x = patient_df["relative_time"].to_numpy(dtype=float) * 100.0
        hit_y = smooth_series(patient_df["prefix_hit_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        precision_y = smooth_series(
            patient_df["prefix_precision_at_k"].to_numpy(dtype=float),
            window=CURVE_SMOOTHING_WINDOW,
        )
        axes[0].plot(x, hit_y, color=metric_color("hit"), alpha=PATIENT_TRACE_ALPHA, linewidth=1.0)
        axes[1].plot(
            x,
            precision_y,
            color=metric_color("precision"),
            alpha=PATIENT_TRACE_ALPHA,
            linewidth=1.0,
        )

    x_mean = mean_df["relative_time_pct"].to_numpy(dtype=float)
    hit_mean = mean_df["cohort_mean_prefix_hit_at_k"].to_numpy(dtype=float)
    hit_std = mean_df["cohort_std_prefix_hit_at_k"].to_numpy(dtype=float)
    hit_low, hit_high = _band_from_mean_std(hit_mean, hit_std)
    hit_mean_smooth, hit_low_smooth, hit_high_smooth = _smooth_metric_and_band(hit_mean, hit_low, hit_high)
    axes[0].fill_between(
        x_mean,
        hit_low_smooth,
        hit_high_smooth,
        color=metric_mean_color("hit"),
        alpha=METRIC_BAND_ALPHA,
    )
    axes[0].plot(
        x_mean,
        hit_mean_smooth,
        color=metric_mean_color("hit"),
        linewidth=3.0,
        label=f"Cohort Mean Hit Rate",
    )
    axes[0].set_xlabel("Relative Time (%)")
    axes[0].set_ylabel(f"Cumulative Hit Rate")
    axes[0].set_title(f"Prefix Cumulative Hit Rate")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    precision_mean = mean_df["cohort_mean_prefix_precision_at_k"].to_numpy(dtype=float)
    precision_std = mean_df["cohort_std_prefix_precision_at_k"].to_numpy(dtype=float)
    precision_low, precision_high = _band_from_mean_std(precision_mean, precision_std)
    precision_mean_smooth, precision_low_smooth, precision_high_smooth = _smooth_metric_and_band(
        precision_mean,
        precision_low,
        precision_high,
    )
    axes[1].fill_between(
        x_mean,
        precision_low_smooth,
        precision_high_smooth,
        color=metric_mean_color("precision"),
        alpha=METRIC_BAND_ALPHA,
    )
    axes[1].plot(
        x_mean,
        precision_mean_smooth,
        color=metric_mean_color("precision"),
        linewidth=3.0,
        label=f"Cohort Mean Precision",
    )
    axes[1].set_xlabel("Relative Time (%)")
    axes[1].set_ylabel(f"Cumulative Precision")
    axes[1].set_title(f"Prefix Cumulative Precision")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return mean_df


def _absolute_hour_bin_frame(subset: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    if "hours_since_admission" not in subset.columns:
        raise ValueError("Absolute-hour plots require hours_since_admission column.")

    frame = subset.copy()
    frame["hours_since_admission"] = pd.to_numeric(frame["hours_since_admission"], errors="coerce")
    frame = frame.dropna(subset=["hours_since_admission"]).copy()
    if frame.empty:
        raise ValueError("Absolute-hour plots require non-empty hours_since_admission values.")
    if (frame["hours_since_admission"] < 0).any():
        raise ValueError("Absolute-hour plots require non-negative hours_since_admission values.")

    max_hour = float(frame["hours_since_admission"].max())
    max_bin = int(math.floor(max_hour / float(ABSOLUTE_TIME_BIN_HOURS)))
    max_edge = float((max_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS))
    frame["absolute_hour_bin"] = np.floor(
        frame["hours_since_admission"].to_numpy(dtype=float) / float(ABSOLUTE_TIME_BIN_HOURS)
    ).astype(int)
    return frame, max_bin, max_edge


def _plot_absolute_hour_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_path: Path,
) -> pd.DataFrame:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")

    subset, max_bin, max_edge = _absolute_hour_bin_frame(subset)

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for hour_bin in range(max_bin + 1):
        bin_rows = subset[subset["absolute_hour_bin"] == int(hour_bin)]
        hit_values = bin_rows["hit_at_k"].astype(float).tolist()
        precision_values = bin_rows["precision_at_k"].astype(float).tolist()
        hit_ci_low, hit_ci_high = _bootstrap_mean_ci(hit_values, n_samples=BOOTSTRAP_SAMPLES, rng=rng)
        precision_ci_low, precision_ci_high = _bootstrap_mean_ci(
            precision_values,
            n_samples=BOOTSTRAP_SAMPLES,
            rng=rng,
        )

        rows.append(
            {
                "k": int(selected_k),
                "absolute_hour_bin": int(hour_bin),
                "hour_start": float(hour_bin * float(ABSOLUTE_TIME_BIN_HOURS)),
                "hour_end": float((hour_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS)),
                "hour_mid": float((hour_bin + 0.5) * float(ABSOLUTE_TIME_BIN_HOURS)),
                "n_windows": int(len(bin_rows)),
                "hit_at_k": float(np.mean(hit_values)) if hit_values else float("nan"),
                "hit_ci_low": float(hit_ci_low),
                "hit_ci_high": float(hit_ci_high),
                "precision_at_k": float(np.mean(precision_values)) if precision_values else float("nan"),
                "precision_ci_low": float(precision_ci_low),
                "precision_ci_high": float(precision_ci_high),
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("absolute_hour_bin").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = plot_df["hour_mid"].to_numpy(dtype=float)

    y_hit = plot_df["hit_at_k"].to_numpy(dtype=float)
    low_hit = plot_df["hit_ci_low"].to_numpy(dtype=float)
    high_hit = plot_df["hit_ci_high"].to_numpy(dtype=float)
    y_hit_smooth, low_hit_smooth, high_hit_smooth = _smooth_metric_and_band(y_hit, low_hit, high_hit)
    axes[0].plot(x, y_hit_smooth, linewidth=2.4, color=metric_color("hit"))
    axes[0].fill_between(x, low_hit_smooth, high_hit_smooth, alpha=METRIC_BAND_ALPHA, color=metric_color("hit"))
    axes[0].set_xlabel("Hours Since ICU Admission")
    axes[0].set_ylabel(f"Hit Rate")
    axes[0].set_title(f"Absolute-Time Hit Rate")
    axes[0].set_xlim(0, max_edge)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    y_precision = plot_df["precision_at_k"].to_numpy(dtype=float)
    low_precision = plot_df["precision_ci_low"].to_numpy(dtype=float)
    high_precision = plot_df["precision_ci_high"].to_numpy(dtype=float)
    y_precision_smooth, low_precision_smooth, high_precision_smooth = _smooth_metric_and_band(
        y_precision,
        low_precision,
        high_precision,
    )
    axes[1].plot(x, y_precision_smooth, linewidth=2.4, color=metric_color("precision"))
    axes[1].fill_between(
        x,
        low_precision_smooth,
        high_precision_smooth,
        alpha=METRIC_BAND_ALPHA,
        color=metric_color("precision"),
    )
    axes[1].set_xlabel("Hours Since ICU Admission")
    axes[1].set_ylabel(f"Precision")
    axes[1].set_title(f"Absolute-Time Precision")
    axes[1].set_xlim(0, max_edge)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _plot_absolute_hour_prefix_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_path: Path,
) -> pd.DataFrame:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")

    subset, max_bin, max_edge = _absolute_hour_bin_frame(subset)
    subset = subset.sort_values(["patient_id", "hours_since_admission", "window_index"]).reset_index(drop=True)
    subset["cum_hit_sum"] = subset.groupby("patient_id")["hit_at_k"].cumsum()
    subset["cum_precision_sum"] = subset.groupby("patient_id")["precision_at_k"].cumsum()
    subset["cum_count"] = subset.groupby("patient_id").cumcount() + 1
    subset["prefix_hit_at_k"] = subset["cum_hit_sum"] / subset["cum_count"]
    subset["prefix_precision_at_k"] = subset["cum_precision_sum"] / subset["cum_count"]

    rows: List[Dict[str, Any]] = []
    thresholds = [float(index * float(ABSOLUTE_TIME_BIN_HOURS)) for index in range(max_bin + 2)]
    for threshold in thresholds:
        patient_hit_values: List[float] = []
        patient_precision_values: List[float] = []

        for _, patient_df in subset.groupby("patient_id"):
            valid = patient_df[patient_df["hours_since_admission"] <= float(threshold)]
            if valid.empty:
                continue
            patient_hit_values.append(float(valid["prefix_hit_at_k"].iloc[-1]))
            patient_precision_values.append(float(valid["prefix_precision_at_k"].iloc[-1]))

        rows.append(
            {
                "k": int(selected_k),
                "hours_since_admission": float(threshold),
                "num_patients": int(len(patient_hit_values)),
                "cohort_mean_prefix_hit_at_k": (
                    float(np.mean(patient_hit_values)) if patient_hit_values else float("nan")
                ),
                "cohort_mean_prefix_precision_at_k": (
                    float(np.mean(patient_precision_values)) if patient_precision_values else float("nan")
                ),
                "cohort_std_prefix_hit_at_k": (
                    float(np.std(np.asarray(patient_hit_values, dtype=float), ddof=0))
                    if patient_hit_values
                    else float("nan")
                ),
                "cohort_std_prefix_precision_at_k": (
                    float(np.std(np.asarray(patient_precision_values, dtype=float), ddof=0))
                    if patient_precision_values
                    else float("nan")
                ),
            }
        )

    mean_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for _, patient_df in subset.groupby("patient_id"):
        x = patient_df["hours_since_admission"].to_numpy(dtype=float)
        hit_y = smooth_series(patient_df["prefix_hit_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        precision_y = smooth_series(
            patient_df["prefix_precision_at_k"].to_numpy(dtype=float),
            window=CURVE_SMOOTHING_WINDOW,
        )
        axes[0].plot(x, hit_y, color=metric_color("hit"), alpha=PATIENT_TRACE_ALPHA, linewidth=1.0)
        axes[1].plot(
            x,
            precision_y,
            color=metric_color("precision"),
            alpha=PATIENT_TRACE_ALPHA,
            linewidth=1.0,
        )

    x_mean = mean_df["hours_since_admission"].to_numpy(dtype=float)
    hit_mean = mean_df["cohort_mean_prefix_hit_at_k"].to_numpy(dtype=float)
    hit_std = mean_df["cohort_std_prefix_hit_at_k"].to_numpy(dtype=float)
    hit_low, hit_high = _band_from_mean_std(hit_mean, hit_std)
    hit_mean_smooth, hit_low_smooth, hit_high_smooth = _smooth_metric_and_band(hit_mean, hit_low, hit_high)
    axes[0].fill_between(
        x_mean,
        hit_low_smooth,
        hit_high_smooth,
        color=metric_mean_color("hit"),
        alpha=METRIC_BAND_ALPHA,
    )
    axes[0].plot(
        x_mean,
        hit_mean_smooth,
        color=metric_mean_color("hit"),
        linewidth=3.0,
        label=f"Cohort Mean Hit Rate",
    )
    axes[0].set_xlabel("Hours Since ICU Admission")
    axes[0].set_ylabel(f"Cumulative Hit Rate")
    axes[0].set_title(f"Absolute-Time Prefix Hit Rate")
    axes[0].set_xlim(0, max_edge)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    precision_mean = mean_df["cohort_mean_prefix_precision_at_k"].to_numpy(dtype=float)
    precision_std = mean_df["cohort_std_prefix_precision_at_k"].to_numpy(dtype=float)
    precision_low, precision_high = _band_from_mean_std(precision_mean, precision_std)
    precision_mean_smooth, precision_low_smooth, precision_high_smooth = _smooth_metric_and_band(
        precision_mean,
        precision_low,
        precision_high,
    )
    axes[1].fill_between(
        x_mean,
        precision_low_smooth,
        precision_high_smooth,
        color=metric_mean_color("precision"),
        alpha=METRIC_BAND_ALPHA,
    )
    axes[1].plot(
        x_mean,
        precision_mean_smooth,
        color=metric_mean_color("precision"),
        linewidth=3.0,
        label=f"Cohort Mean Precision",
    )
    axes[1].set_xlabel("Hours Since ICU Admission")
    axes[1].set_ylabel(f"Cumulative Precision")
    axes[1].set_title(f"Absolute-Time Prefix Precision")
    axes[1].set_xlim(0, max_edge)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return mean_df


def _plot_time_memory_heatmap(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")
    if subset["memory_depth"].isna().all():
        raise ValueError("Time-memory heatmap requires non-empty memory depth values.")

    bin_codes, bin_labels = _memory_depth_bins(subset["memory_depth"])
    subset["memory_depth_bin"] = bin_codes
    valid = subset.dropna(subset=["memory_depth_bin"]).copy()
    valid["memory_depth_bin"] = valid["memory_depth_bin"].astype(int)

    hit_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="hit_at_k",
            aggfunc="mean",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
    )
    precision_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="precision_at_k",
            aggfunc="mean",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
    )
    count_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="hit_at_k",
            aggfunc="count",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
        .fillna(0)
        .astype(int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    hit_data = hit_table.to_numpy(dtype=float)
    hit_image = axes[0].imshow(hit_data, aspect="auto", origin="lower", cmap=heatmap_cmap("hit"), vmin=0.0, vmax=1.0)
    hit_cbar = fig.colorbar(hit_image, ax=axes[0])
    hit_cbar.set_label(f"Hit Rate")

    precision_data = precision_table.to_numpy(dtype=float)
    precision_image = axes[1].imshow(
        precision_data,
        aspect="auto",
        origin="lower",
        cmap=heatmap_cmap("precision"),
        vmin=0.0,
        vmax=1.0,
    )
    precision_cbar = fig.colorbar(precision_image, ax=axes[1])
    precision_cbar.set_label(f"Precision")

    x_tick_labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(NUM_TIME_BINS)]
    y_tick_labels = [bin_labels[i] if i < len(bin_labels) else str(i) for i in hit_table.index.tolist()]

    for axis, title in (
        (axes[0], f"Time × Memory Hit Rate"),
        (axes[1], f"Time × Memory Precision"),
    ):
        axis.set_xticks(np.arange(NUM_TIME_BINS))
        axis.set_xticklabels(x_tick_labels, rotation=30, ha="right")
        axis.set_yticks(np.arange(len(y_tick_labels)))
        axis.set_yticklabels(y_tick_labels)
        axis.set_xlabel("Relative-Time Decile")
        axis.set_title(title)
    axes[0].set_ylabel("Memory-Depth Bin")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    hit_long = hit_table.reset_index().melt(id_vars=["memory_depth_bin"], var_name="time_bin", value_name="hit_at_k")
    precision_long = precision_table.reset_index().melt(
        id_vars=["memory_depth_bin"], var_name="time_bin", value_name="precision_at_k"
    )
    count_long = count_table.reset_index().melt(id_vars=["memory_depth_bin"], var_name="time_bin", value_name="count")

    merged = hit_long.merge(precision_long, on=["memory_depth_bin", "time_bin"], how="outer")
    merged = merged.merge(count_long, on=["memory_depth_bin", "time_bin"], how="left")
    merged["k"] = int(selected_k)
    merged["memory_depth_label"] = merged["memory_depth_bin"].map(
        lambda value: bin_labels[int(value)] if int(value) < len(bin_labels) else str(value)
    )

    source = valid[
        [
            "patient_id",
            "window_index",
            "k",
            "time_bin",
            "memory_depth",
            "hit_at_k",
            "precision_at_k",
        ]
    ].copy()
    return merged, source


def _plot_patient_level_figures(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_dir: Path,
) -> Tuple[int, int]:
    patient_plot_root = output_dir / "patient_plots"
    patient_plot_root.mkdir(parents=True, exist_ok=True)

    patient_ids = sorted(str(value) for value in window_k_frame["patient_id"].dropna().unique().tolist())
    if not patient_ids:
        return 0, 0

    heatmap_generated = 0
    for patient_id in patient_ids:
        patient_subset = window_k_frame[window_k_frame["patient_id"].astype(str) == str(patient_id)].copy()
        if patient_subset.empty:
            continue
        safe_patient_id = str(patient_id).replace("/", "_")
        patient_dir = patient_plot_root / safe_patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        relative_curve_df = _plot_relative_time_curve(
            patient_subset,
            selected_k=selected_k,
            output_path=patient_dir / "plot1_relative_time_hit_precision_curve.png",
        )
        relative_curve_df.to_csv(patient_dir / "plot1_relative_time_hit_precision_curve.csv", index=False)

        prefix_curve_df = _plot_prefix_curve(
            patient_subset,
            selected_k=selected_k,
            output_path=patient_dir / "plot2_prefix_cumulative_hit_precision_curve.png",
        )
        prefix_curve_df.to_csv(patient_dir / "plot2_prefix_cumulative_hit_precision_curve.csv", index=False)

        try:
            heatmap_long_df, heatmap_source_df = _plot_time_memory_heatmap(
                patient_subset,
                selected_k=selected_k,
                output_path=patient_dir / "plot3_time_memory_hit_precision_heatmap.png",
            )
        except ValueError as exc:
            with open(patient_dir / "plot3_time_memory_hit_precision_heatmap_skipped.txt", "w", encoding="utf-8") as f:
                f.write(str(exc))
        else:
            heatmap_generated += 1
            heatmap_long_df.to_csv(patient_dir / "plot3_time_memory_hit_precision_heatmap.csv", index=False)
            heatmap_source_df.to_csv(patient_dir / "plot3_time_memory_source_rows.csv", index=False)

    return len(patient_ids), int(heatmap_generated)


def _build_patient_relative_time_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    patient_id: str,
) -> pd.DataFrame:
    subset = window_k_frame[
        (window_k_frame["k"] == int(selected_k)) & (window_k_frame["patient_id"].astype(str) == str(patient_id))
    ].copy()
    if subset.empty:
        raise ValueError(f"No rows available for patient_id={patient_id}, selected_k={selected_k}")

    rows: List[Dict[str, Any]] = []
    for time_bin in range(NUM_TIME_BINS):
        bin_rows = subset[subset["time_bin"] == int(time_bin)]
        hit_values = bin_rows["hit_at_k"].astype(float).tolist()
        precision_values = bin_rows["precision_at_k"].astype(float).tolist()
        rows.append(
            {
                "time_bin": int(time_bin),
                "time_bin_mid_pct": float((time_bin + 0.5) * (100.0 / NUM_TIME_BINS)),
                "n_windows": int(len(bin_rows)),
                "hit_at_k": float(np.mean(hit_values)) if hit_values else float("nan"),
                "precision_at_k": float(np.mean(precision_values)) if precision_values else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)


def _build_patient_prefix_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    patient_id: str,
) -> pd.DataFrame:
    subset = window_k_frame[
        (window_k_frame["k"] == int(selected_k)) & (window_k_frame["patient_id"].astype(str) == str(patient_id))
    ].copy()
    if subset.empty:
        raise ValueError(f"No rows available for patient_id={patient_id}, selected_k={selected_k}")

    subset = subset.sort_values(["relative_time", "window_index"]).reset_index(drop=True)
    subset["cum_hit_sum"] = subset["hit_at_k"].cumsum()
    subset["cum_precision_sum"] = subset["precision_at_k"].cumsum()
    subset["cum_count"] = subset.index.to_series() + 1
    subset["prefix_hit_at_k"] = subset["cum_hit_sum"] / subset["cum_count"]
    subset["prefix_precision_at_k"] = subset["cum_precision_sum"] / subset["cum_count"]

    thresholds = np.linspace(0.0, 1.0, NUM_TIME_BINS + 1)
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        valid = subset[subset["relative_time"] <= float(threshold)]
        if valid.empty:
            rows.append(
                {
                    "relative_time_pct": float(threshold * 100.0),
                    "prefix_hit_at_k": float("nan"),
                    "prefix_precision_at_k": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "relative_time_pct": float(threshold * 100.0),
                "prefix_hit_at_k": float(valid["prefix_hit_at_k"].iloc[-1]),
                "prefix_precision_at_k": float(valid["prefix_precision_at_k"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def _plot_patient_mode_relative_curve(
    curves_by_mode: Mapping[str, pd.DataFrame],
    *,
    patient_id: str,
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError(f"No mode curves provided for patient_id={patient_id}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color_by_mode = mode_color_map(sorted(curves_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    for mode_label, curve_df in curves_by_mode.items():
        ordered = curve_df.sort_values("time_bin").reset_index(drop=True)
        x = ordered["time_bin_mid_pct"].to_numpy(dtype=float)
        hit = smooth_series(ordered["hit_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        precision = smooth_series(ordered["precision_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        color = color_by_mode[str(mode_label)]
        axes[0].plot(x, hit, linewidth=2.2, label=str(mode_label), color=color, alpha=MODE_TRACE_ALPHA)
        axes[1].plot(
            x,
            precision,
            linewidth=2.2,
            label=str(mode_label),
            color=color,
            alpha=MODE_TRACE_ALPHA,
        )
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "mode": str(mode_label),
                    "time_bin": int(row["time_bin"]),
                    "time_bin_mid_pct": float(row["time_bin_mid_pct"]),
                    "n_windows": int(row["n_windows"]),
                    "hit_at_k": float(row["hit_at_k"]),
                    "precision_at_k": float(row["precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Relative Window Position (%)")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title(f"Patient {patient_id}: Relative-Time Hit Rate by Mode")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Relative Window Position (%)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Patient {patient_id}: Relative-Time Precision by Mode")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError(f"No rows generated for patient_id={patient_id}")
    return output_df


def _plot_patient_mode_prefix_curve(
    curves_by_mode: Mapping[str, pd.DataFrame],
    *,
    patient_id: str,
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError(f"No mode curves provided for patient_id={patient_id}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color_by_mode = mode_color_map(sorted(curves_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    for mode_label, curve_df in curves_by_mode.items():
        ordered = curve_df.sort_values("relative_time_pct").reset_index(drop=True)
        x = ordered["relative_time_pct"].to_numpy(dtype=float)
        hit = smooth_series(ordered["prefix_hit_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        precision = smooth_series(
            ordered["prefix_precision_at_k"].to_numpy(dtype=float),
            window=CURVE_SMOOTHING_WINDOW,
        )
        color = color_by_mode[str(mode_label)]
        axes[0].plot(x, hit, linewidth=2.2, label=str(mode_label), color=color, alpha=MODE_TRACE_ALPHA)
        axes[1].plot(
            x,
            precision,
            linewidth=2.2,
            label=str(mode_label),
            color=color,
            alpha=MODE_TRACE_ALPHA,
        )
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "mode": str(mode_label),
                    "relative_time_pct": float(row["relative_time_pct"]),
                    "prefix_hit_at_k": float(row["prefix_hit_at_k"]),
                    "prefix_precision_at_k": float(row["prefix_precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Relative Time (%)")
    axes[0].set_ylabel("Cumulative Hit Rate")
    axes[0].set_title(f"Patient {patient_id}: Prefix Cumulative Hit Rate by Mode")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Relative Time (%)")
    axes[1].set_ylabel("Cumulative Precision")
    axes[1].set_title(f"Patient {patient_id}: Prefix Cumulative Precision by Mode")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError(f"No rows generated for patient_id={patient_id}")
    return output_df


def _plot_patient_mode_comparison_curves(
    mode_results: Mapping[str, Mapping[str, Any]],
    *,
    output_root: Path,
) -> Tuple[int, int]:
    if not mode_results:
        return 0, 0

    patient_ids: List[str] = sorted(
        {
            str(patient_id)
            for result in mode_results.values()
            for patient_id in result["window_k_frame"]["patient_id"].dropna().astype(str).unique().tolist()
        }
    )
    if not patient_ids:
        return 0, 0

    patient_root = output_root / "patient_mode_plots"
    patient_root.mkdir(parents=True, exist_ok=True)

    plotted_relative = 0
    plotted_prefix = 0
    for patient_id in patient_ids:
        relative_by_mode: Dict[str, pd.DataFrame] = {}
        prefix_by_mode: Dict[str, pd.DataFrame] = {}
        for mode_name, result in mode_results.items():
            mode_frame = result["window_k_frame"]
            selected_k = int(result["selected_k"])
            label = str(mode_name)
            try:
                relative_by_mode[label] = _build_patient_relative_time_curve(
                    mode_frame,
                    selected_k=selected_k,
                    patient_id=patient_id,
                )
                prefix_by_mode[label] = _build_patient_prefix_curve(
                    mode_frame,
                    selected_k=selected_k,
                    patient_id=patient_id,
                )
            except ValueError:
                continue

        if not relative_by_mode:
            continue

        patient_dir = patient_root / str(patient_id).replace("/", "_")
        patient_dir.mkdir(parents=True, exist_ok=True)

        relative_df = _plot_patient_mode_relative_curve(
            relative_by_mode,
            patient_id=patient_id,
            output_path=patient_dir / "plot1_relative_time_hit_precision_curve_by_mode.png",
        )
        relative_df.to_csv(patient_dir / "plot1_relative_time_hit_precision_curve_by_mode.csv", index=False)
        plotted_relative += 1

        prefix_df = _plot_patient_mode_prefix_curve(
            prefix_by_mode,
            patient_id=patient_id,
            output_path=patient_dir / "plot2_prefix_cumulative_hit_precision_curve_by_mode.png",
        )
        prefix_df.to_csv(patient_dir / "plot2_prefix_cumulative_hit_precision_curve_by_mode.csv", index=False)
        plotted_prefix += 1

    return int(plotted_relative), int(plotted_prefix)


def _plot_relative_time_curve_by_mode(
    curve_by_mode: Mapping[str, pd.DataFrame],
    *,
    output_path: Path,
) -> pd.DataFrame:
    if not curve_by_mode:
        raise ValueError("No mode curves provided for relative-time comparison plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color_by_mode = mode_color_map(sorted(curve_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    for mode_label, curve_df in curve_by_mode.items():
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("time_bin").reset_index(drop=True)
        x = ordered["time_bin_mid_pct"].to_numpy(dtype=float)
        y_hit = smooth_series(ordered["hit_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        y_precision = smooth_series(ordered["precision_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        color = color_by_mode[str(mode_label)]
        if {"hit_ci_low", "hit_ci_high"}.issubset(set(ordered.columns)):
            _, hit_low, hit_high = _smooth_metric_and_band(
                ordered["hit_at_k"].to_numpy(dtype=float),
                ordered["hit_ci_low"].to_numpy(dtype=float),
                ordered["hit_ci_high"].to_numpy(dtype=float),
            )
            axes[0].fill_between(x, hit_low, hit_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[0].plot(x, y_hit, linewidth=2.2, label=str(mode_label), color=color, alpha=MODE_TRACE_ALPHA)
        if {"precision_ci_low", "precision_ci_high"}.issubset(set(ordered.columns)):
            _, precision_low, precision_high = _smooth_metric_and_band(
                ordered["precision_at_k"].to_numpy(dtype=float),
                ordered["precision_ci_low"].to_numpy(dtype=float),
                ordered["precision_ci_high"].to_numpy(dtype=float),
            )
            axes[1].fill_between(x, precision_low, precision_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[1].plot(
            x,
            y_precision,
            linewidth=2.2,
            label=str(mode_label),
            color=color,
            alpha=MODE_TRACE_ALPHA,
        )
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_label),
                    "time_bin": int(row["time_bin"]),
                    "time_bin_mid_pct": float(row["time_bin_mid_pct"]),
                    "n_windows": int(row["n_windows"]),
                    "hit_at_k": float(row["hit_at_k"]),
                    "precision_at_k": float(row["precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Relative Window Position (%)")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title("Relative-Time Hit Rate by Mode")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Relative Window Position (%)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Relative-Time Precision by Mode")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for relative-time mode comparison plot.")
    return comparison_df


def _plot_prefix_curve_by_mode(
    curve_by_mode: Mapping[str, pd.DataFrame],
    *,
    output_path: Path,
) -> pd.DataFrame:
    if not curve_by_mode:
        raise ValueError("No mode curves provided for prefix comparison plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color_by_mode = mode_color_map(sorted(curve_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    for mode_label, curve_df in curve_by_mode.items():
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("relative_time_pct").reset_index(drop=True)
        x = ordered["relative_time_pct"].to_numpy(dtype=float)
        y_hit_raw = ordered["cohort_mean_prefix_hit_at_k"].to_numpy(dtype=float)
        y_precision_raw = ordered["cohort_mean_prefix_precision_at_k"].to_numpy(dtype=float)
        y_hit = smooth_series(y_hit_raw, window=CURVE_SMOOTHING_WINDOW)
        y_precision = smooth_series(y_precision_raw, window=CURVE_SMOOTHING_WINDOW)
        color = color_by_mode[str(mode_label)]
        if {"cohort_std_prefix_hit_at_k"}.issubset(set(ordered.columns)):
            hit_low_raw, hit_high_raw = _band_from_mean_std(
                y_hit_raw,
                ordered["cohort_std_prefix_hit_at_k"].to_numpy(dtype=float),
            )
            _, hit_low, hit_high = _smooth_metric_and_band(y_hit_raw, hit_low_raw, hit_high_raw)
            axes[0].fill_between(x, hit_low, hit_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[0].plot(x, y_hit, linewidth=2.2, label=str(mode_label), color=color, alpha=MODE_TRACE_ALPHA)
        if {"cohort_std_prefix_precision_at_k"}.issubset(set(ordered.columns)):
            precision_low_raw, precision_high_raw = _band_from_mean_std(
                y_precision_raw,
                ordered["cohort_std_prefix_precision_at_k"].to_numpy(dtype=float),
            )
            _, precision_low, precision_high = _smooth_metric_and_band(
                y_precision_raw,
                precision_low_raw,
                precision_high_raw,
            )
            axes[1].fill_between(x, precision_low, precision_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[1].plot(
            x,
            y_precision,
            linewidth=2.2,
            label=str(mode_label),
            color=color,
            alpha=MODE_TRACE_ALPHA,
        )
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_label),
                    "relative_time_pct": float(row["relative_time_pct"]),
                    "num_patients": int(row["num_patients"]),
                    "cohort_mean_prefix_hit_at_k": float(row["cohort_mean_prefix_hit_at_k"]),
                    "cohort_mean_prefix_precision_at_k": float(row["cohort_mean_prefix_precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Relative Time (%)")
    axes[0].set_ylabel("Cumulative Hit Rate")
    axes[0].set_title("Prefix Cumulative Hit Rate by Mode")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Relative Time (%)")
    axes[1].set_ylabel("Cumulative Precision")
    axes[1].set_title("Prefix Cumulative Precision by Mode")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for prefix mode comparison plot.")
    return comparison_df


def _plot_absolute_hour_curve_by_mode(
    curve_by_mode: Mapping[str, pd.DataFrame],
    *,
    output_path: Path,
) -> pd.DataFrame:
    if not curve_by_mode:
        raise ValueError("No mode curves provided for absolute-hour comparison plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color_by_mode = mode_color_map(sorted(curve_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    max_hour = 0.0
    global_max_bin = max(
        int(curve_df["absolute_hour_bin"].max())
        for curve_df in curve_by_mode.values()
        if not curve_df.empty and "absolute_hour_bin" in curve_df.columns
    )
    for mode_label, curve_df in curve_by_mode.items():
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("absolute_hour_bin").reset_index(drop=True)
        max_mode_bin = int(ordered["absolute_hour_bin"].max())
        if max_mode_bin < global_max_bin:
            tail_rows: List[Dict[str, Any]] = []
            template_row = ordered.iloc[-1].to_dict()
            for hour_bin in range(max_mode_bin + 1, global_max_bin + 1):
                padded_row = dict(template_row)
                padded_row["absolute_hour_bin"] = int(hour_bin)
                padded_row["hour_start"] = float(hour_bin * float(ABSOLUTE_TIME_BIN_HOURS))
                padded_row["hour_end"] = float((hour_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS))
                padded_row["hour_mid"] = float((hour_bin + 0.5) * float(ABSOLUTE_TIME_BIN_HOURS))
                padded_row["n_windows"] = 0
                padded_row["num_patients"] = 0
                padded_row["hit_at_k"] = 0.0
                padded_row["precision_at_k"] = 0.0
                if "hit_ci_low" in padded_row:
                    padded_row["hit_ci_low"] = 0.0
                if "hit_ci_high" in padded_row:
                    padded_row["hit_ci_high"] = 0.0
                if "precision_ci_low" in padded_row:
                    padded_row["precision_ci_low"] = 0.0
                if "precision_ci_high" in padded_row:
                    padded_row["precision_ci_high"] = 0.0
                tail_rows.append(padded_row)
            if tail_rows:
                ordered = pd.concat([ordered, pd.DataFrame(tail_rows)], ignore_index=True)
        x = ordered["hour_mid"].to_numpy(dtype=float)
        y_hit = smooth_series(ordered["hit_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        y_precision = smooth_series(ordered["precision_at_k"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        color = color_by_mode[str(mode_label)]
        if {"hit_ci_low", "hit_ci_high"}.issubset(set(ordered.columns)):
            _, hit_low, hit_high = _smooth_metric_and_band(
                ordered["hit_at_k"].to_numpy(dtype=float),
                ordered["hit_ci_low"].to_numpy(dtype=float),
                ordered["hit_ci_high"].to_numpy(dtype=float),
            )
            axes[0].fill_between(x, hit_low, hit_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[0].plot(x, y_hit, linewidth=2.2, label=str(mode_label), color=color, alpha=MODE_TRACE_ALPHA)
        if {"precision_ci_low", "precision_ci_high"}.issubset(set(ordered.columns)):
            _, precision_low, precision_high = _smooth_metric_and_band(
                ordered["precision_at_k"].to_numpy(dtype=float),
                ordered["precision_ci_low"].to_numpy(dtype=float),
                ordered["precision_ci_high"].to_numpy(dtype=float),
            )
            axes[1].fill_between(x, precision_low, precision_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[1].plot(
            x,
            y_precision,
            linewidth=2.2,
            label=str(mode_label),
            color=color,
            alpha=MODE_TRACE_ALPHA,
        )
        max_hour = max(max_hour, float(ordered["hour_end"].max()))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_label),
                    "absolute_hour_bin": int(row["absolute_hour_bin"]),
                    "hour_start": float(row["hour_start"]),
                    "hour_end": float(row["hour_end"]),
                    "hour_mid": float(row["hour_mid"]),
                    "n_windows": int(row["n_windows"]),
                    "hit_at_k": float(row["hit_at_k"]),
                    "precision_at_k": float(row["precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Hours Since ICU Admission")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title("Absolute-Time Hit Rate by Mode")
    axes[0].set_xlim(0, max_hour)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Hours Since ICU Admission")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Absolute-Time Precision by Mode")
    axes[1].set_xlim(0, max_hour)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for absolute-hour mode comparison plot.")
    return comparison_df


def _plot_absolute_hour_single_metric_by_mode(
    curve_by_mode: Mapping[str, pd.DataFrame],
    *,
    metric_column: str,
    metric_label: str,
    output_path: Path,
) -> pd.DataFrame:
    if not curve_by_mode:
        raise ValueError(f"No mode curves provided for {metric_column} absolute-hour comparison plot.")

    fig, ax = plt.subplots(figsize=(6, 5))
    color_by_mode = mode_color_map(sorted(curve_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    max_hour = 0.0
    global_max_bin = max(
        int(curve_df["absolute_hour_bin"].max())
        for curve_df in curve_by_mode.values()
        if not curve_df.empty and "absolute_hour_bin" in curve_df.columns
    )
    for mode_label, curve_df in curve_by_mode.items():
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("absolute_hour_bin").reset_index(drop=True)
        max_mode_bin = int(ordered["absolute_hour_bin"].max())
        if max_mode_bin < global_max_bin:
            tail_rows: List[Dict[str, Any]] = []
            template_row = ordered.iloc[-1].to_dict()
            for hour_bin in range(max_mode_bin + 1, global_max_bin + 1):
                padded_row = dict(template_row)
                padded_row["absolute_hour_bin"] = int(hour_bin)
                padded_row["hour_start"] = float(hour_bin * float(ABSOLUTE_TIME_BIN_HOURS))
                padded_row["hour_end"] = float((hour_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS))
                padded_row["hour_mid"] = float((hour_bin + 0.5) * float(ABSOLUTE_TIME_BIN_HOURS))
                padded_row["n_windows"] = 0
                if "num_red_flag_patients" in padded_row:
                    padded_row["num_red_flag_patients"] = 0
                padded_row[metric_column] = 0.0
                ci_low_column = f"{metric_column}_ci_low"
                ci_high_column = f"{metric_column}_ci_high"
                if ci_low_column in padded_row:
                    padded_row[ci_low_column] = 0.0
                if ci_high_column in padded_row:
                    padded_row[ci_high_column] = 0.0
                tail_rows.append(padded_row)
            if tail_rows:
                ordered = pd.concat([ordered, pd.DataFrame(tail_rows)], ignore_index=True)

        x = ordered["hour_mid"].to_numpy(dtype=float)
        y = smooth_series(ordered[metric_column].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        color = color_by_mode[str(mode_label)]
        ci_low_column = f"{metric_column}_ci_low"
        ci_high_column = f"{metric_column}_ci_high"
        if {ci_low_column, ci_high_column}.issubset(set(ordered.columns)):
            _, low, high = _smooth_metric_and_band(
                ordered[metric_column].to_numpy(dtype=float),
                ordered[ci_low_column].to_numpy(dtype=float),
                ordered[ci_high_column].to_numpy(dtype=float),
            )
            ax.fill_between(x, low, high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        ax.plot(x, y, linewidth=2.2, label=str(mode_label), color=color, alpha=MODE_TRACE_ALPHA)
        max_hour = max(max_hour, float(ordered["hour_end"].max()))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_label),
                    "absolute_hour_bin": int(row["absolute_hour_bin"]),
                    "hour_start": float(row["hour_start"]),
                    "hour_end": float(row["hour_end"]),
                    "hour_mid": float(row["hour_mid"]),
                    "n_windows": int(row["n_windows"]),
                    metric_column: float(row[metric_column]),
                    ci_low_column: float(row[ci_low_column]) if ci_low_column in row else float("nan"),
                    ci_high_column: float(row[ci_high_column]) if ci_high_column in row else float("nan"),
                }
            )

    ax.set_xlabel("Hours Since ICU Admission")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Red Flags: Absolute-Time {metric_label} by Mode")
    ax.set_xlim(0, max_hour if max_hour > 0 else float(ABSOLUTE_TIME_BIN_HOURS))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError(f"No rows generated for {metric_column} absolute-hour mode comparison plot.")
    return output_df


def _plot_absolute_hour_prefix_curve_by_mode(
    curve_by_mode: Mapping[str, pd.DataFrame],
    *,
    output_path: Path,
) -> pd.DataFrame:
    if not curve_by_mode:
        raise ValueError("No mode curves provided for absolute-hour prefix comparison plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color_by_mode = mode_color_map(sorted(curve_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    max_hour = 0.0
    for mode_label, curve_df in curve_by_mode.items():
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("hours_since_admission").reset_index(drop=True)
        x = ordered["hours_since_admission"].to_numpy(dtype=float)
        y_hit_raw = ordered["cohort_mean_prefix_hit_at_k"].to_numpy(dtype=float)
        y_precision_raw = ordered["cohort_mean_prefix_precision_at_k"].to_numpy(dtype=float)
        y_hit = smooth_series(y_hit_raw, window=CURVE_SMOOTHING_WINDOW)
        y_precision = smooth_series(y_precision_raw, window=CURVE_SMOOTHING_WINDOW)
        color = color_by_mode[str(mode_label)]
        if {"cohort_std_prefix_hit_at_k"}.issubset(set(ordered.columns)):
            hit_low_raw, hit_high_raw = _band_from_mean_std(
                y_hit_raw,
                ordered["cohort_std_prefix_hit_at_k"].to_numpy(dtype=float),
            )
            _, hit_low, hit_high = _smooth_metric_and_band(y_hit_raw, hit_low_raw, hit_high_raw)
            axes[0].fill_between(x, hit_low, hit_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[0].plot(x, y_hit, linewidth=2.2, label=str(mode_label), color=color, alpha=MODE_TRACE_ALPHA)
        if {"cohort_std_prefix_precision_at_k"}.issubset(set(ordered.columns)):
            precision_low_raw, precision_high_raw = _band_from_mean_std(
                y_precision_raw,
                ordered["cohort_std_prefix_precision_at_k"].to_numpy(dtype=float),
            )
            _, precision_low, precision_high = _smooth_metric_and_band(
                y_precision_raw,
                precision_low_raw,
                precision_high_raw,
            )
            axes[1].fill_between(x, precision_low, precision_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[1].plot(
            x,
            y_precision,
            linewidth=2.2,
            label=str(mode_label),
            color=color,
            alpha=MODE_TRACE_ALPHA,
        )
        max_hour = max(max_hour, float(ordered["hours_since_admission"].max()))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_label),
                    "hours_since_admission": float(row["hours_since_admission"]),
                    "num_patients": int(row["num_patients"]),
                    "cohort_mean_prefix_hit_at_k": float(row["cohort_mean_prefix_hit_at_k"]),
                    "cohort_mean_prefix_precision_at_k": float(row["cohort_mean_prefix_precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Hours Since ICU Admission")
    axes[0].set_ylabel("Cumulative Hit Rate")
    axes[0].set_title("Absolute-Time Prefix Hit Rate by Mode")
    axes[0].set_xlim(0, max_hour)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Hours Since ICU Admission")
    axes[1].set_ylabel("Cumulative Precision")
    axes[1].set_title("Absolute-Time Prefix Precision by Mode")
    axes[1].set_xlim(0, max_hour)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for absolute-hour prefix mode comparison plot.")
    return comparison_df


def _evaluate_single_prediction_dir(
    *,
    prediction_root: Path,
    output_dir: Path,
    gt_dir: Path,
    num_workers: int,
    matcher_backend: str,
    pubmedbert_model: str,
    pubmedbert_similarity_threshold: float,
    pubmedbert_batch_size: int,
    pubmedbert_device: Optional[str],
) -> Dict[str, Any]:
    gt_root = _ensure_gt_dir(gt_dir)

    backend = str(matcher_backend).strip().lower()
    if backend not in SUPPORTED_MATCHER_BACKENDS:
        raise ValueError(
            f"Unsupported matcher_backend={matcher_backend}. Expected one of {SUPPORTED_MATCHER_BACKENDS}"
        )

    if int(num_workers) < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    print(f"Prediction directory: {prediction_root}")
    print(f"GT directory: {gt_root}")
    if backend == MATCHER_BACKEND_LLM:
        print(
            f"Matcher config: backend={backend}, provider={LLM_MATCHER_PROVIDER}, model={LLM_MATCHER_MODEL}, "
            f"max_tokens={int(LLM_MATCHER_MAX_TOKENS)}, workers={int(num_workers)}"
        )
    else:
        print(
            f"Matcher config: backend={backend}, model={pubmedbert_model}, "
            f"similarity_threshold={float(pubmedbert_similarity_threshold)}, "
            f"batch_size={int(pubmedbert_batch_size)}, device={pubmedbert_device}, workers={int(num_workers)}"
        )

    window_frame, prediction_frame, matcher_usage = _load_window_and_prediction_rows(
        prediction_dir=prediction_root,
        gt_dir=gt_root,
        num_workers=int(num_workers),
        matcher_backend=backend,
        pubmedbert_model=str(pubmedbert_model),
        pubmedbert_similarity_threshold=float(pubmedbert_similarity_threshold),
        pubmedbert_batch_size=int(pubmedbert_batch_size),
        pubmedbert_device=pubmedbert_device,
    )
    return _write_recommendation_analysis_outputs(
        window_frame=window_frame,
        prediction_frame=prediction_frame,
        matcher_usage=matcher_usage,
        prediction_root=prediction_root,
        gt_root=gt_root,
        output_dir=output_dir,
    )


def _evaluate_reused_eval_dir(
    *,
    reuse_eval_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    print(f"Reuse eval directory: {reuse_eval_dir}")
    window_frame, prediction_frame, matcher_usage = _load_reused_recommendation_frames(reuse_eval_dir)
    return _write_recommendation_analysis_outputs(
        window_frame=window_frame,
        prediction_frame=prediction_frame,
        matcher_usage=matcher_usage,
        prediction_root=None,
        gt_root=None,
        output_dir=output_dir,
        reuse_eval_dir=reuse_eval_dir,
    )


def run_evaluation(
    *,
    prediction_dir: Optional[Path],
    output_dir: Optional[Path],
    gt_dir: Optional[Path],
    num_workers: int,
    matcher_backend: str,
    pubmedbert_model: str,
    pubmedbert_similarity_threshold: float,
    pubmedbert_batch_size: int,
    pubmedbert_device: Optional[str],
    reuse_eval_dir: Optional[Path] = None,
) -> None:
    if reuse_eval_dir is not None:
        reuse_root = _ensure_recommendation_eval_dir(reuse_eval_dir)
        mode_dirs = _discover_recommendation_eval_mode_dirs(reuse_root)
        if mode_dirs:
            output_root = output_dir if output_dir is not None else reuse_root
            output_root.mkdir(parents=True, exist_ok=True)
            print(
                "Detected reused recommendation modes: " + ", ".join(sorted(mode_dirs.keys())) + f" under {reuse_root}"
            )

            absolute_hour_curves_by_mode: Dict[str, pd.DataFrame] = {}
            for mode in sorted(mode_dirs.keys()):
                mode_output_dir = output_root / mode
                print("")
                print(f"=== Reusing matched results for mode: {mode} ===")
                single_result = _evaluate_reused_eval_dir(
                    reuse_eval_dir=mode_dirs[mode],
                    output_dir=mode_output_dir,
                )
                mode_label = str(mode)
                mode_curve = single_result["absolute_hour_curve_df"].copy()
                mode_curve["mode"] = mode_label
                absolute_hour_curves_by_mode[mode_label] = mode_curve

            if len(absolute_hour_curves_by_mode) > 1:
                combined_absolute_hour_df = pd.concat(
                    [absolute_hour_curves_by_mode[label] for label in sorted(absolute_hour_curves_by_mode.keys())],
                    ignore_index=True,
                )
                combined_absolute_hour_df.to_csv(
                    output_root / "absolute_hour_recommendation_metrics_by_mode.csv", index=False
                )
                combined_absolute_hour_df[
                    ["mode", "k", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_hit_at_k"]
                ].to_csv(output_root / "absolute_hour_recommendation_macro_hit_at_k_by_mode.csv", index=False)
                combined_absolute_hour_df[
                    ["mode", "k", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_recall_at_k"]
                ].to_csv(output_root / "absolute_hour_recommendation_macro_recall_at_k_by_mode.csv", index=False)
                combined_absolute_hour_df[
                    [
                        "mode",
                        "k",
                        "absolute_hour_bin",
                        "hour_start",
                        "hour_end",
                        "hour_mid",
                        "n_windows",
                        "num_red_flag_patients",
                        "harmful_recommendation_rate_case_level_at_k",
                    ]
                ].to_csv(output_root / "absolute_hour_red_flags_hrr_by_mode.csv", index=False)
            print(f"Saved reused multi-mode evaluation outputs to: {output_root}")
            return

        final_output_dir = output_dir if output_dir is not None else reuse_root
        _evaluate_reused_eval_dir(reuse_eval_dir=reuse_root, output_dir=final_output_dir)
        return

    if prediction_dir is None:
        raise ValueError("--prediction-dir is required unless --reuse-eval-dir is provided.")
    if gt_dir is None:
        raise ValueError("--gt-dir is required unless --reuse-eval-dir is provided.")

    mode_dirs = _discover_recommendation_mode_dirs(prediction_dir)
    if mode_dirs:
        base_output_root = (
            output_dir if output_dir is not None else _default_output_root_from_recommendation_input(prediction_dir)
        )
        output_root = base_output_root
        output_root.mkdir(parents=True, exist_ok=True)
        print("Detected recommendation modes: " + ", ".join(sorted(mode_dirs.keys())) + f" under {prediction_dir}")

        absolute_hour_curves_by_mode: Dict[str, pd.DataFrame] = {}
        for mode in sorted(mode_dirs.keys()):
            prediction_root = mode_dirs[mode]
            mode_output_dir = output_root / mode
            print("")
            print(f"=== Evaluating mode: {mode} ===")
            single_result = _evaluate_single_prediction_dir(
                prediction_root=prediction_root,
                output_dir=mode_output_dir,
                gt_dir=gt_dir,
                num_workers=int(num_workers),
                matcher_backend=matcher_backend,
                pubmedbert_model=pubmedbert_model,
                pubmedbert_similarity_threshold=float(pubmedbert_similarity_threshold),
                pubmedbert_batch_size=int(pubmedbert_batch_size),
                pubmedbert_device=pubmedbert_device,
            )
            mode_label = str(mode)
            mode_curve = single_result["absolute_hour_curve_df"].copy()
            mode_curve["mode"] = mode_label
            absolute_hour_curves_by_mode[mode_label] = mode_curve

        if len(absolute_hour_curves_by_mode) > 1:
            combined_absolute_hour_df = pd.concat(
                [absolute_hour_curves_by_mode[label] for label in sorted(absolute_hour_curves_by_mode.keys())],
                ignore_index=True,
            )
            combined_absolute_hour_df.to_csv(
                output_root / "absolute_hour_recommendation_metrics_by_mode.csv", index=False
            )
            combined_absolute_hour_df[
                ["mode", "k", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_hit_at_k"]
            ].to_csv(output_root / "absolute_hour_recommendation_macro_hit_at_k_by_mode.csv", index=False)
            combined_absolute_hour_df[
                ["mode", "k", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_recall_at_k"]
            ].to_csv(output_root / "absolute_hour_recommendation_macro_recall_at_k_by_mode.csv", index=False)
            combined_absolute_hour_df[
                [
                    "mode",
                    "k",
                    "absolute_hour_bin",
                    "hour_start",
                    "hour_end",
                    "hour_mid",
                    "n_windows",
                    "num_red_flag_patients",
                    "harmful_recommendation_rate_case_level_at_k",
                ]
            ].to_csv(output_root / "absolute_hour_red_flags_hrr_by_mode.csv", index=False)
        print(f"Saved multi-mode evaluation outputs to: {output_root}")
        return

    prediction_root = _ensure_predictions_dir(prediction_dir)
    base_final_output_dir = (
        output_dir if output_dir is not None else _default_output_root_from_recommendation_input(prediction_root)
    )
    final_output_root = base_final_output_dir
    if output_dir is None and prediction_root.parent.name == "recommendation_experiment":
        final_output_dir = final_output_root / prediction_root.name
    else:
        final_output_dir = final_output_root
    _evaluate_single_prediction_dir(
        prediction_root=prediction_root,
        output_dir=final_output_dir,
        gt_dir=gt_dir,
        num_workers=int(num_workers),
        matcher_backend=matcher_backend,
        pubmedbert_model=pubmedbert_model,
        pubmedbert_similarity_threshold=float(pubmedbert_similarity_threshold),
        pubmedbert_batch_size=int(pubmedbert_batch_size),
        pubmedbert_device=pubmedbert_device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation predictions with configurable action matching."
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="Recommendation prediction directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where metrics and CSV summaries will be saved. "
            "Default for multi-mode input: "
            "evaluation_results/<memory_run_name>/recommendation_experiment/<mode>. "
            "Default for single-mode input: "
            "evaluation_results/<memory_run_name>/recommendation_experiment/<mode>."
        ),
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help=(
            "GT directory with patients/*/oracle_predictions.json. "
            "Only actions labeled acceptable/best_practice are used as GT."
        ),
    )
    parser.add_argument(
        "--reuse-eval-dir",
        type=str,
        default=None,
        help=(
            "Existing evaluation output directory containing window_level_windows.csv. "
            "If provided, skips matcher inference and recomputes metrics/CSV summaries from saved matcher results."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for window-level matching.",
    )
    parser.add_argument(
        "--matcher-backend",
        type=str,
        choices=SUPPORTED_MATCHER_BACKENDS,
        default=MATCHER_BACKEND_PUBMEDBERT,
        help="Matcher backend to use.",
    )
    parser.add_argument(
        "--pubmedbert-model",
        type=str,
        default=PUBMEDBERT_MODEL,
        help="Sentence-transformers model name for pubmedbert backend.",
    )
    parser.add_argument(
        "--pubmedbert-similarity-threshold",
        type=float,
        default=PUBMEDBERT_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for pubmedbert backend.",
    )
    parser.add_argument(
        "--pubmedbert-batch-size",
        type=int,
        default=PUBMEDBERT_BATCH_SIZE,
        help="Embedding batch size for pubmedbert backend.",
    )
    parser.add_argument(
        "--pubmedbert-device",
        type=str,
        default=None,
        help="Device for pubmedbert backend. Defaults to auto-detected cuda/cpu.",
    )
    args = parser.parse_args()

    run_evaluation(
        prediction_dir=Path(args.prediction_dir) if args.prediction_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        gt_dir=Path(args.gt_dir) if args.gt_dir else None,
        num_workers=int(args.num_workers),
        matcher_backend=str(args.matcher_backend),
        pubmedbert_model=str(args.pubmedbert_model),
        pubmedbert_similarity_threshold=float(args.pubmedbert_similarity_threshold),
        pubmedbert_batch_size=int(args.pubmedbert_batch_size),
        pubmedbert_device=args.pubmedbert_device,
        reuse_eval_dir=Path(args.reuse_eval_dir) if args.reuse_eval_dir else None,
    )


if __name__ == "__main__":
    main()
