"""Evaluate patient-status predictions against GT labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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
    system_color,
)
from experiments.oracle.common import assign_normalized_time_bin, extract_overall_label
from utils.status_scoring import normalize_status_label

configure_plot_style()

NUM_TIME_BINS = 10
MEMORY_DEPTH_NUM_BINS = 6
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260409
ABSOLUTE_TIME_BIN_HOURS = 24.0
ORACLE_INTERVAL_DECIMALS = 6
CURVE_SMOOTHING_WINDOW = 2
KNOWN_PATIENT_STATUS_MODES = (
    "memory",
    "full_history_events",
    "local_events_only",
    "rag_history_events",
)
PATIENT_STATUS_ROOT_NAMES = (
    "patient_status",
    "patient_status_experiment",
)

@dataclass(frozen=True)
class WindowLabel:
    label: str
    hours_since_admission: float
    num_windows: int
    window_order: int
    interval_start_hour: float
    interval_end_hour: float
    marker_id: str


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _extract_icu_stay_id(payload: Dict[str, Any], source_path: Path) -> int:
    icu_stay_id_raw = payload.get("icu_stay_id")
    if icu_stay_id_raw is None:
        raise ValueError(f"Missing icu_stay_id in {source_path}")
    icu_stay_id = int(icu_stay_id_raw)
    if icu_stay_id <= 0:
        raise ValueError(f"Invalid icu_stay_id={icu_stay_id} in {source_path}")
    return icu_stay_id


def _has_patient_status_predictions(path: Path) -> bool:
    patients_dir = path / "patients"
    if not patients_dir.exists() or not patients_dir.is_dir():
        return False
    return any(patients_dir.glob("*/patient_status_predictions.json"))


def _ensure_predictions_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {path}")
    if _has_patient_status_predictions(path):
        return path
    for root_name in PATIENT_STATUS_ROOT_NAMES:
        nested_root = path / root_name
        if nested_root.exists() and nested_root.is_dir() and _has_patient_status_predictions(nested_root):
            return nested_root
    raise FileNotFoundError(
        f"No patient_status_predictions.json found under {path}. "
        "Expected either <dir>/patients/*/patient_status_predictions.json or "
        "<dir>/<patient_status|patient_status_experiment>/patients/*/patient_status_predictions.json."
    )


def _discover_patient_status_mode_dirs(path: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for mode in KNOWN_PATIENT_STATUS_MODES:
        candidate = path / mode
        if candidate.exists() and candidate.is_dir() and _has_patient_status_predictions(candidate):
            discovered[str(mode)] = candidate
    return discovered


def _ensure_gt_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"GT directory not found: {path}")
    patients_dir = path / "patients"
    if not patients_dir.exists():
        raise FileNotFoundError(f"Missing patients directory under GT path: {path}")
    if not list(patients_dir.glob("*/oracle_predictions.json")):
        raise FileNotFoundError(f"No oracle_predictions.json files found under {patients_dir}")
    return path


def _default_output_dir_from_med_evo_root(med_evo_root: Path) -> Path:
    experiment_name = med_evo_root.name
    memory_run_name = med_evo_root.parent.name if med_evo_root.parent != med_evo_root else med_evo_root.name
    if not memory_run_name:
        raise ValueError(f"Cannot infer memory run name from med_evo_root={med_evo_root}")
    if not experiment_name:
        raise ValueError(f"Cannot infer experiment name from med_evo_root={med_evo_root}")
    return Path("evaluation_results") / memory_run_name / experiment_name


def _default_output_root_from_patient_status_input(patient_status_input: Path) -> Path:
    if patient_status_input.name in PATIENT_STATUS_ROOT_NAMES:
        memory_run_name = patient_status_input.parent.name
        if not memory_run_name:
            raise ValueError(f"Cannot infer memory run name from patient_status_input={patient_status_input}")
        return Path("evaluation_results") / memory_run_name / patient_status_input.name
    if patient_status_input.parent.name in PATIENT_STATUS_ROOT_NAMES:
        memory_run_name = patient_status_input.parent.parent.name
        if not memory_run_name:
            raise ValueError(f"Cannot infer memory run name from patient_status_input={patient_status_input}")
        return Path("evaluation_results") / memory_run_name / patient_status_input.parent.name / patient_status_input.name
    return _default_output_dir_from_med_evo_root(patient_status_input)


def _infer_snapshot_window_index(snapshot: Dict[str, Any]) -> Optional[int]:
    window_id = snapshot.get("last_processed_window_index")
    if window_id is None:
        return None
    return int(window_id)


def _collect_windowed_snapshots(memory_snapshots: Sequence[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    snapshots_by_window: Dict[int, Dict[str, Any]] = {}
    for snapshot in memory_snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid memory snapshot row; expected dict.")
        window_index = _infer_snapshot_window_index(snapshot)
        if window_index is None:
            continue
        snapshots_by_window[int(window_index)] = snapshot
    return sorted(snapshots_by_window.items(), key=lambda item: item[0])


def _load_memory_snapshots(patient_dir: Path) -> List[Dict[str, Any]]:
    memory_db_path = patient_dir / "memory_database.json"
    payload = _load_json(memory_db_path)
    memory_snapshots = payload.get("memory_snapshots")
    if not isinstance(memory_snapshots, list):
        raise ValueError(f"Invalid memory_snapshots list in {memory_db_path}")
    return memory_snapshots


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


def _bootstrap_patient_status_macro_ci(
    frame: pd.DataFrame,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    if frame.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")

    patient_groups: List[Tuple[float, List[str], List[str]]] = []
    for _, patient_df in frame.groupby("patient_id"):
        patient_groups.append(
            (
                float(patient_df["is_correct"].mean()),
                [str(value) for value in patient_df["true_label"].astype(str).tolist()],
                [str(value) for value in patient_df["pred_label"].astype(str).tolist()],
            )
        )

    if len(patient_groups) == 1:
        patient_acc, true_labels, pred_labels = patient_groups[0]
        macro_f1 = _compute_macro_f1_from_label_series(true_labels, pred_labels)
        return patient_acc, patient_acc, macro_f1, macro_f1

    acc_samples: List[float] = []
    f1_samples: List[float] = []
    group_count = len(patient_groups)
    for _ in range(int(n_samples)):
        sampled_indices = rng.integers(0, group_count, size=group_count)
        sampled_acc: List[float] = []
        sampled_true: List[str] = []
        sampled_pred: List[str] = []
        for sampled_index in sampled_indices.tolist():
            patient_acc, true_labels, pred_labels = patient_groups[int(sampled_index)]
            sampled_acc.append(float(patient_acc))
            sampled_true.extend(true_labels)
            sampled_pred.extend(pred_labels)
        acc_samples.append(float(np.mean(sampled_acc)))
        f1_samples.append(float(_compute_macro_f1_from_label_series(sampled_true, sampled_pred)))

    acc_low = float(np.quantile(np.asarray(acc_samples, dtype=float), 0.025))
    acc_high = float(np.quantile(np.asarray(acc_samples, dtype=float), 0.975))
    f1_low = float(np.quantile(np.asarray(f1_samples, dtype=float), 0.025))
    f1_high = float(np.quantile(np.asarray(f1_samples, dtype=float), 0.975))
    return acc_low, acc_high, f1_low, f1_high


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


def _marker_best_window_index(marker: Mapping[str, Any]) -> Optional[int]:
    for key in ("stride_source_window_index", "source_window_index", "window_index"):
        value = marker.get(key)
        if value is None:
            continue
        return int(value)
    return None


def _load_gt_labels_by_stay(
    gt_dir: Path,
) -> Tuple[Dict[int, Dict[Tuple[float, float], WindowLabel]], Dict[int, Set[Tuple[float, float]]]]:
    labels_by_stay: Dict[int, Dict[Tuple[float, float], WindowLabel]] = {}
    windows_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    for pred_path in sorted((gt_dir / "patients").glob("*/oracle_predictions.json")):
        payload = _load_json(pred_path)
        icu_stay_id = _extract_icu_stay_id(payload, pred_path)
        if int(icu_stay_id) in labels_by_stay:
            raise ValueError(f"Duplicate GT entries for icu_stay_id={icu_stay_id} in {gt_dir}")
        window_outputs = payload.get("window_outputs")
        if not isinstance(window_outputs, list) or not window_outputs:
            raise ValueError(f"Missing or empty window_outputs in {pred_path}")

        labels_by_interval: Dict[Tuple[float, float], str] = {}
        for output in window_outputs:
            if not isinstance(output, Mapping):
                raise ValueError(f"Invalid window output row in {pred_path}")
            interval_key = _resolve_gt_window_interval(output, pred_path)
            if interval_key in labels_by_interval:
                raise ValueError(f"Duplicate GT interval={interval_key} in {pred_path}")
            oracle_output = output.get("oracle_output")
            if not isinstance(oracle_output, Mapping):
                raise ValueError(f"Missing oracle_output in {pred_path} for icu_stay_id={icu_stay_id}")
            labels_by_interval[interval_key] = normalize_status_label(extract_overall_label(oracle_output))

        ordered_intervals = sorted(labels_by_interval.keys(), key=lambda interval: (float(interval[0]), float(interval[1])))
        total_windows = len(ordered_intervals)
        stay_labels: Dict[Tuple[float, float], WindowLabel] = {}
        for order, interval in enumerate(ordered_intervals):
            stay_labels[interval] = WindowLabel(
                label=str(labels_by_interval[interval]),
                hours_since_admission=float(interval[0]),
                num_windows=int(total_windows),
                window_order=int(order),
                interval_start_hour=float(interval[0]),
                interval_end_hour=float(interval[1]),
                marker_id=_interval_marker_id(interval),
            )
        labels_by_stay[int(icu_stay_id)] = stay_labels
        windows_by_stay[int(icu_stay_id)] = set(ordered_intervals)
    if not labels_by_stay:
        raise ValueError(f"No GT labels loaded from {gt_dir}")
    return labels_by_stay, windows_by_stay


def _memory_depth_by_window(source_patient_dir: Path) -> Dict[int, int]:
    memory_snapshots = _load_memory_snapshots(source_patient_dir)
    windowed = _collect_windowed_snapshots(memory_snapshots)
    depths: Dict[int, int] = {}
    for window_index, snapshot in windowed:
        if not isinstance(snapshot, dict):
            raise ValueError(f"Invalid memory snapshot object in {source_patient_dir}")
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


def _collect_prediction_windows_by_stay(prediction_dir: Path) -> Dict[int, Set[Tuple[float, float]]]:
    prediction_files = sorted((prediction_dir / "patients").glob("*/patient_status_predictions.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No patient_status_predictions.json files found under {prediction_dir / 'patients'}")
    windows_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    for pred_path in prediction_files:
        payload = _load_json(pred_path)
        icu_stay_id = _extract_icu_stay_id(payload, pred_path)
        if int(icu_stay_id) in windows_by_stay:
            raise ValueError(f"Duplicate prediction entries for icu_stay_id={icu_stay_id} in {prediction_dir}")
        status_predictions = payload.get("status_predictions")
        if not isinstance(status_predictions, list):
            raise ValueError(f"Missing status_predictions list in {pred_path}")
        prediction_intervals: Set[Tuple[float, float]] = set()
        for item in status_predictions:
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid status prediction row in {pred_path}")
            prediction_intervals.add(_resolve_prediction_interval(item=item, source_path=pred_path))
        windows_by_stay[int(icu_stay_id)] = prediction_intervals
    return windows_by_stay


def _load_prediction_rows(
    prediction_dir: Path,
    gt_labels_by_stay: Dict[int, Dict[Tuple[float, float], WindowLabel]],
    matched_windows_by_stay: Dict[int, Set[Tuple[float, float]]],
    system_name: str,
    include_memory_depth: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    prediction_files = sorted((prediction_dir / "patients").glob("*/patient_status_predictions.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No patient_status_predictions.json files found under {prediction_dir / 'patients'}")

    for pred_path in prediction_files:
        payload = _load_json(pred_path)
        patient_id = pred_path.parent.name
        subject_id_raw = payload.get("subject_id")
        if subject_id_raw is None:
            raise ValueError(f"Missing subject_id in {pred_path}")
        icu_stay_id = _extract_icu_stay_id(payload, pred_path)
        matched_windows = matched_windows_by_stay.get(int(icu_stay_id))
        if matched_windows is None:
            continue
        subject_id = int(subject_id_raw)

        status_predictions = payload.get("status_predictions")
        if not isinstance(status_predictions, list):
            raise ValueError(f"Missing status_predictions list in {pred_path}")

        gt_map = gt_labels_by_stay.get(int(icu_stay_id))
        if gt_map is None:
            continue

        memory_depth_map: Dict[int, int] = {}
        if include_memory_depth:
            source_patient_dir_raw = payload.get("source_patient_dir")
            if source_patient_dir_raw is None:
                raise ValueError(f"Missing source_patient_dir in {pred_path}")
            memory_depth_map = _memory_depth_by_window(Path(str(source_patient_dir_raw)))

        for item in status_predictions:
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid status prediction row in {pred_path}")
            interval_key = _resolve_prediction_interval(item=item, source_path=pred_path)
            if interval_key not in matched_windows:
                continue
            gt_row = gt_map.get(interval_key)
            if gt_row is None:
                continue
            window_raw = item.get("window_index")
            if window_raw is None:
                window_index = int(gt_row.window_order)
            else:
                window_index = int(window_raw)
                if window_index < 0:
                    raise ValueError(f"Negative agent window_index={window_index} in {pred_path}")

            pred_label = normalize_status_label(item.get("status_label"))
            true_label = gt_row.label
            num_windows = int(gt_row.num_windows)
            if num_windows > 1:
                relative_time = float(gt_row.window_order) / float(num_windows - 1)
            else:
                relative_time = 0.0
            relative_time = min(max(relative_time, 0.0), 1.0)
            time_bin = assign_normalized_time_bin(relative_time, num_bins=NUM_TIME_BINS)
            hours_raw = item.get("hours_since_admission")
            if hours_raw is None:
                hours_since_admission = float(gt_row.interval_start_hour)
            else:
                hours_since_admission = float(hours_raw)
            marker = item.get("oracle_window_marker")
            depth_lookup_window_index = int(window_index)
            if isinstance(marker, Mapping):
                marker_window_index = _marker_best_window_index(marker)
                if marker_window_index is not None:
                    depth_lookup_window_index = int(marker_window_index)
            memory_depth = memory_depth_map.get(depth_lookup_window_index)
            window_start_hour = float(item.get("window_start_hour") or gt_row.interval_start_hour)
            window_end_hour = float(item.get("window_end_hour") or gt_row.interval_end_hour)
            marker_id = str(item.get("oracle_window_marker_id") or gt_row.marker_id).strip()
            if not marker_id:
                marker_id = str(gt_row.marker_id)

            rows.append(
                {
                    "system": system_name,
                    "patient_id": patient_id,
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "window_index": window_index,
                    "gt_window_order": int(gt_row.window_order),
                    "num_windows": num_windows,
                    "relative_time": relative_time,
                    "time_bin": time_bin,
                    "hours_since_admission": float(hours_since_admission),
                    "window_start_hour": float(window_start_hour),
                    "window_end_hour": float(window_end_hour),
                    "oracle_window_marker_id": marker_id,
                    "oracle_interval_start_hour": float(gt_row.interval_start_hour),
                    "oracle_interval_end_hour": float(gt_row.interval_end_hour),
                    "pred_label": pred_label,
                    "true_label": true_label,
                    "is_correct": int(pred_label == true_label),
                    "memory_depth": memory_depth,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            f"No matched prediction windows for system='{system_name}'. "
            "Check overlap between prediction windows and GT windows."
        )
    return frame


def _compute_accuracy_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    total_windows = int(len(frame))
    micro_acc = float(frame["is_correct"].mean()) if total_windows > 0 else float("nan")
    per_patient = frame.groupby("patient_id")["is_correct"].mean().astype(float)
    macro_acc = float(per_patient.mean()) if len(per_patient) > 0 else float("nan")
    true_labels = [str(value) for value in frame["true_label"].astype(str).tolist()]
    pred_labels = [str(value) for value in frame["pred_label"].astype(str).tolist()]
    label_space = sorted(set(true_labels) | set(pred_labels))
    per_label_f1: List[float] = []
    for label in label_space:
        tp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred != label)
        precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_label_f1.append(float(f1))
    macro_f1 = float(np.mean(per_label_f1)) if per_label_f1 else float("nan")
    return {
        "num_windows": total_windows,
        "num_patients": int(frame["patient_id"].nunique()),
        "micro_acc": micro_acc,
        "macro_acc": macro_acc,
        "macro_f1": macro_f1,
    }


def _compute_macro_f1_from_label_series(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    label_space = sorted(set(true_labels) | set(pred_labels))
    per_label_f1: List[float] = []
    for label in label_space:
        tp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred != label)
        precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_label_f1.append(float(f1))
    return float(np.mean(per_label_f1)) if per_label_f1 else float("nan")


def _build_absolute_hour_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized, max_bin, _ = _absolute_hour_bin_frame(frame)
    plot_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for system_name, system_df in normalized.groupby("system"):
        for hour_bin in range(max_bin + 1):
            subset = system_df[system_df["absolute_hour_bin"] == int(hour_bin)].copy()
            if subset.empty:
                macro_acc = float("nan")
                macro_f1 = float("nan")
                macro_acc_ci_low = float("nan")
                macro_acc_ci_high = float("nan")
                macro_f1_ci_low = float("nan")
                macro_f1_ci_high = float("nan")
                num_patients = 0
            else:
                patient_acc = subset.groupby("patient_id")["is_correct"].mean().astype(float)
                macro_acc = float(patient_acc.mean()) if len(patient_acc) > 0 else float("nan")
                macro_f1 = _compute_macro_f1_from_label_series(
                    [str(value) for value in subset["true_label"].astype(str).tolist()],
                    [str(value) for value in subset["pred_label"].astype(str).tolist()],
                )
                (
                    macro_acc_ci_low,
                    macro_acc_ci_high,
                    macro_f1_ci_low,
                    macro_f1_ci_high,
                ) = _bootstrap_patient_status_macro_ci(
                    subset,
                    n_samples=BOOTSTRAP_SAMPLES,
                    rng=rng,
                )
                num_patients = int(subset["patient_id"].nunique())
            plot_rows.append(
                {
                    "system": str(system_name),
                    "absolute_hour_bin": int(hour_bin),
                    "hour_start": float(hour_bin * float(ABSOLUTE_TIME_BIN_HOURS)),
                    "hour_end": float((hour_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS)),
                    "hour_mid": float((hour_bin + 0.5) * float(ABSOLUTE_TIME_BIN_HOURS)),
                    "n_windows": int(len(subset)),
                    "num_patients": int(num_patients),
                    "macro_acc": float(macro_acc),
                    "macro_acc_ci_low": float(macro_acc_ci_low),
                    "macro_acc_ci_high": float(macro_acc_ci_high),
                    "macro_f1": float(macro_f1),
                    "macro_f1_ci_low": float(macro_f1_ci_low),
                    "macro_f1_ci_high": float(macro_f1_ci_high),
                }
            )
    plot_df = pd.DataFrame(plot_rows).sort_values(["system", "absolute_hour_bin"]).reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No absolute-hour patient-status metric rows generated.")
    return plot_df


def _plot_absolute_hour_metric_frame(
    curve_df: pd.DataFrame,
    *,
    group_column: str,
    metric_column: str,
    metric_label: str,
    title: str,
    output_path: Path,
) -> None:
    if curve_df.empty:
        raise ValueError(f"No rows available for {metric_column} plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    max_hour = 0.0
    color_by_group = (
        {group: system_color(str(group)) for group in curve_df[group_column].astype(str).unique().tolist()}
        if group_column == "system"
        else mode_color_map(sorted(curve_df[group_column].astype(str).unique().tolist()))
    )
    for group_name in sorted(curve_df[group_column].astype(str).unique().tolist()):
        subset = curve_df[curve_df[group_column].astype(str) == str(group_name)].sort_values("absolute_hour_bin")
        if subset.empty:
            continue
        x = subset["hour_mid"].to_numpy(dtype=float)
        y = smooth_series(subset[metric_column].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        ax.plot(
            x,
            y,
            linewidth=2.2,
            label=str(group_name),
            color=color_by_group[str(group_name)],
            alpha=MODE_TRACE_ALPHA if group_column != "system" else 1.0,
        )
        ci_low_column = f"{metric_column}_ci_low"
        ci_high_column = f"{metric_column}_ci_high"
        if ci_low_column in subset.columns and ci_high_column in subset.columns:
            _, low, high = _smooth_metric_and_band(
                subset[metric_column].to_numpy(dtype=float),
                subset[ci_low_column].to_numpy(dtype=float),
                subset[ci_high_column].to_numpy(dtype=float),
            )
            ax.fill_between(
                x,
                low,
                high,
                alpha=METRIC_BAND_ALPHA,
                color=color_by_group[str(group_name)],
            )
        max_hour = max(max_hour, float(subset["hour_end"].max()))

    ax.set_xlabel("Hours Since ICU Admission")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.set_xlim(0, max_hour if max_hour > 0 else float(ABSOLUTE_TIME_BIN_HOURS))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def _plot_relative_time_accuracy(
    frame: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    plot_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for system_name, system_df in frame.groupby("system"):
        for time_bin in range(NUM_TIME_BINS):
            subset = system_df[system_df["time_bin"] == time_bin]
            values = subset["is_correct"].astype(float).tolist()
            mean_acc = float(np.mean(values)) if values else float("nan")
            ci_low, ci_high = _bootstrap_mean_ci(values, n_samples=BOOTSTRAP_SAMPLES, rng=rng)
            plot_rows.append(
                {
                    "system": system_name,
                    "time_bin": int(time_bin),
                    "time_bin_mid_pct": float((time_bin + 0.5) * (100.0 / NUM_TIME_BINS)),
                    "n_windows": int(len(values)),
                    "accuracy": mean_acc,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    plot_df = pd.DataFrame(plot_rows).sort_values(["system", "time_bin"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for system_name in plot_df["system"].unique().tolist():
        subset = plot_df[plot_df["system"] == system_name].copy()
        x = subset["time_bin_mid_pct"].to_numpy(dtype=float)
        y = subset["accuracy"].to_numpy(dtype=float)
        low = subset["ci_low"].to_numpy(dtype=float)
        high = subset["ci_high"].to_numpy(dtype=float)
        color = system_color(str(system_name))
        ax.plot(x, y, marker="o", linewidth=2.2, label=system_name, color=color)
        ax.fill_between(x, low, high, alpha=METRIC_BAND_ALPHA, color=color)

    ax.set_xlabel("Relative Window Position (%)")
    ax.set_ylabel("Window-Level Accuracy")
    ax.set_title("Relative-Time Accuracy Curve")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _plot_prefix_accuracy(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    med_evo = frame[frame["system"] == "ICU-Evo"].copy()
    if med_evo.empty:
        raise ValueError("Prefix accuracy plot requires ICU-Evo rows.")

    med_evo = med_evo.sort_values(["patient_id", "relative_time", "window_index"]).reset_index(drop=True)
    med_evo["cum_correct"] = med_evo.groupby("patient_id")["is_correct"].cumsum()
    med_evo["cum_count"] = med_evo.groupby("patient_id").cumcount() + 1
    med_evo["prefix_accuracy"] = med_evo["cum_correct"] / med_evo["cum_count"]

    thresholds = np.linspace(0.0, 1.0, NUM_TIME_BINS + 1)
    mean_rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        patient_values: List[float] = []
        for _, patient_df in med_evo.groupby("patient_id"):
            valid = patient_df[patient_df["relative_time"] <= threshold]
            if valid.empty:
                continue
            patient_values.append(float(valid["prefix_accuracy"].iloc[-1]))
        mean_rows.append(
            {
                "relative_time_pct": float(threshold * 100.0),
                "num_patients": int(len(patient_values)),
                "cohort_mean_prefix_accuracy": float(np.mean(patient_values)) if patient_values else float("nan"),
            }
        )
    mean_df = pd.DataFrame(mean_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    for _, patient_df in med_evo.groupby("patient_id"):
        x = patient_df["relative_time"].to_numpy(dtype=float) * 100.0
        y = patient_df["prefix_accuracy"].to_numpy(dtype=float)
        ax.plot(x, y, color=metric_color("accuracy"), alpha=PATIENT_TRACE_ALPHA, linewidth=1.0)

    ax.plot(
        mean_df["relative_time_pct"].to_numpy(dtype=float),
        mean_df["cohort_mean_prefix_accuracy"].to_numpy(dtype=float),
        color=metric_mean_color("accuracy"),
        linewidth=3.0,
        marker="o",
        label="Cohort Mean (ICU-Evo)",
    )
    ax.set_xlabel("Relative Time (%)")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Prefix (Cumulative) Accuracy Curve")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return mean_df


def _absolute_hour_bin_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    if "hours_since_admission" not in frame.columns:
        raise ValueError("Absolute-hour plots require hours_since_admission column.")

    normalized = frame.copy()
    normalized["hours_since_admission"] = pd.to_numeric(normalized["hours_since_admission"], errors="coerce")
    normalized = normalized.dropna(subset=["hours_since_admission"]).copy()
    if normalized.empty:
        raise ValueError("Absolute-hour plots require non-empty hours_since_admission values.")
    if (normalized["hours_since_admission"] < 0).any():
        raise ValueError("Absolute-hour plots require non-negative hours_since_admission values.")

    max_hour = float(normalized["hours_since_admission"].max())
    max_bin = int(math.floor(max_hour / float(ABSOLUTE_TIME_BIN_HOURS)))
    max_edge = float((max_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS))
    normalized["absolute_hour_bin"] = np.floor(
        normalized["hours_since_admission"].to_numpy(dtype=float) / float(ABSOLUTE_TIME_BIN_HOURS)
    ).astype(int)
    return normalized, max_bin, max_edge


def _plot_absolute_hour_accuracy(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    normalized, max_bin, max_edge = _absolute_hour_bin_frame(frame)

    plot_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for system_name, system_df in normalized.groupby("system"):
        for hour_bin in range(max_bin + 1):
            subset = system_df[system_df["absolute_hour_bin"] == int(hour_bin)]
            values = subset["is_correct"].astype(float).tolist()
            mean_acc = float(np.mean(values)) if values else float("nan")
            ci_low, ci_high = _bootstrap_mean_ci(values, n_samples=BOOTSTRAP_SAMPLES, rng=rng)
            plot_rows.append(
                {
                    "system": str(system_name),
                    "absolute_hour_bin": int(hour_bin),
                    "hour_start": float(hour_bin * float(ABSOLUTE_TIME_BIN_HOURS)),
                    "hour_end": float((hour_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS)),
                    "hour_mid": float((hour_bin + 0.5) * float(ABSOLUTE_TIME_BIN_HOURS)),
                    "n_windows": int(len(values)),
                    "accuracy": mean_acc,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    plot_df = pd.DataFrame(plot_rows).sort_values(["system", "absolute_hour_bin"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for system_name in plot_df["system"].unique().tolist():
        subset = plot_df[plot_df["system"] == system_name].copy()
        x = subset["hour_mid"].to_numpy(dtype=float)
        y = subset["accuracy"].to_numpy(dtype=float)
        low = subset["ci_low"].to_numpy(dtype=float)
        high = subset["ci_high"].to_numpy(dtype=float)
        color = system_color(str(system_name))
        ax.plot(x, y, marker="o", linewidth=2.2, label=system_name, color=color)
        ax.fill_between(x, low, high, alpha=METRIC_BAND_ALPHA, color=color)

    ax.set_xlabel("Hours Since ICU Admission")
    ax.set_ylabel("Window-Level Accuracy")
    ax.set_title("Absolute-Time Accuracy Curve")
    ax.set_xlim(0, max_edge)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _plot_absolute_hour_prefix_accuracy(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    med_evo = frame[frame["system"] == "ICU-Evo"].copy()
    if med_evo.empty:
        raise ValueError("Absolute-time prefix accuracy plot requires ICU-Evo rows.")

    med_evo, max_bin, max_edge = _absolute_hour_bin_frame(med_evo)
    med_evo = med_evo.sort_values(["patient_id", "hours_since_admission", "window_index"]).reset_index(drop=True)
    med_evo["cum_correct"] = med_evo.groupby("patient_id")["is_correct"].cumsum()
    med_evo["cum_count"] = med_evo.groupby("patient_id").cumcount() + 1
    med_evo["prefix_accuracy"] = med_evo["cum_correct"] / med_evo["cum_count"]

    thresholds = [float(index * float(ABSOLUTE_TIME_BIN_HOURS)) for index in range(max_bin + 2)]
    mean_rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        patient_values: List[float] = []
        for _, patient_df in med_evo.groupby("patient_id"):
            valid = patient_df[patient_df["hours_since_admission"] <= float(threshold)]
            if valid.empty:
                continue
            patient_values.append(float(valid["prefix_accuracy"].iloc[-1]))
        mean_rows.append(
            {
                "hours_since_admission": float(threshold),
                "num_patients": int(len(patient_values)),
                "cohort_mean_prefix_accuracy": float(np.mean(patient_values)) if patient_values else float("nan"),
            }
        )
    mean_df = pd.DataFrame(mean_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    for _, patient_df in med_evo.groupby("patient_id"):
        x = patient_df["hours_since_admission"].to_numpy(dtype=float)
        y = patient_df["prefix_accuracy"].to_numpy(dtype=float)
        ax.plot(x, y, color=metric_color("accuracy"), alpha=PATIENT_TRACE_ALPHA, linewidth=1.0)

    ax.plot(
        mean_df["hours_since_admission"].to_numpy(dtype=float),
        mean_df["cohort_mean_prefix_accuracy"].to_numpy(dtype=float),
        color=metric_mean_color("accuracy"),
        linewidth=3.0,
        marker="o",
        label="Cohort Mean (ICU-Evo)",
    )
    ax.set_xlabel("Hours Since ICU Admission")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Absolute-Time Prefix Cumulative Accuracy Curve")
    ax.set_xlim(0, max_edge)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return mean_df


def _plot_relative_time_accuracy_by_mode(
    curves_by_mode: Dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError("No mode curves provided for relative-time comparison plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    color_by_mode = mode_color_map(sorted(curves_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    for mode_name in sorted(curves_by_mode.keys()):
        curve_df = curves_by_mode[mode_name]
        med_evo_curve = curve_df[curve_df["system"] == "ICU-Evo"].copy()
        if med_evo_curve.empty:
            continue
        ordered = med_evo_curve.sort_values("time_bin").reset_index(drop=True)
        x = ordered["time_bin_mid_pct"].to_numpy(dtype=float)
        y = ordered["accuracy"].to_numpy(dtype=float)
        color = color_by_mode[str(mode_name)]
        ax.plot(x, y, marker="o", linewidth=2.2, label=str(mode_name), color=color, alpha=MODE_TRACE_ALPHA)
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_name),
                    "time_bin": int(row["time_bin"]),
                    "time_bin_mid_pct": float(row["time_bin_mid_pct"]),
                    "n_windows": int(row["n_windows"]),
                    "accuracy": float(row["accuracy"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

    ax.set_xlabel("Relative Window Position (%)")
    ax.set_ylabel("Window-Level Accuracy")
    ax.set_title("Relative-Time Accuracy by Mode")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for relative-time mode comparison plot.")
    return comparison_df


def _plot_prefix_accuracy_by_mode(
    curves_by_mode: Dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError("No mode curves provided for prefix comparison plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    color_by_mode = mode_color_map(sorted(curves_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    for mode_name in sorted(curves_by_mode.keys()):
        curve_df = curves_by_mode[mode_name]
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("relative_time_pct").reset_index(drop=True)
        x = ordered["relative_time_pct"].to_numpy(dtype=float)
        y = ordered["cohort_mean_prefix_accuracy"].to_numpy(dtype=float)
        color = color_by_mode[str(mode_name)]
        ax.plot(x, y, marker="o", linewidth=2.2, label=str(mode_name), color=color, alpha=MODE_TRACE_ALPHA)
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_name),
                    "relative_time_pct": float(row["relative_time_pct"]),
                    "num_patients": int(row["num_patients"]),
                    "cohort_mean_prefix_accuracy": float(row["cohort_mean_prefix_accuracy"]),
                }
            )

    ax.set_xlabel("Relative Time (%)")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Prefix Cumulative Accuracy by Mode")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for prefix mode comparison plot.")
    return comparison_df


def _plot_absolute_hour_accuracy_by_mode(
    curves_by_mode: Dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError("No mode curves provided for absolute-hour comparison plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    color_by_mode = mode_color_map(sorted(curves_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    max_hour = 0.0
    for mode_name in sorted(curves_by_mode.keys()):
        curve_df = curves_by_mode[mode_name]
        med_evo_curve = curve_df[curve_df["system"] == "ICU-Evo"].copy()
        if med_evo_curve.empty:
            continue
        ordered = med_evo_curve.sort_values("absolute_hour_bin").reset_index(drop=True)
        x = ordered["hour_mid"].to_numpy(dtype=float)
        y = ordered["accuracy"].to_numpy(dtype=float)
        color = color_by_mode[str(mode_name)]
        ax.plot(x, y, marker="o", linewidth=2.2, label=str(mode_name), color=color, alpha=MODE_TRACE_ALPHA)
        max_hour = max(max_hour, float(ordered["hour_end"].max()))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_name),
                    "absolute_hour_bin": int(row["absolute_hour_bin"]),
                    "hour_start": float(row["hour_start"]),
                    "hour_end": float(row["hour_end"]),
                    "hour_mid": float(row["hour_mid"]),
                    "n_windows": int(row["n_windows"]),
                    "accuracy": float(row["accuracy"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

    ax.set_xlabel("Hours Since ICU Admission")
    ax.set_ylabel("Window-Level Accuracy")
    ax.set_title("Absolute-Time Accuracy by Mode")
    ax.set_xlim(0, max_hour)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for absolute-hour mode comparison plot.")
    return comparison_df


def _pad_absolute_hour_mode_curve(
    curve_df: pd.DataFrame,
    *,
    max_bin: int,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    ordered = curve_df.sort_values("absolute_hour_bin").reset_index(drop=True)
    if ordered.empty:
        return ordered
    max_mode_bin = int(ordered["absolute_hour_bin"].max())
    if max_mode_bin >= int(max_bin):
        return ordered

    template_row = ordered.iloc[-1].to_dict()
    tail_rows: List[Dict[str, Any]] = []
    for hour_bin in range(max_mode_bin + 1, int(max_bin) + 1):
        padded_row = dict(template_row)
        padded_row["absolute_hour_bin"] = int(hour_bin)
        padded_row["hour_start"] = float(hour_bin * float(ABSOLUTE_TIME_BIN_HOURS))
        padded_row["hour_end"] = float((hour_bin + 1) * float(ABSOLUTE_TIME_BIN_HOURS))
        padded_row["hour_mid"] = float((hour_bin + 0.5) * float(ABSOLUTE_TIME_BIN_HOURS))
        padded_row["n_windows"] = 0
        padded_row["num_patients"] = 0
        for metric_column in metric_columns:
            padded_row[metric_column] = 0.0
            ci_low_column = f"{metric_column}_ci_low"
            ci_high_column = f"{metric_column}_ci_high"
            if ci_low_column in padded_row:
                padded_row[ci_low_column] = 0.0
            if ci_high_column in padded_row:
                padded_row[ci_high_column] = 0.0
        tail_rows.append(padded_row)
    if not tail_rows:
        return ordered
    return pd.concat([ordered, pd.DataFrame(tail_rows)], ignore_index=True)


def _plot_absolute_hour_acc_f1_curve_by_mode(
    curves_by_mode: Dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError("No mode curves provided for patient-status absolute-hour comparison plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color_by_mode = mode_color_map(sorted(curves_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    max_hour = 0.0
    global_max_bin = max(
        int(curve_df["absolute_hour_bin"].max())
        for curve_df in curves_by_mode.values()
        if not curve_df.empty and "absolute_hour_bin" in curve_df.columns
    )

    for mode_name in sorted(curves_by_mode.keys()):
        curve_df = curves_by_mode[mode_name]
        med_evo_curve = curve_df[curve_df["system"] == "ICU-Evo"].copy() if "system" in curve_df.columns else curve_df.copy()
        if med_evo_curve.empty:
            continue
        ordered = _pad_absolute_hour_mode_curve(
            med_evo_curve,
            max_bin=global_max_bin,
            metric_columns=("macro_acc", "macro_f1"),
        )
        x = ordered["hour_mid"].to_numpy(dtype=float)
        y_acc = smooth_series(ordered["macro_acc"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        y_f1 = smooth_series(ordered["macro_f1"].to_numpy(dtype=float), window=CURVE_SMOOTHING_WINDOW)
        color = color_by_mode[str(mode_name)]

        if {"macro_acc_ci_low", "macro_acc_ci_high"}.issubset(set(ordered.columns)):
            _, acc_low, acc_high = _smooth_metric_and_band(
                ordered["macro_acc"].to_numpy(dtype=float),
                ordered["macro_acc_ci_low"].to_numpy(dtype=float),
                ordered["macro_acc_ci_high"].to_numpy(dtype=float),
            )
            axes[0].fill_between(x, acc_low, acc_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[0].plot(x, y_acc, linewidth=2.2, label=str(mode_name), color=color, alpha=MODE_TRACE_ALPHA)
        if {"macro_f1_ci_low", "macro_f1_ci_high"}.issubset(set(ordered.columns)):
            _, f1_low, f1_high = _smooth_metric_and_band(
                ordered["macro_f1"].to_numpy(dtype=float),
                ordered["macro_f1_ci_low"].to_numpy(dtype=float),
                ordered["macro_f1_ci_high"].to_numpy(dtype=float),
            )
            axes[1].fill_between(x, f1_low, f1_high, color=color, alpha=METRIC_BAND_ALPHA * 0.8)
        axes[1].plot(x, y_f1, linewidth=2.2, label=str(mode_name), color=color, alpha=MODE_TRACE_ALPHA)

        max_hour = max(max_hour, float(ordered["hour_end"].max()))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_name),
                    "absolute_hour_bin": int(row["absolute_hour_bin"]),
                    "hour_start": float(row["hour_start"]),
                    "hour_end": float(row["hour_end"]),
                    "hour_mid": float(row["hour_mid"]),
                    "n_windows": int(row["n_windows"]),
                    "num_patients": int(row["num_patients"]),
                    "macro_acc": float(row["macro_acc"]),
                    "macro_acc_ci_low": float(row["macro_acc_ci_low"]) if "macro_acc_ci_low" in row else float("nan"),
                    "macro_acc_ci_high": float(row["macro_acc_ci_high"]) if "macro_acc_ci_high" in row else float("nan"),
                    "macro_f1": float(row["macro_f1"]),
                    "macro_f1_ci_low": float(row["macro_f1_ci_low"]) if "macro_f1_ci_low" in row else float("nan"),
                    "macro_f1_ci_high": float(row["macro_f1_ci_high"]) if "macro_f1_ci_high" in row else float("nan"),
                }
            )

    axes[0].set_xlabel("Hours Since ICU Admission")
    axes[0].set_ylabel("Macro Accuracy")
    axes[0].set_title("Patient Status: Absolute-Time Macro Accuracy by Mode")
    axes[0].set_xlim(0, max_hour if max_hour > 0 else float(ABSOLUTE_TIME_BIN_HOURS))
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Hours Since ICU Admission")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Patient Status: Absolute-Time Macro F1 by Mode")
    axes[1].set_xlim(0, max_hour if max_hour > 0 else float(ABSOLUTE_TIME_BIN_HOURS))
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError("No rows generated for patient-status absolute-hour acc/f1 comparison plot.")
    return output_df


def _plot_absolute_hour_prefix_accuracy_by_mode(
    curves_by_mode: Dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError("No mode curves provided for absolute-hour prefix comparison plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    color_by_mode = mode_color_map(sorted(curves_by_mode.keys()))
    rows: List[Dict[str, Any]] = []
    max_hour = 0.0
    for mode_name in sorted(curves_by_mode.keys()):
        curve_df = curves_by_mode[mode_name]
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("hours_since_admission").reset_index(drop=True)
        x = ordered["hours_since_admission"].to_numpy(dtype=float)
        y = ordered["cohort_mean_prefix_accuracy"].to_numpy(dtype=float)
        color = color_by_mode[str(mode_name)]
        ax.plot(x, y, marker="o", linewidth=2.2, label=str(mode_name), color=color, alpha=MODE_TRACE_ALPHA)
        max_hour = max(max_hour, float(ordered["hours_since_admission"].max()))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_name),
                    "hours_since_admission": float(row["hours_since_admission"]),
                    "num_patients": int(row["num_patients"]),
                    "cohort_mean_prefix_accuracy": float(row["cohort_mean_prefix_accuracy"]),
                }
            )

    ax.set_xlabel("Hours Since ICU Admission")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Absolute-Time Prefix Cumulative Accuracy by Mode")
    ax.set_xlim(0, max_hour)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for absolute-hour prefix mode comparison plot.")
    return comparison_df


def _plot_time_memory_heatmap(frame: pd.DataFrame, output_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med_evo = frame[frame["system"] == "ICU-Evo"].copy()
    if med_evo.empty:
        raise ValueError("Time-memory heatmap requires ICU-Evo rows.")
    if med_evo["memory_depth"].isna().all():
        raise ValueError("Time-memory heatmap requires non-empty ICU-Evo memory depth values.")

    bin_codes, bin_labels = _memory_depth_bins(med_evo["memory_depth"])
    med_evo["memory_depth_bin"] = bin_codes
    valid = med_evo.dropna(subset=["memory_depth_bin"]).copy()
    valid["memory_depth_bin"] = valid["memory_depth_bin"].astype(int)

    mean_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="is_correct",
            aggfunc="mean",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
    )
    count_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="is_correct",
            aggfunc="count",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    heatmap_data = mean_table.to_numpy(dtype=float)
    image = ax.imshow(heatmap_data, aspect="auto", origin="lower", cmap=heatmap_cmap("accuracy"), vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Accuracy")

    x_tick_labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(NUM_TIME_BINS)]
    y_tick_labels = [bin_labels[i] if i < len(bin_labels) else str(i) for i in mean_table.index.tolist()]
    ax.set_xticks(np.arange(NUM_TIME_BINS))
    ax.set_xticklabels(x_tick_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("Relative-Time Decile")
    ax.set_ylabel("Memory-Depth Bin")
    ax.set_title("Time × Memory Accuracy Heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    accuracy_long = mean_table.reset_index().melt(
        id_vars=["memory_depth_bin"], var_name="time_bin", value_name="accuracy"
    )
    counts_long = count_table.reset_index().melt(
        id_vars=["memory_depth_bin"], var_name="time_bin", value_name="count"
    )
    merged = accuracy_long.merge(counts_long, on=["memory_depth_bin", "time_bin"], how="left")
    merged["memory_depth_label"] = merged["memory_depth_bin"].map(
        lambda value: bin_labels[int(value)] if int(value) < len(bin_labels) else str(value)
    )
    return merged, med_evo[["patient_id", "window_index", "time_bin", "memory_depth", "is_correct"]].copy()


def _save_metrics(
    output_path: Path,
    *,
    med_evo_metrics: Dict[str, Any],
    baseline_metrics: Optional[Dict[str, Any]],
    med_evo_frame: pd.DataFrame,
    baseline_frame: Optional[pd.DataFrame],
) -> None:
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "systems": {
            "ICU-Evo": {
                "macro_acc": float(med_evo_metrics["macro_acc"]),
                "macro_f1": float(med_evo_metrics["macro_f1"]),
            },
        },
    }
    if baseline_metrics is not None and baseline_frame is not None:
        payload["systems"]["Baseline"] = {
            "macro_acc": float(baseline_metrics["macro_acc"]),
            "macro_f1": float(baseline_metrics["macro_f1"]),
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _match_prediction_gt_intervals_by_stay(
    *,
    prediction_windows_by_stay: Dict[int, Set[Tuple[float, float]]],
    gt_windows_by_stay: Dict[int, Set[Tuple[float, float]]],
) -> Tuple[Dict[int, Set[Tuple[float, float]]], Dict[str, int]]:
    matched_windows_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    skipped_missing_gt_stays = 0
    skipped_unmatched_prediction_windows = 0
    matched_windows_total = 0
    for icu_stay_id, prediction_intervals in prediction_windows_by_stay.items():
        gt_intervals = gt_windows_by_stay.get(int(icu_stay_id))
        if gt_intervals is None:
            skipped_missing_gt_stays += 1
            skipped_unmatched_prediction_windows += int(len(prediction_intervals))
            continue
        shared_intervals = prediction_intervals.intersection(gt_intervals)
        skipped_unmatched_prediction_windows += int(len(prediction_intervals) - len(shared_intervals))
        if shared_intervals:
            matched_windows_by_stay[int(icu_stay_id)] = set(shared_intervals)
            matched_windows_total += int(len(shared_intervals))
    stats = {
        "prediction_stays": int(len(prediction_windows_by_stay)),
        "gt_stays": int(len(gt_windows_by_stay)),
        "matched_stays": int(len(matched_windows_by_stay)),
        "prediction_windows": int(sum(len(v) for v in prediction_windows_by_stay.values())),
        "gt_windows": int(sum(len(v) for v in gt_windows_by_stay.values())),
        "matched_windows": int(matched_windows_total),
        "skipped_prediction_stays_missing_gt": int(skipped_missing_gt_stays),
        "skipped_prediction_windows_missing_gt": int(skipped_unmatched_prediction_windows),
    }
    return matched_windows_by_stay, stats


def _evaluate_single_prediction_dir(
    *,
    prediction_dir: Path,
    gt_labels_by_stay: Dict[int, Dict[Tuple[float, float], WindowLabel]],
    gt_windows_by_stay: Dict[int, Set[Tuple[float, float]]],
    output_dir: Path,
    baseline_root: Optional[Path],
) -> Dict[str, Any]:
    med_evo_prediction_windows_by_stay = _collect_prediction_windows_by_stay(prediction_dir)
    med_evo_matched_windows_by_stay, med_evo_window_match_stats = _match_prediction_gt_intervals_by_stay(
        prediction_windows_by_stay=med_evo_prediction_windows_by_stay,
        gt_windows_by_stay=gt_windows_by_stay,
    )
    med_evo_frame = _load_prediction_rows(
        prediction_dir=prediction_dir,
        gt_labels_by_stay=gt_labels_by_stay,
        matched_windows_by_stay=med_evo_matched_windows_by_stay,
        system_name="ICU-Evo",
        include_memory_depth=True,
    )

    baseline_frame: Optional[pd.DataFrame] = None
    baseline_window_match_stats: Optional[Dict[str, int]] = None
    if baseline_root is not None:
        baseline_prediction_windows_by_stay = _collect_prediction_windows_by_stay(baseline_root)
        baseline_matched_windows_by_stay, baseline_window_match_stats = _match_prediction_gt_intervals_by_stay(
            prediction_windows_by_stay=baseline_prediction_windows_by_stay,
            gt_windows_by_stay=gt_windows_by_stay,
        )
        baseline_frame = _load_prediction_rows(
            prediction_dir=baseline_root,
            gt_labels_by_stay=gt_labels_by_stay,
            matched_windows_by_stay=baseline_matched_windows_by_stay,
            system_name="Baseline",
            include_memory_depth=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    med_evo_metrics = _compute_accuracy_metrics(med_evo_frame)
    baseline_metrics = _compute_accuracy_metrics(baseline_frame) if baseline_frame is not None else None

    all_frames = [med_evo_frame]
    if baseline_frame is not None:
        all_frames.append(baseline_frame)
    combined_frame = pd.concat(all_frames, ignore_index=True)

    metrics_path = output_dir / "metrics.json"
    _save_metrics(
        metrics_path,
        med_evo_metrics=med_evo_metrics,
        baseline_metrics=baseline_metrics,
        med_evo_frame=med_evo_frame,
        baseline_frame=baseline_frame,
    )

    combined_frame.sort_values(["system", "patient_id", "window_index"]).to_csv(
        output_dir / "window_level_matches.csv", index=False
    )

    absolute_hour_curve_df = _build_absolute_hour_metric_frame(combined_frame)
    absolute_hour_curve_df.to_csv(output_dir / "absolute_hour_patient_status_metrics.csv", index=False)
    absolute_hour_curve_df[["system", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_acc"]].to_csv(
        output_dir / "absolute_hour_patient_status_macro_acc.csv",
        index=False,
    )
    absolute_hour_curve_df[["system", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_f1"]].to_csv(
        output_dir / "absolute_hour_patient_status_macro_f1.csv",
        index=False,
    )

    print(f"Saved evaluation outputs to: {output_dir}")
    print("Status Metrics:")
    print(
        f"  ICU-Evo macro_acc={med_evo_metrics['macro_acc']:.4f} "
        f"macro_f1={med_evo_metrics['macro_f1']:.4f}"
    )
    if baseline_metrics is not None:
        print(
            f"  Baseline macro_acc={baseline_metrics['macro_acc']:.4f} "
            f"macro_f1={baseline_metrics['macro_f1']:.4f}"
        )

    return {
        "absolute_hour_curve_df": absolute_hour_curve_df,
        "output_dir": output_dir,
    }


def run_evaluation(
    *,
    med_evo_dir: Path,
    gt_dir: Path,
    output_dir: Optional[Path],
    baseline_dir: Optional[Path],
) -> None:
    gt_root = _ensure_gt_dir(gt_dir)
    gt_labels_by_stay, gt_windows_by_stay = _load_gt_labels_by_stay(gt_root)

    mode_root_candidates: List[Path] = [med_evo_dir]
    for root_name in PATIENT_STATUS_ROOT_NAMES:
        nested_candidate = med_evo_dir / root_name
        if nested_candidate not in mode_root_candidates:
            mode_root_candidates.append(nested_candidate)

    discovered_mode_dirs: Dict[str, Path] = {}
    resolved_mode_root: Optional[Path] = None
    for candidate in mode_root_candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        candidate_modes = _discover_patient_status_mode_dirs(candidate)
        if candidate_modes:
            discovered_mode_dirs = candidate_modes
            resolved_mode_root = candidate
            break

    if discovered_mode_dirs:
        if baseline_dir is not None:
            raise ValueError(
                "--baseline-dir is not supported when --icu-evo-dir points to a multi-mode patient-status root."
            )
        if resolved_mode_root is None:
            raise ValueError(f"Failed to resolve multi-mode root from {med_evo_dir}")
        output_root = (
            output_dir
            if output_dir is not None
            else _default_output_root_from_patient_status_input(resolved_mode_root)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        print(
            "Detected patient-status modes: "
            + ", ".join(sorted(discovered_mode_dirs.keys()))
            + f" under {resolved_mode_root}"
        )
        absolute_hour_curves_by_mode: Dict[str, pd.DataFrame] = {}

        for mode in sorted(discovered_mode_dirs.keys()):
            mode_prediction_dir = discovered_mode_dirs[mode]
            mode_output_dir = output_root / mode
            print(f"=== Evaluating mode: {mode} ===")
            mode_result = _evaluate_single_prediction_dir(
                prediction_dir=mode_prediction_dir,
                gt_labels_by_stay=gt_labels_by_stay,
                gt_windows_by_stay=gt_windows_by_stay,
                output_dir=mode_output_dir,
                baseline_root=None,
            )
            med_evo_curve = mode_result["absolute_hour_curve_df"][
                mode_result["absolute_hour_curve_df"]["system"] == "ICU-Evo"
            ].copy()
            med_evo_curve["mode"] = str(mode)
            absolute_hour_curves_by_mode[mode] = med_evo_curve

        if len(absolute_hour_curves_by_mode) > 1:
            combined_absolute_hour_df = pd.concat(
                [absolute_hour_curves_by_mode[mode] for mode in sorted(absolute_hour_curves_by_mode.keys())],
                ignore_index=True,
            )
            combined_absolute_hour_df.to_csv(
                output_root / "absolute_hour_patient_status_metrics_by_mode.csv",
                index=False,
            )
            combined_absolute_hour_df[
                ["mode", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_acc"]
            ].to_csv(output_root / "absolute_hour_patient_status_macro_acc_by_mode.csv", index=False)
            combined_absolute_hour_df[
                ["mode", "absolute_hour_bin", "hour_start", "hour_end", "hour_mid", "n_windows", "num_patients", "macro_f1"]
            ].to_csv(output_root / "absolute_hour_patient_status_macro_f1_by_mode.csv", index=False)

        print(f"Saved multi-mode evaluation outputs to: {output_root}")
        return

    med_evo_root = _ensure_predictions_dir(med_evo_dir)
    baseline_root = _ensure_predictions_dir(baseline_dir) if baseline_dir is not None else None
    final_output_dir = output_dir if output_dir is not None else _default_output_dir_from_med_evo_root(med_evo_root)
    _evaluate_single_prediction_dir(
        prediction_dir=med_evo_root,
        gt_labels_by_stay=gt_labels_by_stay,
        gt_windows_by_stay=gt_windows_by_stay,
        output_dir=final_output_dir,
        baseline_root=baseline_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate patient-status predictions against GT labels.")
    parser.add_argument(
        "--icu-evo-dir",
        type=str,
        required=True,
        help=(
            "Patient-status prediction directory for ICU-Evo. "
            "Supports single-mode input (<dir>/patients) or multi-mode root "
            "(<dir>/{memory,full_history_events,local_events_only})."
        ),
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="GT directory containing patients/*/oracle_predictions.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where metrics and CSV summaries will be saved. "
            "Default single-mode: evaluation_results/<memory_run_name>/<experiment_name>. "
            "Default multi-mode: evaluation_results/<memory_run_name>/<patient_status_root>/<mode>."
        ),
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        required=False,
        help="Optional patient-status prediction directory for baseline comparison (single-mode only).",
    )
    args = parser.parse_args()

    run_evaluation(
        med_evo_dir=Path(args.icu_evo_dir),
        gt_dir=Path(args.gt_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None,
    )


if __name__ == "__main__":
    main()
