"""Read the published RealICU annotations and their actual observation intervals."""

import json
import math
from pathlib import Path

import pandas as pd


STATUS_LABELS = {"stable", "improving", "deteriorating", "insufficient_data"}
POSITIVE_ACTION_LABELS = {"acceptable", "best_practice"}
TASKS = {"patient_status", "active_risks", "recommended_action", "red_flags"}
INTERVAL_DECIMALS = 6


def validate_labels_dir(path: Path) -> Path:
    path = Path(path)
    if not path.is_dir() or not any(path.glob("*.json")):
        raise FileNotFoundError(f"No RealICU label files found in {path}")
    return path


def interval_key(start_hour: float, end_hour: float) -> tuple[float, float]:
    return round(start_hour, INTERVAL_DECIMALS), round(end_hour, INTERVAL_DECIMALS)


def window_interval(payload: dict, window: dict) -> tuple[float, float]:
    enter = pd.Timestamp(payload["trajectory_metadata"]["enter_time"])
    metadata = window["window_metadata"]
    start = pd.Timestamp(metadata["window_start_time"])
    end = pd.Timestamp(metadata["window_end_time"])
    return interval_key((start - enter).total_seconds() / 3600, (end - enter).total_seconds() / 3600)


def load_labels(path: Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        stay_id = payload["stay_id"]
        if stay_id != path.stem or stay_id != f"{payload['subject_id']}_{payload['icu_stay_id']}":
            raise ValueError("Inconsistent stay identifiers")
        enter = pd.Timestamp(payload["trajectory_metadata"]["enter_time"])
        leave = pd.Timestamp(payload["trajectory_metadata"]["leave_time"])
        windows = payload["window_outputs"]
        if not windows or not enter < leave:
            raise ValueError("Empty windows or invalid stay timestamps")
        intervals = set()
        indices = set()
        samples = set()
        for window in windows:
            metadata = window["window_metadata"]
            start = pd.Timestamp(metadata["window_start_time"])
            end = pd.Timestamp(metadata["window_end_time"])
            interval = window_interval(payload, window)
            if not enter <= start < end <= leave or interval in intervals:
                raise ValueError("Invalid or duplicate observation interval")
            index = window["window_index"]
            sample_id = window["sample_id"]
            if not isinstance(sample_id, str):
                raise ValueError("sample_id must be a string")
            parts = sample_id.split("/")
            if len(parts) != 3 or parts[0] not in {"gold", "scale"} or parts[1] != stay_id or not parts[2]:
                raise ValueError("sample_id does not identify this published stay")
            if type(index) is not int or index < 0 or index in indices or sample_id in samples:
                raise ValueError("Invalid or duplicate window identifiers")
            availability = window["label_availability"]
            if any(type(availability[task]) is not bool for task in TASKS):
                raise ValueError("Task availability must be boolean")
            references = window["reference_labels"]
            assessment = references["patient_assessment"]
            review = references["action_review"]
            if availability["patient_status"] and assessment["overall"]["label"] not in STATUS_LABELS:
                raise ValueError("Invalid available patient-status label")
            for task, section, key in (
                ("active_risks", assessment, "active_risks"),
                ("recommended_action", review, "evaluations"),
                ("red_flags", review, "red_flags"),
            ):
                if availability[task] and not isinstance(section[key], list):
                    raise ValueError("Reference items must be lists")
            intervals.add(interval)
            indices.add(index)
            samples.add(sample_id)
        windows.sort(key=lambda window: window_interval(payload, window))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid RealICU labels in {path}: {error}") from error
    return payload


def window_marker(payload: dict, window: dict) -> dict:
    start, end = window_interval(payload, window)
    return {
        "marker_id": f"{start:.6f}|{end:.6f}",
        "sample_id": window["sample_id"],
        "stay_id": payload["stay_id"],
        "window_index": window["window_index"],
        "interval_start_hour": start,
        "interval_end_hour": end,
        "window_start_time": window["window_metadata"]["window_start_time"],
        "window_end_time": window["window_metadata"]["window_end_time"],
    }


def action_targets(payload: dict, window: dict) -> tuple[list[dict], list[dict]]:
    marker = window_marker(payload, window)
    metadata = {
        "reference_interval_marker_id": marker["marker_id"],
        "reference_interval_start_hour": marker["interval_start_hour"],
        "reference_interval_end_hour": marker["interval_end_hour"],
        "sample_id": marker["sample_id"],
    }
    availability = window["label_availability"]
    review = window["reference_labels"]["action_review"]
    actions = []
    if availability["recommended_action"]:
        for action in review["evaluations"]:
            if action["label"] not in POSITIVE_ACTION_LABELS:
                continue
            name = action["action_name"].strip()
            if not name:
                raise ValueError(f"Empty action name in {marker['sample_id']}")
            rationale = action["rationale"].strip()
            actions.append({
                **metadata,
                "action_name": name,
                "action_description": rationale,
                "gt_action_text": f"{name}. {rationale}" if rationale else name,
                "reference_label": action["label"],
                "reference_action_id": str(action["action_id"]),
            })
    red_flags = []
    if availability["red_flags"]:
        for flag in review["red_flags"]:
            name = flag["contraindicated_action"].strip()
            reason = flag["rationale"].strip()
            red_flags.append({
                **metadata,
                "contraindicated_action": name,
                "reason": reason,
                "gt_action_text": f"{name}. {reason}" if reason else name,
            })
    return actions, red_flags


def collect_action_targets(actions_by_interval: dict, start_hour: float, horizon_hours: float) -> list[dict]:
    if not math.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ValueError("prediction_horizon_hours must be finite and positive")
    end_hour = round(start_hour + horizon_hours, INTERVAL_DECIMALS)
    return [
        action
        for interval, actions in sorted(actions_by_interval.items())
        if start_hour <= interval[0] <= end_hour
        for action in actions
    ]
