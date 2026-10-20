"""Build deterministic observation windows without model or agent dependencies."""

import math
from pathlib import Path

import pandas as pd

from dataset.labels import load_labels, window_interval
from utils.event_format import format_event_line


def clean_event(event: dict) -> dict:
    cleaned = {"time": pd.Timestamp(event["time"]).isoformat(), "code": event["code"]}
    for key in ("code_specifics", "numeric_value", "text_value", "source_event_index", "icu_event_index"):
        value = event[key]
        if pd.notna(value):
            cleaned[key] = int(value) if key.endswith("index") else value
    if "icu_event_index" in cleaned:
        cleaned["event_id"] = cleaned["icu_event_index"]
        cleaned["event_index"] = cleaned["icu_event_index"]
    return cleaned


def create_time_windows(
    labels_dir: Path,
    trajectory: dict,
    *,
    current_window_hours: float,
    include_pre_icu_data: bool,
    use_first_n_hours_after_icu: float | None,
    pre_icu_history_hours: float | None,
    excluded_event_codes: list[str],
) -> list[dict]:
    if not math.isfinite(current_window_hours) or current_window_hours <= 0:
        raise ValueError("current_window_hours must be finite and positive")
    for hours in (use_first_n_hours_after_icu, pre_icu_history_hours):
        if hours is not None and (not math.isfinite(hours) or hours < 0):
            raise ValueError("History and observation hours must be finite and nonnegative, or null")
    payload = load_labels(labels_dir / f"{trajectory['stay_id']}.json")
    enter = pd.Timestamp(trajectory["enter_time"])
    leave = pd.Timestamp(trajectory["leave_time"])
    metadata = payload["trajectory_metadata"]
    if enter != pd.Timestamp(metadata["enter_time"]) or leave != pd.Timestamp(metadata["leave_time"]):
        raise ValueError(f"Trajectory and annotation timestamps differ for {trajectory['stay_id']}")
    label_ends = {}
    step = pd.Timedelta(hours=current_window_hours)
    for window in payload["window_outputs"]:
        start_hour, _ = window_interval(payload, window)
        position = start_hour / current_window_hours
        if not math.isclose(position, round(position), abs_tol=1e-5):
            raise ValueError("Configured current_window_hours does not align with published windows")
        start = pd.Timestamp(window["window_metadata"]["window_start_time"])
        end = pd.Timestamp(window["window_metadata"]["window_end_time"])
        if end - start > step or (end - start < step and window is not payload["window_outputs"][-1]):
            raise ValueError("Configured current_window_hours does not match published intervals")
        label_ends[start] = end
    stop = max(label_ends.values())
    if use_first_n_hours_after_icu is not None:
        stop = min(stop, enter + pd.Timedelta(hours=use_first_n_hours_after_icu))
    events = trajectory["events"]
    events = events.loc[~events["code"].isin(excluded_event_codes)]
    times = pd.DatetimeIndex(events["time"])
    cleaned = [clean_event(event) for event in events.to_dict("records")]
    history_left = 0 if pre_icu_history_hours is None else times.searchsorted(
        enter - pd.Timedelta(hours=pre_icu_history_hours), side="left"
    )
    history_right = times.searchsorted(enter, side="left")
    pre_events = cleaned[history_left:history_right] if include_pre_icu_data else []
    history = {
        "source": "pre_icu_history" if pre_events else "none",
        "items": len(pre_events),
        "history_hours": pre_icu_history_hours,
        "historical_discharge_summary_items": sum(event["code"] == "NOTE_DISCHARGESUMMARY" for event in pre_events),
        "content": "\n".join(format_event_line(event) for event in pre_events),
    }
    windows = []
    start = enter
    while start < stop:
        end = label_ends.get(start, min(start + step, stop))
        if end > stop:
            break
        left = times.searchsorted(start, side="left")
        right = times.searchsorted(end, side="left")
        windows.append({
            "stay_id": trajectory["stay_id"],
            "subject_id": trajectory["subject_id"],
            "icu_stay_id": trajectory["icu_stay_id"],
            "current_window_start": start.isoformat(),
            "current_window_end": end.isoformat(),
            "hours_since_admission": (start - enter).total_seconds() / 3600,
            "current_window_hours": (end - start).total_seconds() / 3600,
            "current_events": cleaned[left:right],
            "num_current_events": int(right - left),
            "pre_icu_history": history,
            "pre_icu_history_source": history["source"],
            "pre_icu_history_items": history["items"],
        })
        start += step
    return windows
