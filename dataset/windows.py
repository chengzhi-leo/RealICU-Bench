"""Build deterministic observation windows without model or agent dependencies."""

import math
from pathlib import Path

import pandas as pd

from dataset.labels import load_labels, window_interval
from utils.event_format import format_event_line
from utils.time_format import format_timestamp_minute


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


def pre_icu_history(events: pd.DataFrame, enter: pd.Timestamp, hours: float, discharge_summaries: int) -> dict:
    source_order = events.sort_values("source_event_index", kind="stable")
    recent = source_order.loc[(source_order["time"] >= enter - pd.Timedelta(hours=hours)) & (source_order["time"] < enter)]
    recent = recent.sort_values("time", kind="quicksort").to_dict("records")
    reports = source_order.loc[(source_order["code"] == "NOTE_DISCHARGESUMMARY") & (source_order["time"] < enter)]
    reports = reports.sort_values("time", ascending=False, kind="stable").head(discharge_summaries).to_dict("records")
    lines = []
    if recent:
        lines.append(f"Pre-ICU history events ({len(recent)} events in last {float(hours):.1f}h):")
        lines.extend(format_event_line({**clean_event(event), "event_id": int(event["source_event_index"])}) for event in recent)
    if reports:
        if lines:
            lines.append("")
        lines.append("Pre-ICU historical reports:")
        for number, report in enumerate(reports, 1):
            before = (enter - report["time"]).total_seconds() / 3600
            text = str(report["text_value"]).strip() if pd.notna(report["text_value"]) else ""
            lines.extend([
                f"--- Report {number}: Discharge Summary (timestamp: {format_timestamp_minute(report['time'])}; "
                f"{before / 24:.1f} days / {before:.1f} hours before ICU admission) ---",
                text or "(No report text)",
                "",
            ])
    content = "\n".join(lines).strip()
    return {
        "source": "pre_icu_history" if content else "none",
        "items": len(recent) + len(reports),
        "history_hours": float(hours),
        "historical_discharge_summary_items": len(reports),
        "content": content,
    }


def create_time_windows(
    labels_dir: Path,
    trajectory: dict,
    *,
    current_window_hours: float,
    include_pre_icu_data: bool,
    use_first_n_hours_after_icu: float | None,
    pre_icu_history_hours: float,
    pre_icu_discharge_summaries: int,
    excluded_event_codes: list[str],
) -> list[dict]:
    if not math.isfinite(current_window_hours) or current_window_hours <= 0:
        raise ValueError("current_window_hours must be finite and positive")
    if use_first_n_hours_after_icu is not None and (not math.isfinite(use_first_n_hours_after_icu) or use_first_n_hours_after_icu < 0):
        raise ValueError("observation hours must be finite and nonnegative, or null")
    if not math.isfinite(pre_icu_history_hours) or pre_icu_history_hours < 0:
        raise ValueError("pre_icu_history_hours must be finite and nonnegative")
    if type(pre_icu_discharge_summaries) is not int or pre_icu_discharge_summaries < 0:
        raise ValueError("pre_icu_discharge_summaries must be a nonnegative integer")
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
    history = pre_icu_history(events, enter, pre_icu_history_hours, pre_icu_discharge_summaries) if include_pre_icu_data else {
        "source": "none", "items": 0, "history_hours": float(pre_icu_history_hours),
        "historical_discharge_summary_items": 0, "content": "",
    }
    events = events.loc[~events["code"].isin(excluded_event_codes)]
    times = pd.DatetimeIndex(events["time"])
    cleaned = [clean_event(event) for event in events.to_dict("records")]
    windows = []
    start = enter
    while start < stop:
        end = label_ends.get(start, min(start + step, stop))
        if end > stop:
            break
        left = times.searchsorted(start, side="left")
        right = times.searchsorted(end, side="left")
        if right > left:
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
