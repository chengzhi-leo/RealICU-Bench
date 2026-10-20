"""Persist ICU-Evo memory independently of prediction and evaluation artifacts."""

from pathlib import Path

from utils.artifacts import read_json


def load_snapshots(memory_dir: str | Path, stay_id: str) -> list[dict]:
    payload = read_json(Path(memory_dir) / "memory" / f"{stay_id}.json")
    if payload["stay_id"] != stay_id or not payload["snapshots"]:
        raise ValueError(f"Invalid memory artifact for {stay_id}")
    return payload["snapshots"]


def snapshots_by_interval(snapshots: list[dict]) -> dict[tuple[float, float], dict]:
    result = {}
    for snapshot in snapshots:
        start, end = snapshot["last_processed_start_hour"], snapshot["last_processed_end_hour"]
        key = round(start, 6), round(end, 6)
        if end <= start or key in result:
            raise ValueError(f"Invalid or duplicate memory interval: {key}")
        result[key] = snapshot
    return result


def compact_snapshot(snapshot: dict) -> dict:
    entries = snapshot["trend_memory"]
    if not entries:
        return snapshot
    stats_by_vital = {}
    for entry in entries:
        for name, stats in entry["vital_trends"].items():
            if stats["count"] <= 0:
                continue
            bucket = stats_by_vital.setdefault(name, {"count": 0, "sum": 0., "min": float(stats["min"]), "max": float(stats["max"])})
            bucket["count"] += stats["count"]
            bucket["sum"] += stats["mean"] * stats["count"]
            bucket["min"] = min(bucket["min"], float(stats["min"]))
            bucket["max"] = max(bucket["max"], float(stats["max"]))
    current = {**entries[-1], "trend_scope": "current_window"}
    overall = {
        "trend_scope": "global", "window_index": current["window_index"],
        "start_window": entries[0]["window_index"], "end_window": current["window_index"],
        "start_hour": entries[0]["start_hour"], "end_hour": current["end_hour"],
        "raw_event_count": sum(entry["raw_event_count"] for entry in entries),
        "vital_trends": {
            name: {"count": stats["count"], "mean": stats["sum"] / stats["count"], "min": stats["min"], "max": stats["max"]}
            for name, stats in stats_by_vital.items()
        },
    }
    return {**snapshot, "trend_memory": [current, overall]}
