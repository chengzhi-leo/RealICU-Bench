"""Score standard predictions: python -m evaluation.evaluate --help."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from config.config import Config, load_protocol
from dataset.labels import action_targets, collect_action_targets, load_labels, validate_labels_dir, window_interval
from evaluation import patient_status, recommendation
from evaluation.predictions import TASKS, read_predictions
from utils.artifacts import prepare_run, read_json, write_json


def label_index(labels_dir: Path) -> tuple[dict, dict, str]:
    index, actions = {}, {}
    digest = hashlib.sha256()
    for path in sorted(validate_labels_dir(labels_dir).glob("*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        payload = load_labels(path)
        actions[payload["stay_id"]] = {}
        for window in payload["window_outputs"]:
            sample_id = window["sample_id"]
            if sample_id in index:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            interval = window_interval(payload, window)
            positive, flags = action_targets(payload, window)
            index[sample_id] = {
                "stay_id": payload["stay_id"], "window": window, "interval": interval, "red_flags": flags,
            }
            if window["label_availability"]["recommended_action"]:
                actions[payload["stay_id"]][interval] = positive
    return index, actions, digest.hexdigest()


def load_matches(path: Path) -> dict:
    cached = {}
    if path.exists():
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item["sample_id"] in cached:
                raise ValueError(f"Duplicate cached action match in {path}")
            cached[item["sample_id"]] = item["result"]
    return cached


def run(
    config: Config, predictions: str | Path, output: str | Path, *, matcher_factory=None,
    reuse_matches: str | Path | None = None,
) -> dict:
    task = config.require("task", str)
    if task not in TASKS:
        raise ValueError(f"Unsupported scored task: {task}")
    protocol = load_protocol(config.require("protocol", str))
    labels_dir = Path(config.require("labels_dir", str)).resolve()
    predictions = Path(predictions).resolve()
    index, actions_by_stay, labels_hash = label_index(labels_dir)
    records = read_predictions(predictions)
    unknown = {sample for sample, _ in records if sample not in index}
    if unknown:
        raise ValueError(f"Predictions do not belong to this annotation set: {sorted(unknown)[:5]}")
    horizon, k_values, matcher = None, [], None
    workers = config.positive_int("execution.num_workers")
    if task == "recommendation":
        horizon = config.number("evaluation.horizon_hours", minimum=0)
        k_values = config.require("evaluation.k_values", list)
        if horizon == 0 or not k_values or len(set(k_values)) != len(k_values) or any(type(k) is not int or k < 1 for k in k_values):
            raise ValueError("A positive horizon and distinct positive evaluation.k_values are required")
        if matcher_factory is None:
            from evaluation.matching import ActionMatcher

            matcher_factory = ActionMatcher
        matcher = matcher_factory(config.require("matcher", dict))
    sidecar = predictions.parent / "run_config.json"
    if sidecar.exists():
        source = read_json(sidecar)
        if source["workflow"] == "prediction":
            if source["protocol"] != protocol or source["settings"]["task"] != task:
                raise ValueError("Prediction and evaluation task or protocol differ")
            if task == "recommendation" and source["settings"]["prediction"]["horizon_hours"] != horizon:
                raise ValueError("Prediction and evaluation horizons differ")
    settings = {
        "workflow": "evaluation", "protocol": protocol, "settings": config.snapshot(),
        "predictions": str(predictions), "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "labels_dir": str(labels_dir), "labels_sha256": labels_hash,
    }
    cached = {}
    if reuse_matches is not None:
        if task != "recommendation":
            raise ValueError("--reuse-matches is available for recommendation evaluation")
        reuse_matches = Path(reuse_matches).resolve()
        previous = read_json(reuse_matches / "run_config.json")
        for key in ("protocol", "predictions_sha256", "labels_sha256"):
            if previous[key] != settings[key]:
                raise ValueError(f"Matching cache {key} differs")
        if previous["settings"]["matcher"] != config.require("matcher", dict):
            raise ValueError("Matching cache uses different matcher settings")
        if previous["settings"]["evaluation"]["horizon_hours"] != horizon:
            raise ValueError("Matching cache uses a different horizon")
        cached = load_matches(reuse_matches / "matches.jsonl")
        settings["matching_cache"] = str(reuse_matches)
    output = prepare_run(output, settings)
    matches_path = output / "matches.jsonl"
    existing = load_matches(matches_path)
    persisted = set(existing)
    for sample_id, result in existing.items():
        if sample_id in cached and cached[sample_id] != result:
            raise ValueError("Conflicting cached action matches")
        cached[sample_id] = result
    selected = []
    unavailable = 0
    for (sample_id, record_task), record in sorted(records.items()):
        if record_task != task:
            continue
        available = index[sample_id]["window"]["label_availability"]
        usable = available["patient_status"] if task == "patient_status" else (available["recommended_action"] or available["red_flags"])
        if usable:
            selected.append(record)
        else:
            unavailable += 1
    if not selected:
        raise ValueError("No predictions with available references for the requested task")

    def score_record(record):
        sample_id = record["sample_id"]
        sample = index[sample_id]
        row = {
            "sample_id": sample_id, "stay_id": sample["stay_id"], "task": task,
            "hours_since_admission": sample["interval"][0],
            "inference_failed": bool(record["error"]),
        }
        if task == "patient_status":
            return [{**row, **patient_status.score(record, sample["window"])}], None
        available = sample["window"]["label_availability"]
        actions = collect_action_targets(actions_by_stay[sample["stay_id"]], sample["interval"][0], horizon) if available["recommended_action"] else []
        predicted_actions = [] if record["error"] else record["prediction"]["recommended_actions"]
        result = cached.get(sample_id)
        if result is None:
            result = matcher.match(predicted_actions, actions, sample["red_flags"])
        rows = [{**row, **recommendation.score(predicted_actions, actions, sample["red_flags"], result, k)} for k in k_values]
        return rows, {"sample_id": sample_id, "result": result}

    rows = []
    match_file = matches_path.open("a", encoding="utf-8") if task == "recommendation" else nullcontext(None)
    with match_file as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        for details, matched in pool.map(score_record, selected):
            rows.extend(details)
            if matched is not None and matched["sample_id"] not in persisted:
                handle.write(json.dumps(matched, allow_nan=False) + "\n")
                handle.flush()
                persisted.add(matched["sample_id"])
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "details.csv", index=False)
    summary = patient_status.summarize(frame) if task == "patient_status" else recommendation.summarize(frame)
    available_total = sum(
        sample["window"]["label_availability"]["patient_status"] if task == "patient_status" else (
            sample["window"]["label_availability"]["recommended_action"] or sample["window"]["label_availability"]["red_flags"]
        ) for sample in index.values()
    )
    metrics = {
        "task": task, "protocol": protocol["version"], "summary": summary,
        "coverage": {"available_label_windows": available_total, "scored_prediction_windows": len(selected),
                     "unavailable_prediction_windows": unavailable, "missing_prediction_windows": available_total - len(selected),
                     "failed_prediction_windows": sum(record["error"] is not None for record in selected)},
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-matches", type=Path, help="Reuse a verified recommendation evaluation cache")
    args = parser.parse_args()
    print(json.dumps(run(Config(args.config), args.predictions, args.output, reuse_matches=args.reuse_matches), indent=2))


if __name__ == "__main__":
    main()
