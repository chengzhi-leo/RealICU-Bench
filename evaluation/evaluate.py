"""Score standard predictions: python -m evaluation.evaluate --help."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from config.config import Config, load_protocol
from dataset.labels import action_targets, load_labels, validate_labels_dir, window_interval
from evaluation import active_risks, patient_status, recommendation
from evaluation.active_risks import risk_texts
from evaluation.predictions import read_predictions
from utils.artifacts import prepare_run, read_json, write_json


PREDICTION_TASKS = {"patient_status": "patient_status", "active_risks": "patient_status", "recommendation": "recommendation"}
MATCHED_TASKS = {"active_risks", "recommendation"}
SCORERS = {"patient_status": patient_status, "active_risks": active_risks, "recommendation": recommendation}


def label_index(labels_dir: Path) -> tuple[dict, str]:
    index = {}
    digest = hashlib.sha256()
    for path in sorted(validate_labels_dir(labels_dir).glob("*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        payload = load_labels(path)
        for window in payload["window_outputs"]:
            sample_id = window["sample_id"]
            if sample_id in index:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            available = window["label_availability"]
            actions, flags = action_targets(payload, window)
            risks = risk_texts(window["reference_labels"]["patient_assessment"]["active_risks"]) if available["active_risks"] else []
            index[sample_id] = {
                "stay_id": payload["stay_id"], "window": window, "interval": window_interval(payload, window),
                "actions": actions, "red_flags": flags, "risks": risks,
                "scored": {"patient_status": available["patient_status"], "active_risks": bool(risks), "recommendation": bool(actions)},
            }
    return index, digest.hexdigest()


def load_matches(path: Path) -> dict:
    cached = {}
    if path.exists():
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item["sample_id"] in cached:
                raise ValueError(f"Duplicate cached match in {path}")
            cached[item["sample_id"]] = item["result"]
    return cached


def run(
    config: Config, predictions: str | Path, output: str | Path, *, matcher_factory=None,
    reuse_matches: str | Path | None = None,
) -> dict:
    task = config.require("task", str)
    if task not in PREDICTION_TASKS:
        raise ValueError(f"Unsupported scored task: {task}")
    protocol = load_protocol(config.require("protocol", str))
    labels_dir = Path(config.require("labels_dir", str)).resolve()
    predictions = Path(predictions).resolve()
    index, labels_hash = label_index(labels_dir)
    records = read_predictions(predictions)
    unknown = {sample for sample, _ in records if sample not in index}
    if unknown:
        raise ValueError(f"Predictions do not belong to this annotation set: {sorted(unknown)[:5]}")
    matcher = None
    workers = config.positive_int("execution.num_workers")
    top_k = config.positive_int("evaluation.top_k_actions") if task == "recommendation" else None
    if task in MATCHED_TASKS:
        if matcher_factory is None:
            from evaluation.matching import Matcher

            matcher_factory = Matcher
        matcher = matcher_factory(config.require("matcher", dict))
    sidecar = predictions.parent / "run_config.json"
    if sidecar.exists():
        source = read_json(sidecar)
        if source["workflow"] == "prediction" and (
            source["protocol"] != protocol or source["settings"]["task"] != PREDICTION_TASKS[task]
        ):
            raise ValueError("Prediction and evaluation task or protocol differ")
    settings = {
        "workflow": "evaluation", "protocol": protocol, "settings": config.snapshot(),
        "predictions": str(predictions), "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "labels_dir": str(labels_dir), "labels_sha256": labels_hash,
    }
    cached = {}
    if reuse_matches is not None:
        if task not in MATCHED_TASKS:
            raise ValueError("--reuse-matches is available for active-risk and recommendation evaluation")
        reuse_matches = Path(reuse_matches).resolve()
        previous = read_json(reuse_matches / "run_config.json")
        for key in ("protocol", "predictions_sha256", "labels_sha256"):
            if previous[key] != settings[key]:
                raise ValueError(f"Matching cache {key} differs")
        if previous["settings"]["task"] != task or previous["settings"]["matcher"] != config.require("matcher", dict):
            raise ValueError("Matching cache uses a different task or matcher settings")
        cached = load_matches(reuse_matches / "matches.jsonl")
        settings["matching_cache"] = str(reuse_matches)
    output = prepare_run(output, settings)
    matches_path = output / "matches.jsonl"
    existing = load_matches(matches_path)
    persisted = set(existing)
    for sample_id, result in existing.items():
        if sample_id in cached and cached[sample_id] != result:
            raise ValueError("Conflicting cached matches")
        cached[sample_id] = result
    selected = []
    unavailable = 0
    for (sample_id, record_task), record in sorted(records.items()):
        if record_task != PREDICTION_TASKS[task]:
            continue
        if index[sample_id]["scored"][task]:
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
            return {**row, **patient_status.score(record, sample["window"])}, None
        result = cached.get(sample_id)
        if task == "active_risks":
            predicted = [] if record["error"] else risk_texts(record["prediction"]["patient_assessment"]["active_risks"])
            if result is None:
                result = matcher.match_risks(predicted, sample["risks"])
            return {**row, **active_risks.score(predicted, sample["risks"], result["pairs"])}, {"sample_id": sample_id, "result": result}
        predicted = [] if record["error"] else record["prediction"]["recommended_actions"]
        if result is None:
            result = matcher.match_actions(predicted, sample["actions"], sample["red_flags"])
        scores = recommendation.score(predicted, sample["actions"], sample["red_flags"], result, top_k)
        return {**row, **scores}, {"sample_id": sample_id, "result": result}

    rows = []
    match_file = matches_path.open("a", encoding="utf-8") if task in MATCHED_TASKS else nullcontext(None)
    with match_file as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        for scored, matched in pool.map(score_record, selected):
            rows.append(scored)
            if matched is not None and matched["sample_id"] not in persisted:
                handle.write(json.dumps(matched, allow_nan=False) + "\n")
                handle.flush()
                persisted.add(matched["sample_id"])
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "details.csv", index=False)
    available_total = sum(sample["scored"][task] for sample in index.values())
    metrics = {
        "task": task, "protocol": protocol["version"], "summary": SCORERS[task].summarize(frame),
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
    parser.add_argument("--reuse-matches", type=Path, help="Reuse a verified active-risk or recommendation matching cache")
    args = parser.parse_args()
    print(json.dumps(run(Config(args.config), args.predictions, args.output, reuse_matches=args.reuse_matches), indent=2))


if __name__ == "__main__":
    main()
