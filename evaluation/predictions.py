"""The public prediction contract shared by model runners and evaluators."""

import json
from pathlib import Path

from dataset.labels import STATUS_LABELS


TASKS = {"patient_status", "recommendation"}


def validate_prediction(task: str, prediction: dict) -> None:
    if task not in TASKS or not isinstance(prediction, dict):
        raise ValueError("Expected a supported task and prediction object")
    if task == "patient_status":
        assessment = prediction["patient_assessment"]
        if assessment["overall"]["label"] not in STATUS_LABELS:
            raise ValueError("Invalid predicted patient-status label")
        if not isinstance(assessment["active_risks"], list):
            raise ValueError("active_risks must be a list")
    else:
        for key in ("recommended_actions", "red_flags"):
            if not isinstance(prediction[key], list) or any(not isinstance(item, dict) for item in prediction[key]):
                raise ValueError(f"{key} must be a list of objects")
        for action in prediction["recommended_actions"]:
            if not isinstance(action["action_name"], str) or not action["action_name"].strip():
                raise ValueError("Recommended actions require a nonempty action_name")


def validate_record(record: dict) -> tuple[str, str]:
    if not isinstance(record, dict):
        raise ValueError("Each prediction line must be an object")
    sample_id, task = record["sample_id"], record["task"]
    if not isinstance(sample_id, str) or len(sample_id.split("/")) != 3 or task not in TASKS:
        raise ValueError("A published sample_id and supported task are required")
    if record["error"] is None:
        validate_prediction(task, record["prediction"])
    elif not isinstance(record["error"], dict) or not record["error"].get("message") or record["prediction"] is not None:
        raise ValueError("Failed predictions require prediction=null and an error object with a message")
    return sample_id, task


def read_predictions(path: str | Path) -> dict[tuple[str, str], dict]:
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                key = validate_record(record)
                if key in records:
                    raise ValueError(f"Duplicate prediction: {key}")
                records[key] = record
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid prediction in {path}:{line_number}: {error}") from error
    return records


def append_prediction(handle, record: dict) -> None:
    validate_record(record)
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
