"""Patient-status metrics independent of model and memory implementations."""

import pandas as pd


def macro_f1(truth: list[str], predictions: list[str]) -> float:
    scores = []
    for label in sorted(set(truth) | set(predictions)):
        tp = sum(a == label and b == label for a, b in zip(truth, predictions))
        fp = sum(a != label and b == label for a, b in zip(truth, predictions))
        fn = sum(a == label and b != label for a, b in zip(truth, predictions))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return sum(scores) / len(scores)


def score(record: dict, window: dict) -> dict:
    truth = window["reference_labels"]["patient_assessment"]["overall"]["label"]
    prediction = "inference_error" if record["error"] else record["prediction"]["patient_assessment"]["overall"]["label"]
    return {"true_label": truth, "pred_label": prediction, "is_correct": int(truth == prediction)}


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "num_windows": len(frame), "num_patients": frame["stay_id"].nunique(),
        "micro_acc": float(frame["is_correct"].mean()),
        "macro_acc": float(frame.groupby("stay_id")["is_correct"].mean().mean()),
        "macro_f1": macro_f1(frame["true_label"].tolist(), frame["pred_label"].tolist()),
    }
