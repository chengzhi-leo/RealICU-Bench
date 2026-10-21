"""Acute-problem metrics over predicted and reference active risks."""

import pandas as pd


TEXT_KEYS = ("problem", "problem_name", "name", "diagnosis", "risk_name", "label", "title", "description", "text")


def risk_text(item) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def risk_texts(items: list) -> list[str]:
    texts = {}
    for item in items:
        text = risk_text(item)
        if text:
            texts.setdefault(text.lower(), text)
    return list(texts.values())


def score(predictions: list[str], references: list[str], pairs: list[dict]) -> dict:
    matched_predictions = {pair["prediction_index"] for pair in pairs}
    matched_references = {pair["gt_index"] for pair in pairs if pair["gt_index"] >= 0}
    return {
        "num_predictions": len(predictions), "num_references": len(references),
        "num_matches": len(matched_predictions), "num_matched_references": len(matched_references),
        "hit": float(bool(matched_predictions)), "recall": len(matched_references) / len(references),
    }


def summarize(frame: pd.DataFrame) -> dict:
    by_stay = frame.groupby("stay_id")[["hit", "recall"]].mean()
    return {
        "num_windows": len(frame), "num_stays": len(by_stay),
        "hit": float(by_stay["hit"].mean()), "recall": float(by_stay["recall"].mean()),
    }
