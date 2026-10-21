"""Recommended-action and harmful-recommendation metrics for the declared RealICU protocol."""

import math

import pandas as pd


def score(predictions: list[dict], actions: list[dict], flags: list[dict], matches: dict, top_k: int) -> dict:
    considered = min(top_k, len(predictions))
    matched = len({pair["prediction_index"] for pair in matches["recommended_pairs"] if pair["prediction_index"] < considered})
    harmful = len({pair["prediction_index"] for pair in matches["red_flag_pairs"] if pair["prediction_index"] < considered})
    return {
        "num_recommendations": considered, "num_references": len(actions), "num_red_flags": len(flags),
        "num_matches": matched, "num_harmful": harmful,
        "hit": float(matched > 0), "recall": matched / len(actions),
    }


def summarize(frame: pd.DataFrame) -> dict:
    flags = frame[frame["num_red_flags"] > 0]
    totals = flags.groupby("stay_id")[["num_harmful", "num_recommendations"]].sum()
    totals = totals[totals["num_recommendations"] > 0]
    hrr = (totals["num_harmful"] / totals["num_recommendations"]).mean()
    by_stay = frame.groupby("stay_id")[["hit", "recall"]].mean()
    return {
        "num_windows": len(frame), "num_stays": len(by_stay), "num_red_flag_windows": len(flags),
        "hit": float(by_stay["hit"].mean()), "recall": float(by_stay["recall"].mean()),
        "hrr": float(hrr) if math.isfinite(hrr) else None,
    }
