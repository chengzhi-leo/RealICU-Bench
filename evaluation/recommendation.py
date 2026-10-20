"""Recommendation and red-flag metrics for the declared RealICU protocol."""

import math

import pandas as pd


def score(predictions: list[dict], actions: list[dict], flags: list[dict], matches: dict, k: int) -> dict:
    considered = min(k, len(predictions))
    recommended = {pair["prediction_index"] for pair in matches["recommended_pairs"]}
    harmful = {pair["prediction_index"] for pair in matches["red_flag_pairs"]}
    count = sum(index < considered for index in recommended)
    red_count = sum(index < considered for index in harmful)
    return {
        "k": k, "num_recommendations": len(predictions), "num_recommendations_at_k": considered,
        "num_ground_truth_actions": len(actions), "num_ground_truth_red_flags": len(flags),
        "num_matches_at_k": count, "num_red_flag_matches_at_k": red_count,
        "hit_at_k": float(count >= 1) if actions else float("nan"),
        "precision_at_k": (count / considered if considered else 0.) if actions else float("nan"),
        "recall_at_k": count / len(actions) if actions else float("nan"),
        "red_flag_hit_at_k": float(red_count > 0) if flags else float("nan"),
        "red_flag_precision_at_k": (red_count / considered if considered else 0.) if flags else float("nan"),
    }


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def summarize(frame: pd.DataFrame) -> list[dict]:
    results = []
    for k, subset in frame.groupby("k"):
        flags = subset[subset["num_ground_truth_red_flags"] > 0]
        totals = flags.groupby("stay_id")[["num_red_flag_matches_at_k", "num_recommendations_at_k"]].sum()
        totals = totals[totals["num_recommendations_at_k"] > 0]
        ratios = totals["num_red_flag_matches_at_k"] / totals["num_recommendations_at_k"]
        row = {
            "k": int(k), "num_windows": len(subset), "num_patients": subset["stay_id"].nunique(),
            "num_windows_with_positive_actions": int((subset["num_ground_truth_actions"] > 0).sum()),
            "num_windows_with_red_flags": len(flags),
            "num_patients_with_red_flags": flags["stay_id"].nunique(),
            "harmful_recommendation_rate_case_level_at_k": finite(ratios.mean()),
            "total_matches_at_k": int(subset["num_matches_at_k"].sum()),
            "total_recommendations_at_k": int(subset["num_recommendations_at_k"].sum()),
            "total_red_flag_matches_at_k": int(flags["num_red_flag_matches_at_k"].sum()),
            "total_red_flag_recommendations_at_k": int(flags["num_recommendations_at_k"].sum()),
        }
        for metric in ("hit_at_k", "precision_at_k", "recall_at_k"):
            row[f"micro_{metric}"] = finite(subset[metric].mean())
            row[f"macro_{metric}"] = finite(subset.groupby("stay_id")[metric].mean().mean())
        for metric in ("red_flag_hit_at_k", "red_flag_precision_at_k"):
            row[metric.replace("red_flag_", "red_flag_macro_")] = finite(flags.groupby("stay_id")[metric].mean().mean())
        results.append(row)
    return results
