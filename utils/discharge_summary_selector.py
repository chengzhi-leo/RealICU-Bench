"""Discharge-summary selection utilities for ICU stays."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def _mode_or_na(series: pd.Series) -> Any:
    mode_values = series.mode(dropna=True)
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    return pd.NA


def _normalize_events(events_df: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "time", "code", "hosp_stay_id", "icu_stay_id"}
    missing = required.difference(set(events_df.columns))
    if missing:
        raise ValueError(f"events_df is missing required columns: {sorted(missing)}")

    events = events_df.copy()
    events["subject_id"] = pd.to_numeric(events["subject_id"], errors="coerce").astype("Int64")
    events["hosp_stay_id"] = pd.to_numeric(events["hosp_stay_id"], errors="coerce").astype("Int64")
    events["icu_stay_id"] = pd.to_numeric(events["icu_stay_id"], errors="coerce").astype("Int64")
    events["time"] = pd.to_datetime(events["time"], errors="coerce")
    return events


def _normalize_icu_stays(icu_stay_df: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "icu_stay_id", "enter_time", "leave_time"}
    missing = required.difference(set(icu_stay_df.columns))
    if missing:
        raise ValueError(f"icu_stay_df is missing required columns: {sorted(missing)}")

    stays = icu_stay_df.copy()
    stays["subject_id"] = pd.to_numeric(stays["subject_id"], errors="coerce").astype("Int64")
    stays["icu_stay_id"] = pd.to_numeric(stays["icu_stay_id"], errors="coerce").astype("Int64")
    stays["enter_time"] = pd.to_datetime(stays["enter_time"], errors="coerce")
    stays["leave_time"] = pd.to_datetime(stays["leave_time"], errors="coerce")
    return stays


def _build_stay_hosp_map(events_df: pd.DataFrame) -> pd.DataFrame:
    bridge = events_df[
        events_df["icu_stay_id"].notna() & events_df["hosp_stay_id"].notna()
    ][["subject_id", "icu_stay_id", "hosp_stay_id"]].copy()
    if len(bridge) == 0:
        return pd.DataFrame(columns=["subject_id", "icu_stay_id", "stay_hosp_stay_id"])

    mapped = (
        bridge.groupby(["subject_id", "icu_stay_id"], as_index=False)["hosp_stay_id"]
        .agg(_mode_or_na)
        .rename(columns={"hosp_stay_id": "stay_hosp_stay_id"})
    )
    return mapped


def select_discharge_summaries_for_icu_stays(
    events_df: pd.DataFrame,
    icu_stay_df: pd.DataFrame,
    *,
    max_days_after_leave: float = 7.0,
) -> pd.DataFrame:
    """
    Select one discharge summary per ICU stay using strict rules.

    Rule 1:
    - Exactly one NOTE_DISCHARGESUMMARY appears inside ICU window [enter_time, leave_time].

    Rule 2:
    - No in-window discharge summary.
    - A NOTE_DISCHARGESUMMARY appears after leave_time but within `max_days_after_leave`.
    - It has same hosp_stay_id as the ICU stay.
    - No new ICU admission occurs between leave_time and that summary time.

    Returns:
        DataFrame with one row per ICU stay including:
        - selected (bool)
        - selection_rule (str or None)
        - selected_note_* fields when selected
    """
    if max_days_after_leave <= 0:
        raise ValueError("max_days_after_leave must be > 0")

    events = _normalize_events(events_df)
    stays = _normalize_icu_stays(icu_stay_df)

    stay_hosp_map = _build_stay_hosp_map(events)
    stays = stays.merge(stay_hosp_map, on=["subject_id", "icu_stay_id"], how="left")

    notes = events[
        (events["code"] == "NOTE_DISCHARGESUMMARY") & events["time"].notna()
    ][["subject_id", "time", "hosp_stay_id", "text_value", "code_specifics"]].copy()
    notes = notes.sort_values(["subject_id", "time"], kind="stable").reset_index(drop=True)

    notes_by_subject: Dict[int, pd.DataFrame] = {
        int(subject_id): group.reset_index(drop=True)
        for subject_id, group in notes.groupby("subject_id", sort=False)
        if pd.notna(subject_id)
    }

    stays_by_subject: Dict[int, pd.DataFrame] = {
        int(subject_id): group.sort_values("enter_time", kind="stable").reset_index(drop=True)
        for subject_id, group in stays.groupby("subject_id", sort=False)
        if pd.notna(subject_id)
    }

    max_delta = pd.Timedelta(days=float(max_days_after_leave))
    rows = []

    for _, stay in stays.iterrows():
        subject_id = stay["subject_id"]
        icu_stay_id = stay["icu_stay_id"]
        enter_time = stay["enter_time"]
        leave_time = stay["leave_time"]
        stay_hosp_id = stay.get("stay_hosp_stay_id")

        base = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "enter_time": enter_time,
            "leave_time": leave_time,
            "stay_hosp_stay_id": stay_hosp_id,
            "in_icu_note_count": 0,
            "selected": False,
            "selection_rule": None,
            "selected_note_time": pd.NaT,
            "selected_note_hosp_stay_id": pd.NA,
            "selected_note_code_specifics": None,
            "selected_note_text_value": None,
            "selected_note_delta_hours_after_leave": pd.NA,
        }

        if pd.isna(subject_id) or pd.isna(enter_time) or pd.isna(leave_time):
            rows.append(base)
            continue

        subject_notes = notes_by_subject.get(int(subject_id))
        if subject_notes is None or len(subject_notes) == 0:
            rows.append(base)
            continue

        in_window = subject_notes[
            (subject_notes["time"] >= enter_time) & (subject_notes["time"] <= leave_time)
        ]
        base["in_icu_note_count"] = int(len(in_window))

        # Rule 1: exactly one summary in ICU window.
        if len(in_window) == 1:
            note = in_window.iloc[0]
            base["selected"] = True
            base["selection_rule"] = "in_icu_exactly_one"
            base["selected_note_time"] = note["time"]
            base["selected_note_hosp_stay_id"] = note["hosp_stay_id"]
            base["selected_note_code_specifics"] = note.get("code_specifics")
            base["selected_note_text_value"] = note.get("text_value")
            base["selected_note_delta_hours_after_leave"] = float(
                (note["time"] - leave_time).total_seconds() / 3600.0
            )
            rows.append(base)
            continue

        # Rule 2 applies only when in-window count is zero.
        if len(in_window) == 0 and pd.notna(stay_hosp_id):
            candidates = subject_notes[
                (subject_notes["hosp_stay_id"] == stay_hosp_id)
                & (subject_notes["time"] > leave_time)
                & (subject_notes["time"] <= leave_time + max_delta)
            ].sort_values("time", kind="stable")

            subject_stays = stays_by_subject[int(subject_id)]
            for _, note in candidates.iterrows():
                note_time = note["time"]
                intervening = subject_stays[
                    (subject_stays["enter_time"] > leave_time)
                    & (subject_stays["enter_time"] <= note_time)
                    & (subject_stays["icu_stay_id"] != icu_stay_id)
                ]
                if len(intervening) > 0:
                    continue

                base["selected"] = True
                base["selection_rule"] = "post_icu_same_hosp_within_7d_no_new_icu"
                base["selected_note_time"] = note_time
                base["selected_note_hosp_stay_id"] = note["hosp_stay_id"]
                base["selected_note_code_specifics"] = note.get("code_specifics")
                base["selected_note_text_value"] = note.get("text_value")
                base["selected_note_delta_hours_after_leave"] = float(
                    (note_time - leave_time).total_seconds() / 3600.0
                )
                break

        rows.append(base)

    result = pd.DataFrame(rows)
    return result


def summarize_discharge_summary_selection(selection_df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize coverage stats from `select_discharge_summaries_for_icu_stays` output."""
    if len(selection_df) == 0:
        return {
            "total_icu_stays": 0,
            "selected_icu_stays": 0,
            "selected_ratio": 0.0,
            "in_icu_exactly_one": 0,
            "post_icu_same_hosp_within_7d_no_new_icu": 0,
        }

    total = int(len(selection_df))
    selected = int(selection_df["selected"].sum())
    in_icu_one = int((selection_df["selection_rule"] == "in_icu_exactly_one").sum())
    post_icu = int((selection_df["selection_rule"] == "post_icu_same_hosp_within_7d_no_new_icu").sum())

    return {
        "total_icu_stays": total,
        "selected_icu_stays": selected,
        "selected_ratio": selected / total if total else 0.0,
        "in_icu_exactly_one": in_icu_one,
        "post_icu_same_hosp_within_7d_no_new_icu": post_icu,
    }
