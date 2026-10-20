"""Construct ICU-Evo observation windows from the published RealICU trajectories."""

from pathlib import Path

import pandas as pd

from dataset.labels import validate_labels_dir


EVENT_COLUMNS = [
    "time", "code", "code_specifics", "numeric_value", "text_value",
    "source_event_index", "icu_event_index",
]


class RealICUDataset:
    def __init__(self, root: str, label_set: str):
        if label_set not in {"gold", "scale"}:
            raise ValueError("label_set must be 'gold' or 'scale'")
        self.root = Path(root)
        self.labels_dir = validate_labels_dir(self.root / label_set / "labels")
        self._load_stays()

    def _load_stays(self) -> None:
        stays = pd.read_parquet(self.root / "trajectories/icu_stay.parquet")
        expected_ids = stays["subject_id"].astype(str) + "_" + stays["icu_stay_id"].astype(str)
        if stays["stay_id"].duplicated().any() or not stays["stay_id"].equals(expected_ids):
            raise ValueError("ICU stay table has duplicate or inconsistent stay_id values")
        label_ids = {path.stem for path in self.labels_dir.glob("*.json")}
        missing = label_ids - set(stays["stay_id"])
        if missing:
            raise ValueError(f"Labels have no matching trajectory: {sorted(missing)}")
        self.stays = stays.loc[stays["stay_id"].isin(label_ids)].sort_values("stay_id").reset_index(drop=True)

    def load_trajectory(self, stay_id: str) -> dict:
        matches = self.stays.loc[self.stays["stay_id"] == stay_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one published ICU stay for {stay_id}, found {len(matches)}")
        stay = matches.iloc[0]
        events = pd.read_parquet(
            self.root / "trajectories/events.parquet",
            filters=[("stay_id", "==", stay_id)],
            columns=EVENT_COLUMNS,
        ).sort_values(["time", "source_event_index"], kind="stable").reset_index(drop=True)
        if events["time"].isna().any() or events["source_event_index"].isna().any():
            raise ValueError(f"Missing event timestamps or source IDs for {stay_id}")
        gender = events.loc[events["code"] == "META_GENDER", "code_specifics"]
        return {
            "stay_id": stay_id,
            "subject_id": int(stay["subject_id"]),
            "icu_stay_id": int(stay["icu_stay_id"]),
            "enter_time": stay["enter_time"].isoformat(),
            "leave_time": stay["leave_time"].isoformat(),
            "age_at_admission": (stay["enter_time"] - stay["birth_time"]).total_seconds() / (365.25 * 86400),
            "gender": gender.iloc[0] if not gender.empty else None,
            "icu_duration_hours": (stay["leave_time"] - stay["enter_time"]).total_seconds() / 3600,
            "events": events,
        }
