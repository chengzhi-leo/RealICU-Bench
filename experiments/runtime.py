"""Shared run setup for trajectory-based experiments."""

from pathlib import Path

from config.config import Config, load_protocol
from dataset.loader import RealICUDataset
from dataset.windows import create_time_windows


CONTEXT_MODES = {"local_events_only", "full_history_events", "rag_history_events", "memory"}


def load_dataset(config: Config) -> tuple[RealICUDataset, list[str]]:
    dataset = RealICUDataset(config.require("data.root", str), config.require("data.label_set", str))
    requested = config.require("data.stay_ids", list, allow_none=True)
    available = set(dataset.stays["stay_id"])
    if requested is None:
        return dataset, sorted(available)
    if not requested or any(not isinstance(stay, str) or stay not in available for stay in requested):
        raise ValueError("data.stay_ids must be null or a nonempty list of published stay IDs")
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate data.stay_ids")
    return dataset, sorted(requested)


def window_options(config: Config) -> dict:
    codes = config.require("input.excluded_event_codes", list)
    if any(not isinstance(code, str) or not code for code in codes):
        raise ValueError("input.excluded_event_codes must contain nonempty strings")
    options = {
        "current_window_hours": config.number("input.current_window_hours", minimum=0),
        "include_pre_icu_data": config.require("input.include_pre_icu_data", bool),
        "use_first_n_hours_after_icu": config.number("input.observation_hours", minimum=0, allow_none=True),
        "pre_icu_history_hours": config.number("input.pre_icu_history_hours", minimum=0),
        "pre_icu_discharge_summaries": config.integer("input.pre_icu_discharge_summaries", minimum=0),
        "excluded_event_codes": codes,
    }
    if options["current_window_hours"] == 0:
        raise ValueError("input.current_window_hours must be positive")
    return options


def observation_windows(dataset: RealICUDataset, stay_id: str, options: dict) -> tuple[dict, list[dict]]:
    trajectory = dataset.load_trajectory(stay_id)
    windows = create_time_windows(dataset.labels_dir, trajectory, **options)
    if not windows:
        raise ValueError(f"No observation windows for {stay_id}; check input.observation_hours")
    return trajectory, windows


def select_with_stride(records: list, stride: int) -> list:
    if type(stride) is not int or stride < 1:
        raise ValueError("snapshot_stride must be a positive integer")
    selected = records[::stride]
    if records and selected[-1] is not records[-1]:
        selected.append(records[-1])
    return selected


def run_settings(config: Config, workflow: str) -> dict:
    protocol = load_protocol(config.require("protocol", str))
    settings = config.snapshot()
    if "data" in settings:
        settings["data"]["root"] = str(Path(settings["data"]["root"]).resolve())
    if "icu_evo" in settings:
        path = settings["icu_evo"]["observation_config_path"]
        settings["icu_evo"]["observation_config_path"] = str(Path(path).resolve())
    return {"workflow": workflow, "protocol": protocol, "settings": settings}
