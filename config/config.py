"""Explicit, per-run configuration without process-wide state."""

import copy
import json
import math
from pathlib import Path


class Config:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        with self.path.open(encoding="utf-8") as handle:
            self.values = json.load(handle)
        if not isinstance(self.values, dict):
            raise ValueError(f"Expected a configuration object in {self.path}")

    def require(self, key: str, kind=None, *, allow_none: bool = False):
        value = self.values
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"Missing required configuration: {key}")
            value = value[part]
        if value is None and allow_none:
            return None
        if value is None or (kind is not None and not isinstance(value, kind)):
            raise ValueError(f"Invalid configuration value for {key}: {value!r}")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Configuration value must be non-empty: {key}")
        return value

    def number(self, key: str, *, minimum: float, allow_none: bool = False):
        value = self.require(key, (int, float), allow_none=allow_none)
        if value is None:
            return None
        if isinstance(value, bool) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"{key} must be finite and >= {minimum}")
        return value

    def positive_int(self, key: str) -> int:
        value = self.require(key, int)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        return value

    def snapshot(self) -> dict:
        return copy.deepcopy(self.values)


def load_protocol(version: str) -> dict:
    if version != "realicu-v1":
        raise ValueError(f"Unsupported protocol: {version}")
    path = Path(__file__).parent / "protocols" / f"{version}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
