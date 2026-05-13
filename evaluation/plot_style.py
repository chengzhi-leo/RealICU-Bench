"""Shared plotting helpers for the ICU-Evo evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PRIMARY_BLUE = "#2563EB"
FOREST_TEAL = "#0F766E"
AMBER = "#D97706"
INDIGO = "#4F46E5"
ROSE = "#C2416C"
SLATE = "#64748B"
LIGHT_SLATE = "#94A3B8"

SYSTEM_PALETTE: Dict[str, str] = {
    "ICU-Evo": PRIMARY_BLUE,
    "Baseline": LIGHT_SLATE,
}

METRIC_PALETTE: Dict[str, str] = {
    "hit": PRIMARY_BLUE,
    "precision": AMBER,
    "recall": FOREST_TEAL,
    "accuracy": INDIGO,
    "macro_f1": ROSE,
    "hrr": SLATE,
}

MODE_PALETTE: Sequence[str] = (
    LIGHT_SLATE,
    PRIMARY_BLUE,
    FOREST_TEAL,
    AMBER,
    INDIGO,
)

MODE_COLOR_OVERRIDES: Dict[str, str] = {
    "full_history_events": LIGHT_SLATE,
    "memory": PRIMARY_BLUE,
    "local_events_only": AMBER,
    "rag_history_events": FOREST_TEAL,
}

METRIC_BAND_ALPHA = 0.10
PATIENT_TRACE_ALPHA = 0.14
MODE_TRACE_ALPHA = 0.92

_STYLE_CONFIGURED = False


def configure_plot_style() -> None:
    global _STYLE_CONFIGURED
    if _STYLE_CONFIGURED:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "font.size": 14.5,
            "axes.labelsize": 17,
            "axes.titlesize": 19,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "grid.color": "#8A8E99",
            "legend.frameon": False,
            "legend.fontsize": 14.5,
            "legend.title_fontsize": 14.5,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 21,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    _STYLE_CONFIGURED = True


def system_color(system_name: str) -> str:
    return SYSTEM_PALETTE.get(str(system_name), "#46505A")


def metric_color(metric: str) -> str:
    return METRIC_PALETTE[str(metric)]


def mode_color_map(labels: Sequence[str]) -> Dict[str, str]:
    ordered = [str(label) for label in labels]

    def _override_color(label: str) -> Optional[str]:
        lowered = str(label).strip().lower()
        for mode_name, color in MODE_COLOR_OVERRIDES.items():
            canonical = str(mode_name).strip().lower()
            if canonical in lowered or canonical.replace("_", " ") in lowered:
                return color
        return None

    palette = [color for color in MODE_PALETTE if color not in MODE_COLOR_OVERRIDES.values()]
    size = len(palette)
    result: Dict[str, str] = {}
    next_index = 0
    for label in ordered:
        override = _override_color(label)
        if override is not None:
            result[label] = override
            continue
        result[label] = palette[next_index % size]
        next_index += 1
    return result


def heatmap_cmap(metric: str) -> LinearSegmentedColormap:
    metric_name = str(metric)
    if metric_name == "precision":
        return LinearSegmentedColormap.from_list(
            "icu_evo_precision",
            ["#FFF8EE", "#F6C988", "#E89B2E", AMBER],
        )
    return LinearSegmentedColormap.from_list(
        "icu_evo_primary",
        ["#EFF6FF", "#BFDBFE", "#60A5FA", PRIMARY_BLUE],
    )


def smooth_series(values: Sequence[float], *, window: int = 9) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("smooth_series expects a 1-D array-like input.")
    if len(arr) == 0:
        return arr.copy()
    width = int(window)
    if width <= 1:
        return arr.copy()
    left = width // 2
    right = width - 1 - left
    out = np.empty_like(arr, dtype=float)
    for idx in range(len(arr)):
        start = max(0, idx - left)
        end = min(len(arr), idx + right + 1)
        chunk = arr[start:end]
        valid = chunk[np.isfinite(chunk)]
        out[idx] = float(np.mean(valid)) if len(valid) > 0 else float("nan")
    finite_mask = np.isfinite(arr) & np.isfinite(out)
    out[finite_mask] = 0.15 * arr[finite_mask] + 0.85 * out[finite_mask]
    return out
