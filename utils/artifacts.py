"""Read and write portable run artifacts."""

import json
import os
from pathlib import Path


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def prepare_run(output: str | Path, settings: dict) -> Path:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists():
        if read_json(config_path) != settings:
            raise ValueError(f"Run settings changed in {output}; choose a new output directory")
    else:
        if any(output.iterdir()):
            raise ValueError(f"Output directory is nonempty without run_config.json: {output}")
        write_json(config_path, settings)
    return output
