"""Build reusable ICU-Evo memory: python -m experiments.create_memory --help."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config.config import Config
from experiments.runtime import load_dataset, observation_windows, run_settings, window_options
from model.llms import LLMClient
from utils.artifacts import prepare_run, write_json


def run(config: Config, output: str | Path, *, client_factory=LLMClient) -> dict:
    from agents.icu_evo_agent import ICUEvoAgent
    from agents.artifacts import load_snapshots

    options = window_options(config)
    dataset, stay_ids = load_dataset(config)
    settings = run_settings(config, "memory")
    llm = config.require("llm", dict)
    client_factory(**llm)
    agent_options = dict(config.require("icu_evo", dict))
    agent_options["observation_config_path"] = str(Path(agent_options["observation_config_path"]).resolve())
    workers = config.positive_int("execution.num_workers")
    logging = config.require("logging.enabled", bool)
    output = prepare_run(output, settings)

    def process(stay_id):
        path = output / "memory" / f"{stay_id}.json"
        if path.exists():
            return {"stay_id": stay_id, "num_windows": len(load_snapshots(output, stay_id))}
        trajectory, windows = observation_windows(dataset, stay_id, options)
        client = client_factory(**llm)
        client.validate_credentials()
        agent = ICUEvoAgent(llm_client=client, enable_logging=logging, **agent_options)
        metadata = {
            "age": trajectory["age_at_admission"],
            "gender": trajectory["gender"],
            "subject_id": trajectory["subject_id"],
            "icu_stay_id": trajectory["icu_stay_id"],
        }
        try:
            _, snapshots = agent.create_memory_snapshots(windows, metadata, verbose=False)
        finally:
            if logging:
                write_json(output / "logs" / f"{stay_id}.json", {"calls": agent.get_logs()})
        write_json(path, {"stay_id": stay_id, "snapshots": snapshots})
        print(f"Memory {stay_id}: {len(windows)} windows", flush=True)
        return {"stay_id": stay_id, "num_windows": len(windows)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(process, stay_ids))
    summary = {"stays": len(records), "windows": sum(record["num_windows"] for record in records)}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(Config(args.config), args.output))


if __name__ == "__main__":
    main()
