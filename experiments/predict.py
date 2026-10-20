"""Run a baseline or ICU-Evo predictor: python -m experiments.predict --help."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config.config import Config
from dataset.labels import load_labels, window_interval
from evaluation.predictions import TASKS, append_prediction, read_predictions, validate_prediction
from experiments.contexts import build_context
from experiments.runtime import CONTEXT_MODES, load_dataset, observation_windows, run_settings, select_with_stride, window_options
from model.llms import LLMClient
from prompts.predictor_prompts import get_patient_status_prediction_prompt, get_recommendation_action_prompt
from utils.artifacts import prepare_run, read_json, write_json
from utils.event_format import format_event_line
from utils.json_parse import parse_json_dict_or_raise


def make_retriever(config: Config, windows: list[dict], encoder):
    from experiments.window_rag import IncrementalWindowRAGRetriever, build_rag_window_record
    records = [
        build_rag_window_record(
            window_index=index, start_hour=window["hours_since_admission"],
            end_hour=window["hours_since_admission"] + window["current_window_hours"],
            event_lines=[format_event_line(event) for event in window["current_events"]],
        ) for index, window in enumerate(windows)
    ]
    return IncrementalWindowRAGRetriever(window_records=records, encoder=encoder, top_k=config.positive_int("rag.top_k_windows"))


def run(config: Config, output: str | Path, *, memory_dir: str | Path | None = None, client_factory=LLMClient) -> dict:
    task, mode = config.require("task", str), config.require("context_mode", str)
    if task not in TASKS or mode not in CONTEXT_MODES:
        raise ValueError("Unsupported task or context_mode")
    if (mode == "memory") != (memory_dir is not None):
        raise ValueError("--memory-dir is required exactly when context_mode=memory")
    options = window_options(config)
    if mode == "local_events_only" and options["include_pre_icu_data"]:
        raise ValueError("local_events_only requires input.include_pre_icu_data=false")
    stride = config.positive_int("prediction.snapshot_stride")
    if task == "recommendation":
        top_k = config.positive_int("prediction.top_k_actions")
        horizon = config.number("prediction.horizon_hours", minimum=0)
        if horizon == 0:
            raise ValueError("prediction.horizon_hours must be positive")
        template = get_recommendation_action_prompt(top_k, horizon)
    else:
        template = get_patient_status_prediction_prompt()
    llm = config.require("llm", dict)
    client_factory(**llm)
    workers = config.positive_int("execution.num_workers")
    logging = config.require("logging.enabled", bool)
    encoder = None
    if mode == "rag_history_events":
        from model.embeddings import TextEncoder

        encoder = TextEncoder(**config.require("rag.encoder", dict))
        config.positive_int("rag.top_k_windows")
    dataset, stay_ids = load_dataset(config)
    settings = run_settings(config, "prediction")
    if memory_dir is not None:
        memory_dir = Path(memory_dir).resolve()
        source = read_json(memory_dir / "run_config.json")
        if source["workflow"] != "memory" or source["protocol"] != settings["protocol"]:
            raise ValueError("Memory source workflow or protocol differs")
        if source["settings"]["input"] != settings["settings"]["input"]:
            raise ValueError("Memory source input configuration differs; use matching inputs")
        for key in ("root", "label_set"):
            if source["settings"]["data"][key] != settings["settings"]["data"][key]:
                raise ValueError(f"Memory source data.{key} differs; use matching data")
        settings["memory_dir"] = str(memory_dir)
    output = prepare_run(output, settings)
    path = output / "predictions.jsonl"
    completed = read_predictions(path) if path.exists() else {}
    codes = settings["protocol"]["actionable_event_codes"]

    def process(stay_id):
        _, windows = observation_windows(dataset, stay_id, options)
        payload = load_labels(dataset.labels_dir / f"{stay_id}.json")
        by_interval = {
            (round(window["hours_since_admission"], 6), round(window["hours_since_admission"] + window["current_window_hours"], 6)): index
            for index, window in enumerate(windows)
        }
        eligible = []
        for label in payload["window_outputs"]:
            available = label["label_availability"]
            supported = (available["patient_status"] or available["active_risks"]) if task == "patient_status" else (
                available["recommended_action"] or available["red_flags"]
            )
            interval = window_interval(payload, label)
            if supported and interval in by_interval:
                eligible.append((label["sample_id"], by_interval[interval]))
        selected = [item for item in select_with_stride(eligible, stride) if (item[0], task) not in completed]
        if not selected:
            return []
        client = client_factory(**llm)
        client.validate_credentials()
        snapshots = {}
        if mode == "memory":
            from agents.artifacts import load_snapshots, snapshots_by_interval

            snapshots = snapshots_by_interval(load_snapshots(memory_dir, stay_id))
        retriever = make_retriever(config, windows, encoder) if mode == "rag_history_events" else None
        records = []
        context_exhausted = False
        for sample_id, index in selected:
            context = build_context(
                mode=mode, task=task, windows=windows, index=index, actionable_codes=codes,
                snapshots=snapshots, retriever=retriever,
            )
            prompt = template.format(context=context) if task == "patient_status" else template.replace("{context}", context)
            record = {"sample_id": sample_id, "task": task, "prediction": None, "error": None}
            response = None
            try:
                if context_exhausted:
                    raise ValueError("Skipped after a previous full-history context length limit")
                if client.estimate_input_tokens(prompt=prompt) > client.max_input_tokens:
                    context_exhausted = mode == "full_history_events"
                    raise ValueError("Input context length exceeds the configured token limit")
                response = client.chat(prompt=prompt, response_format="text")
                prediction = parse_json_dict_or_raise(response["content"])
                if task == "recommendation":
                    prediction["recommended_actions"] = prediction["recommended_actions"][:top_k]
                validate_prediction(task, prediction)
                record["prediction"] = prediction
                record["usage"] = response["usage"]
            except Exception as error:
                message = str(error).lower()
                if mode == "full_history_events" and any(term in message for term in (
                    "context length", "context window", "token limit", "too many tokens", "maximum context",
                )):
                    context_exhausted = True
                record["error"] = {"type": type(error).__name__, "message": str(error)}
            if logging:
                label_key = sample_id.rsplit("/", 1)[1]
                write_json(output / "logs" / stay_id / f"{label_key}.json", {"prompt": prompt, "response": response, "error": record["error"]})
            records.append(record)
        return records

    with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        for stay_id, records in zip(stay_ids, pool.map(process, stay_ids)):
            for record in records:
                append_prediction(handle, record)
                completed[(record["sample_id"], task)] = record
            print(f"Predictions {stay_id}: {len(records)} new samples", flush=True)
    summary = {"predictions": len(completed), "failed": sum(record["error"] is not None for record in completed.values())}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--memory-dir", type=Path)
    args = parser.parse_args()
    print(run(Config(args.config), args.output, memory_dir=args.memory_dir))


if __name__ == "__main__":
    main()
