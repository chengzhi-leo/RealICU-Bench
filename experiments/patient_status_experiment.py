"""Patient-status prediction from precomputed ICU-Evo memory runs and event baselines."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.icu_evo_agent import ICUEvoAgent
from config.config import get_config
from experiments.create_memory import (
    collect_windowed_snapshots,
    extract_snapshot_window_features,
    filter_memory_patient_records_by_stay_ids,
    infer_snapshot_window_index,
    load_memory_patient_records,
    load_memory_run_config,
    load_patient_memory_payload,
    render_snapshot_to_text,
    resolve_memory_run_dir,
)
from experiments.oracle.common import extract_overall_label
from experiments.window_rag import (
    IncrementalWindowRAGRetriever,
    RAGRetrievedWindow,
    SentenceTransformerWindowEncoder,
    build_rag_window_record,
    render_rag_event_context,
)
from prompts.predictor_prompts import get_patient_status_prediction_prompt
from utils.event_format import format_event_line
from utils.json_parse import parse_json_dict_best_effort
from utils.llm_errors import is_context_length_exceeded_error
from utils.llm_log_viewer import save_llm_calls_html
from utils.status_scoring import normalize_status_label

RUN_CONFIG_FILENAME = "run_config.json"
AGGREGATE_FILENAME = "aggregate_results.json"
CONTEXT_MODE_MEMORY = "memory"
CONTEXT_MODE_FULL_HISTORY_EVENTS = "full_history_events"
CONTEXT_MODE_LOCAL_EVENTS_ONLY = "local_events_only"
CONTEXT_MODE_RAG_HISTORY_EVENTS = "rag_history_events"
CONTEXT_MODE_ALL = "all"
SUPPORTED_CONTEXT_MODES = (
    CONTEXT_MODE_MEMORY,
    CONTEXT_MODE_FULL_HISTORY_EVENTS,
    CONTEXT_MODE_LOCAL_EVENTS_ONLY,
    CONTEXT_MODE_RAG_HISTORY_EVENTS,
)
SUPPORTED_CONTEXT_MODE_CHOICES = SUPPORTED_CONTEXT_MODES + (CONTEXT_MODE_ALL,)
ORACLE_INTERVAL_DECIMALS = 6
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
RAG_TOP_K_WINDOWS = 3
RAG_EMBEDDING_BATCH_SIZE = 64
RAG_RETRIEVAL_SCOPE = "past_only"


def _normalize_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return 0
    return 0


def _extract_status_fields(parsed_prediction: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    patient_assessment = parsed_prediction.get("patient_assessment")
    if not isinstance(patient_assessment, dict):
        return "insufficient_data", "", []

    overall = patient_assessment.get("overall")
    if isinstance(overall, dict):
        label = normalize_status_label(overall.get("label"))
        rationale = str(overall.get("rationale") or "").strip()
    else:
        label = "insufficient_data"
        rationale = ""

    active_risks = patient_assessment.get("active_risks")
    if not isinstance(active_risks, list):
        active_risks = []

    if not label:
        label = "insufficient_data"

    return label, rationale, active_risks


def _sum_numeric_stats(items: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _filter_completed_patient_records(
    records: Sequence[Dict[str, Any]],
    results_dir: Path,
) -> Tuple[List[Dict[str, Any]], int]:
    pending_records: List[Dict[str, Any]] = []
    skipped = 0
    for record in records:
        subject_id = int(record["subject_id"])
        icu_stay_id = int(record["icu_stay_id"])
        patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
        if (patient_dir / "prediction.json").exists() and (patient_dir / "patient_status_predictions.json").exists():
            skipped += 1
            continue
        pending_records.append(record)
    return pending_records, skipped


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _interval_key(start_hour: float, end_hour: float) -> Tuple[float, float]:
    return (round(float(start_hour), ORACLE_INTERVAL_DECIMALS), round(float(end_hour), ORACLE_INTERVAL_DECIMALS))


def _interval_marker_id(interval_key: Tuple[float, float]) -> str:
    return f"{interval_key[0]:.{ORACLE_INTERVAL_DECIMALS}f}|{interval_key[1]:.{ORACLE_INTERVAL_DECIMALS}f}"


def _safe_optional_int(value: Any, *, field_name: str, source_path: Path) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}={value!r} in {source_path}") from exc


def _snapshot_interval(snapshot: Mapping[str, Any], *, source_label: str) -> Tuple[float, float]:
    start_raw = snapshot.get("last_processed_start_hour")
    end_raw = snapshot.get("last_processed_end_hour")
    try:
        start_hour = float(start_raw)
        end_hour = float(end_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing/invalid snapshot hour bounds in {source_label}: "
            f"last_processed_start_hour={start_raw!r}, last_processed_end_hour={end_raw!r}"
        ) from exc
    if end_hour <= start_hour:
        raise ValueError(f"Invalid snapshot hour bounds in {source_label}: start={start_hour}, end={end_hour}")
    return start_hour, end_hour


def _oracle_interval_from_row(row: Mapping[str, Any], oracle_prediction_path: Path) -> Tuple[float, float]:
    metadata = row.get("window_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Missing window_metadata in row in {oracle_prediction_path}")
    start_raw = metadata.get("hours_since_admission")
    duration_raw = metadata.get("current_window_hours")
    try:
        start_hour = float(start_raw)
        duration_hours = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid oracle window hours in {oracle_prediction_path}: "
            f"hours_since_admission={start_raw!r}, current_window_hours={duration_raw!r}"
        ) from exc
    if duration_hours <= 0:
        raise ValueError(
            f"Oracle current_window_hours must be > 0 in {oracle_prediction_path}, got {duration_hours!r}"
        )
    return start_hour, start_hour + duration_hours


def _build_oracle_marker(
    *,
    row: Mapping[str, Any],
    interval_key: Tuple[float, float],
    oracle_prediction_path: Path,
) -> Dict[str, Any]:
    metadata = row.get("window_metadata")
    metadata_payload = metadata if isinstance(metadata, Mapping) else {}
    marker = {
        "marker_id": _interval_marker_id(interval_key),
        "interval_start_hour": float(interval_key[0]),
        "interval_end_hour": float(interval_key[1]),
        "window_index": _safe_optional_int(
            row.get("window_index"),
            field_name="window_index",
            source_path=oracle_prediction_path,
        ),
        "source_window_index": _safe_optional_int(
            row.get("source_window_index", metadata_payload.get("source_window_index")),
            field_name="source_window_index",
            source_path=oracle_prediction_path,
        ),
        "stride_source_window_index": _safe_optional_int(
            row.get("stride_source_window_index", metadata_payload.get("stride_source_window_index")),
            field_name="stride_source_window_index",
            source_path=oracle_prediction_path,
        ),
    }
    return marker


def _load_oracle_status_labels_by_interval(
    oracle_prediction_path: Path,
) -> Tuple[Dict[Tuple[float, float], str], Set[Tuple[float, float]], Dict[Tuple[float, float], Dict[str, Any]]]:
    payload = _read_json(oracle_prediction_path)
    window_outputs = payload.get("window_outputs")
    if not isinstance(window_outputs, list):
        raise ValueError(f"Invalid window_outputs list in {oracle_prediction_path}")

    labels_by_interval: Dict[Tuple[float, float], str] = {}
    labeled_intervals: Set[Tuple[float, float]] = set()
    markers_by_interval: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for row in window_outputs:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid row in window_outputs in {oracle_prediction_path}")
        interval_start, interval_end = _oracle_interval_from_row(
            row=row, oracle_prediction_path=oracle_prediction_path
        )
        interval = _interval_key(interval_start, interval_end)
        if interval in markers_by_interval:
            raise ValueError(f"Duplicate oracle interval marker for interval={interval} in {oracle_prediction_path}")
        marker = _build_oracle_marker(row=row, interval_key=interval, oracle_prediction_path=oracle_prediction_path)
        markers_by_interval[interval] = marker

        oracle_output = row.get("oracle_output")
        if not isinstance(oracle_output, Mapping):
            continue
        label = normalize_status_label(extract_overall_label(oracle_output))
        labels_by_interval[interval] = label
        labeled_intervals.add(interval)

    return labels_by_interval, labeled_intervals, markers_by_interval


def _select_records_with_stride(records: List[Dict[str, Any]], stride: int) -> List[Dict[str, Any]]:
    if not records:
        return []
    normalized_stride = int(stride)
    if normalized_stride < 1:
        raise ValueError(f"stride must be >= 1, got {normalized_stride}")
    selected = [record for idx, record in enumerate(records) if idx % normalized_stride == 0]
    if not selected or selected[-1] is not records[-1]:
        selected.append(records[-1])
    return selected


def _extract_current_window(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    working_memory = snapshot.get("working_memory")
    if not isinstance(working_memory, list) or not working_memory:
        raise ValueError("Snapshot missing non-empty working_memory")
    current_window = working_memory[-1]
    if not isinstance(current_window, Mapping):
        raise ValueError("Snapshot current working_memory entry must be a mapping")
    return current_window


def _window_events(window: Mapping[str, Any]) -> List[Any]:
    events = window.get("events")
    if not isinstance(events, list):
        return []
    return list(events)


def _event_line_text(event: Any) -> str:
    if isinstance(event, Mapping):
        return str(format_event_line(dict(event))).strip()
    if isinstance(event, str):
        return str(event).strip()
    return ""


def _event_lines(events: Sequence[Any]) -> List[str]:
    lines = [_event_line_text(event) for event in events]
    return [line for line in lines if line]


def _render_flat_raw_events(events: Sequence[Any]) -> str:
    lines = _event_lines(events)
    if not lines:
        return "(No events)"
    return "\n".join(lines)


def _build_rag_record_from_snapshot(
    *,
    window_index: int,
    start_hour: float,
    end_hour: float,
    snapshot: Mapping[str, Any],
):
    current_window = _extract_current_window(snapshot)
    return build_rag_window_record(
        window_index=int(window_index),
        start_hour=float(start_hour),
        end_hour=float(end_hour),
        event_lines=_event_lines(_window_events(current_window)),
    )


def _retrieved_windows_payload(items: Sequence[RAGRetrievedWindow]) -> List[Dict[str, Any]]:
    return [dict(item.to_payload()) for item in items]


def _build_full_history_event_context(
    *,
    selected_window_index: int,
    selected_snapshot: Mapping[str, Any],
    snapshot_by_window: Mapping[int, Mapping[str, Any]],
) -> str:
    candidate_windows = {int(index) for index in snapshot_by_window.keys() if int(index) <= int(selected_window_index)}
    candidate_windows.add(int(selected_window_index))
    merged_events: List[Any] = []
    for window_index in sorted(candidate_windows):
        if int(window_index) == int(selected_window_index):
            snapshot = selected_snapshot
        else:
            snapshot = snapshot_by_window.get(int(window_index))
            if not isinstance(snapshot, Mapping):
                continue
        merged_events.extend(_window_events(_extract_current_window(snapshot)))
    return _render_flat_raw_events(merged_events)


def _build_local_events_only_context(selected_snapshot: Mapping[str, Any]) -> str:
    current_window = _extract_current_window(selected_snapshot)
    return _render_flat_raw_events(_window_events(current_window))


def _build_status_context(
    *,
    context_mode: str,
    selected_window_index: int,
    selected_snapshot: Mapping[str, Any],
    snapshot_by_window: Mapping[int, Mapping[str, Any]],
) -> Tuple[str, str]:
    if context_mode == CONTEXT_MODE_MEMORY:
        return render_snapshot_to_text(dict(selected_snapshot)), "precomputed_icu_evo_memory"
    if context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS:
        return (
            _build_full_history_event_context(
                selected_window_index=int(selected_window_index),
                selected_snapshot=selected_snapshot,
                snapshot_by_window=snapshot_by_window,
            ),
            CONTEXT_MODE_FULL_HISTORY_EVENTS,
        )
    if context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY:
        return _build_local_events_only_context(selected_snapshot), CONTEXT_MODE_LOCAL_EVENTS_ONLY
    raise ValueError(
        f"Unsupported context_mode={context_mode}. " f"Supported modes: {', '.join(SUPPORTED_CONTEXT_MODES)}"
    )


def _results_dir_name_for_context_mode(context_mode: str) -> str:
    if context_mode == CONTEXT_MODE_MEMORY:
        return "patient_status/memory"
    return f"patient_status/{context_mode}"


def process_single_patient(
    patient_record: Dict[str, Any],
    agent: ICUEvoAgent,
    memory_run_config: Mapping[str, Any],
    snapshot_stride: int,
    context_mode: str,
    oracle_results_dir: Path,
    memory_run_dir: Path,
    results_dir: Path,
    patient_idx: int,
    total_patients: int,
    enable_logging: bool,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    subject_id = int(patient_record["subject_id"])
    icu_stay_id = int(patient_record["icu_stay_id"])
    actual_outcome = str(patient_record.get("actual_outcome") or "unknown")
    source_patient_dir = Path(str(patient_record["patient_dir"]))

    if verbose:
        print(f"\n[Patient {patient_idx}/{total_patients}] Subject: {subject_id}, ICU Stay: {icu_stay_id}")
        print(f"   Source memory: {source_patient_dir}")
        print(f"   Context mode: {context_mode}")

    try:
        payload = load_patient_memory_payload(source_patient_dir, run_config=memory_run_config)
        memory_snapshots = payload.get("memory_snapshots", [])
        if not isinstance(memory_snapshots, list):
            memory_snapshots = []
        final_memory = payload.get("final_memory", {})
        if not isinstance(final_memory, dict):
            final_memory = {}

        windowed_snapshots = collect_windowed_snapshots(memory_snapshots)
        if not windowed_snapshots and final_memory:
            final_window_index = infer_snapshot_window_index(final_memory)
            windowed_snapshots = [(int(final_window_index or -1), final_memory)]

        if not windowed_snapshots:
            print("   WARNING: No snapshots found in source memory, skipping...")
            return None

        oracle_prediction_path = (
            oracle_results_dir / "patients" / f"{subject_id}_{icu_stay_id}" / "oracle_predictions.json"
        )
        (
            oracle_labels_by_interval,
            oracle_labeled_intervals,
            oracle_markers_by_interval,
        ) = _load_oracle_status_labels_by_interval(oracle_prediction_path)
        snapshot_records: List[Dict[str, Any]] = []
        for window_idx, snapshot in windowed_snapshots:
            start_hour, end_hour = _snapshot_interval(
                snapshot,
                source_label=f"{source_patient_dir}/memory_database.json window_index={window_idx}",
            )
            snapshot_records.append(
                {
                    "window_index": int(window_idx),
                    "snapshot": snapshot,
                    "start_hour": float(start_hour),
                    "end_hour": float(end_hour),
                    "interval_key": _interval_key(start_hour, end_hour),
                }
            )
        labeled_snapshot_records = [
            record for record in snapshot_records if record["interval_key"] in oracle_labeled_intervals
        ]
        selected_snapshot_records = _select_records_with_stride(labeled_snapshot_records, snapshot_stride)
        snapshot_by_window = {int(window_idx): snapshot for window_idx, snapshot in windowed_snapshots}
        if not selected_snapshot_records:
            print("   WARNING: No selected snapshots found, skipping...")
            return None

        if verbose:
            print(
                f"   Memory snapshots: total={len(windowed_snapshots)}, "
                f"oracle_labeled={len(labeled_snapshot_records)}, "
                f"selected={len(selected_snapshot_records)}, stride={snapshot_stride}"
            )
            print("   Status inference uses every selected snapshot.")

        status_call_logs: List[Dict[str, Any]] = []
        rag_retriever: Optional[IncrementalWindowRAGRetriever] = None
        if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS:
            rag_window_records = [
                _build_rag_record_from_snapshot(
                    window_index=int(record["window_index"]),
                    start_hour=float(record["start_hour"]),
                    end_hour=float(record["end_hour"]),
                    snapshot=record["snapshot"],
                )
                for record in snapshot_records
            ]
            rag_retriever = IncrementalWindowRAGRetriever(
                window_records=rag_window_records,
                encoder=SentenceTransformerWindowEncoder(
                    model_name=RAG_EMBEDDING_MODEL,
                    device="cuda",
                    batch_size=RAG_EMBEDDING_BATCH_SIZE,
                ),
                top_k=RAG_TOP_K_WINDOWS,
            )
        total_status_input_tokens = 0
        total_status_output_tokens = 0
        status_predictions: List[Dict[str, Any]] = []
        full_history_context_limit_reached = False
        full_history_context_limit_window_index: Optional[int] = None
        full_history_context_limit_message = ""
        for sequence_idx, selected_record in enumerate(selected_snapshot_records, start=1):
            selected_window_index = int(selected_record["window_index"])
            snapshot_for_status = selected_record["snapshot"]
            current_start_hour = float(selected_record["start_hour"])
            current_end_hour = float(selected_record["end_hour"])
            current_interval_key = selected_record["interval_key"]
            current_oracle_marker = oracle_markers_by_interval.get(current_interval_key)
            if not isinstance(current_oracle_marker, Mapping):
                raise ValueError(
                    f"Missing oracle marker for current interval={current_interval_key} "
                    f"for patient={subject_id}_{icu_stay_id}"
                )
            ground_truth_status_label = oracle_labels_by_interval.get(current_interval_key)
            if ground_truth_status_label is None:
                raise ValueError(
                    f"Missing oracle status label for current interval={current_interval_key} "
                    f"for patient={subject_id}_{icu_stay_id}"
                )

            inferred_window_index, _, num_current_events = extract_snapshot_window_features(snapshot_for_status)
            if inferred_window_index >= 0:
                window_index = inferred_window_index
            else:
                window_index = int(selected_window_index)

            skip_due_to_prior_context_limit = (
                context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS and full_history_context_limit_reached
            )
            rag_retrieved_windows: List[RAGRetrievedWindow] = []
            if skip_due_to_prior_context_limit:
                context = ""
                snapshot_source = CONTEXT_MODE_FULL_HISTORY_EVENTS
            elif context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS:
                if rag_retriever is None:
                    raise ValueError("rag_retriever must be initialized for rag_history_events mode")
                current_rag_record = _build_rag_record_from_snapshot(
                    window_index=int(selected_window_index),
                    start_hour=float(current_start_hour),
                    end_hour=float(current_end_hour),
                    snapshot=snapshot_for_status,
                )
                rag_retrieved_windows = rag_retriever.retrieve(
                    current_window_index=int(selected_window_index),
                    query_text=current_rag_record.event_text,
                )
                context = render_rag_event_context(
                    current_window=current_rag_record,
                    retrieved_windows=rag_retrieved_windows,
                    retrieval_top_k=RAG_TOP_K_WINDOWS,
                )
                snapshot_source = CONTEXT_MODE_RAG_HISTORY_EVENTS
            else:
                context, snapshot_source = _build_status_context(
                    context_mode=str(context_mode),
                    selected_window_index=int(selected_window_index),
                    selected_snapshot=snapshot_for_status,
                    snapshot_by_window=snapshot_by_window,
                )
            prompt = ""
            raw_response = ""
            usage: Dict[str, Any] = {}
            prediction_error: Optional[Dict[str, str]] = None
            if skip_due_to_prior_context_limit:
                prediction_error = {
                    "type": "llm_context_length_exceeded",
                    "message": (
                        f"Skipped inference after prior context-length failure at window "
                        f"{full_history_context_limit_window_index}: {full_history_context_limit_message}"
                    ).strip(),
                }
                if verbose:
                    print(
                        f"   WARNING: Window {window_index} skipped after prior full-history token limit at "
                        f"window {full_history_context_limit_window_index}; counted as prediction error."
                    )
            else:
                prompt = get_patient_status_prediction_prompt().format(context=context)
                estimated_input_tokens = agent.llm_client.estimate_input_tokens(prompt=prompt)
                usage = {"input_tokens": int(estimated_input_tokens), "output_tokens": 0}
                if estimated_input_tokens > int(agent.llm_client.max_input_tokens):
                    message = (
                        f"Input prompt token limit exceeded: {estimated_input_tokens} > "
                        f"{agent.llm_client.max_input_tokens} "
                        f"(provider={agent.llm_client.provider}, model={agent.llm_client.model})"
                    )
                    prediction_error = {
                        "type": "llm_context_length_exceeded",
                        "message": message,
                    }
                    if context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS:
                        full_history_context_limit_reached = True
                        full_history_context_limit_window_index = int(window_index)
                        full_history_context_limit_message = message
                    if verbose:
                        print(
                            f"   WARNING: Window {window_index} prompt exceeds token limit; "
                            "counted as prediction error without API call."
                        )
                else:
                    try:
                        response = agent.llm_client.chat(prompt=prompt, response_format="text")
                        raw_response = str(response.get("content", ""))
                        usage_obj = response.get("usage", {})
                        if isinstance(usage_obj, dict):
                            usage = usage_obj
                    except Exception as e:
                        if is_context_length_exceeded_error(e):
                            if context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS:
                                full_history_context_limit_reached = True
                                full_history_context_limit_window_index = int(window_index)
                                full_history_context_limit_message = str(e)
                            prediction_error = {
                                "type": "llm_context_length_exceeded",
                                "message": str(e),
                            }
                            if verbose:
                                print(
                                    f"   WARNING: Window {window_index} hit token/context limit; "
                                    "counted as prediction error."
                                )
                        else:
                            raise

            input_tokens = _normalize_token_count(usage.get("input_tokens"))
            output_tokens = _normalize_token_count(usage.get("output_tokens"))
            total_status_input_tokens += input_tokens
            total_status_output_tokens += output_tokens

            parsed_prediction = parse_json_dict_best_effort(raw_response)
            if parsed_prediction is None:
                parsed_prediction = {}

            if prediction_error is not None:
                status_label = "inference_error"
                status_rationale = str(prediction_error.get("message") or "").strip()
                active_risks = []
            else:
                status_label, status_rationale, active_risks = _extract_status_fields(parsed_prediction)

            status_predictions.append(
                {
                    "window_index": int(window_index),
                    "snapshot_sequence": int(sequence_idx),
                    "hours_since_admission": float(current_start_hour),
                    "window_start_hour": float(current_start_hour),
                    "window_end_hour": float(current_end_hour),
                    "window_duration_hours": float(current_end_hour - current_start_hour),
                    "num_current_events": int(num_current_events),
                    "snapshot_source": str(snapshot_source),
                    "context_mode": str(context_mode),
                    "oracle_window_marker_id": str(current_oracle_marker.get("marker_id") or ""),
                    "oracle_window_marker": dict(current_oracle_marker),
                    "num_rag_retrieved_windows": int(len(rag_retrieved_windows)),
                    "rag_retrieved_window_indices": [int(item.window_index) for item in rag_retrieved_windows],
                    "rag_retrieved_windows": _retrieved_windows_payload(rag_retrieved_windows),
                    "rag_embedding_model": (
                        RAG_EMBEDDING_MODEL if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
                    ),
                    "rag_top_k_windows": (
                        int(RAG_TOP_K_WINDOWS) if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
                    ),
                    "rag_retrieval_scope": (
                        RAG_RETRIEVAL_SCOPE if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
                    ),
                    "ground_truth_status_label": str(ground_truth_status_label),
                    "status_label": status_label,
                    "status_rationale": status_rationale,
                    "active_risks": active_risks,
                    "parsed_prediction": parsed_prediction,
                    "prediction_error": prediction_error,
                }
            )

            status_call_logs.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "patient_id": f"{subject_id}_{icu_stay_id}",
                    "window_index": int(window_index),
                    "hours_since_admission": float(current_start_hour),
                    "prompt": prompt,
                    "response": raw_response,
                    "parsed_response": parsed_prediction,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "metadata": {
                        "step_type": "patient_status_predictor",
                        "llm_provider": agent.llm_client.provider,
                        "llm_model": agent.llm_client.model,
                        "snapshot_source": str(snapshot_source),
                        "context_mode": str(context_mode),
                        "snapshot_sequence": int(sequence_idx),
                        "memory_run": str(memory_run_dir),
                        "num_rag_retrieved_windows": int(len(rag_retrieved_windows)),
                        "oracle_window_marker_id": str(current_oracle_marker.get("marker_id") or ""),
                        "prediction_error": prediction_error,
                    },
                }
            )

            if verbose:
                print(f"   Status Window {window_index}: {status_label}")

        status_distribution: Dict[str, int] = {}
        for item in status_predictions:
            label = item.get("status_label", "insufficient_data")
            status_distribution[label] = status_distribution.get(label, 0) + 1
        total_rag_retrieved_windows = sum(int(item.get("num_rag_retrieved_windows", 0)) for item in status_predictions)
        num_failed_status_predictions = sum(
            1 for item in status_predictions if isinstance(item.get("prediction_error"), dict)
        )

        patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        patient_status_payload = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "memory_run": str(memory_run_dir),
            "source_patient_dir": str(source_patient_dir),
            "context_mode": str(context_mode),
            "oracle_results_dir": str(oracle_results_dir),
            "num_memory_snapshots": len(windowed_snapshots),
            "num_oracle_labeled_snapshots": len(labeled_snapshot_records),
            "num_selected_snapshots_for_prediction": len(selected_snapshot_records),
            "snapshot_stride": snapshot_stride,
            "window_alignment_mode": "oracle_interval_time_match",
            "selection_mode": "oracle_labeled_window_time_match_then_stride",
            "status_distribution": status_distribution,
            "num_failed_status_predictions": int(num_failed_status_predictions),
            "num_rag_retrieved_windows": int(total_rag_retrieved_windows),
            "rag_embedding_model": RAG_EMBEDDING_MODEL if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_top_k_windows": int(RAG_TOP_K_WINDOWS) if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_retrieval_scope": RAG_RETRIEVAL_SCOPE if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "status_predictions": status_predictions,
            "status_prediction_tokens": {
                "input_tokens": total_status_input_tokens,
                "output_tokens": total_status_output_tokens,
                "total_tokens": total_status_input_tokens + total_status_output_tokens,
            },
        }

        with open(patient_dir / "patient_status_predictions.json", "w") as f:
            json.dump(patient_status_payload, f, indent=2)

        summary_payload = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "memory_run": str(memory_run_dir),
            "context_mode": str(context_mode),
            "num_memory_snapshots": len(windowed_snapshots),
            "num_oracle_labeled_snapshots": len(labeled_snapshot_records),
            "num_selected_snapshots_for_prediction": len(selected_snapshot_records),
            "snapshot_stride": snapshot_stride,
            "final_status_label": (
                status_predictions[-1]["status_label"] if status_predictions else "insufficient_data"
            ),
            "status_distribution": status_distribution,
            "num_failed_status_predictions": int(num_failed_status_predictions),
            "num_rag_retrieved_windows": int(total_rag_retrieved_windows),
            "rag_embedding_model": RAG_EMBEDDING_MODEL if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_top_k_windows": int(RAG_TOP_K_WINDOWS) if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_retrieval_scope": RAG_RETRIEVAL_SCOPE if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
        }
        with open(patient_dir / "prediction.json", "w") as f:
            json.dump(summary_payload, f, indent=2)

        if enable_logging:
            patient_logs = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "agent_type": f"patient_status_{context_mode}",
                "llm_provider": getattr(agent.llm_client, "provider", None),
                "llm_model": getattr(agent.llm_client, "model", None),
                "context_mode": str(context_mode),
                "pipeline_agents": [{"name": "patient_status_predictor", "used": True}],
                "total_calls": len(status_call_logs),
                "calls": status_call_logs,
            }
            with open(patient_dir / "llm_calls.json", "w") as f:
                json.dump(patient_logs, f, indent=2)
            save_llm_calls_html(patient_logs, patient_dir / "llm_calls.html")
            if verbose:
                print("   Saved log viewer: llm_calls.html")

        return {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "context_mode": str(context_mode),
            "num_memory_snapshots": len(windowed_snapshots),
            "num_oracle_labeled_snapshots": len(labeled_snapshot_records),
            "num_selected_snapshots_for_prediction": len(selected_snapshot_records),
            "snapshot_stride": snapshot_stride,
            "num_status_predictions": len(status_predictions),
            "status_distribution": status_distribution,
            "num_failed_status_predictions": int(num_failed_status_predictions),
            "num_rag_retrieved_windows": int(total_rag_retrieved_windows),
            "rag_embedding_model": RAG_EMBEDDING_MODEL if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_top_k_windows": int(RAG_TOP_K_WINDOWS) if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "rag_retrieval_scope": RAG_RETRIEVAL_SCOPE if context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None,
            "final_status_label": (
                status_predictions[-1]["status_label"] if status_predictions else "insufficient_data"
            ),
            "status_prediction_tokens": {
                "input_tokens": total_status_input_tokens,
                "output_tokens": total_status_output_tokens,
                "total_tokens": total_status_input_tokens + total_status_output_tokens,
            },
        }

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def _save_run_config(results_dir: Path, payload: Dict[str, Any]) -> None:
    with open(results_dir / RUN_CONFIG_FILENAME, "w") as f:
        json.dump(payload, f, indent=2)


def run_experiment(
    memory_run: str,
    snapshot_stride: int = 1,
    context_mode: str = "",
    oracle_results_dir: str = "",
    verbose: bool = True,
    enable_logging: bool = True,
    patient_stay_ids_path: Optional[str] = None,
    num_workers: int = 2,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    requested_context_mode = str(context_mode).strip()
    if requested_context_mode == CONTEXT_MODE_ALL:
        mode_results: Dict[str, Any] = {}
        for mode in SUPPORTED_CONTEXT_MODES:
            print("\n" + "=" * 80)
            print(f"RUNNING CONTEXT MODE: {mode}")
            print("=" * 80)
            mode_results[mode] = run_experiment(
                memory_run=memory_run,
                snapshot_stride=snapshot_stride,
                context_mode=mode,
                oracle_results_dir=oracle_results_dir,
                verbose=verbose,
                enable_logging=enable_logging,
                patient_stay_ids_path=patient_stay_ids_path,
                num_workers=num_workers,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        return {
            "context_mode": CONTEXT_MODE_ALL,
            "executed_context_modes": list(SUPPORTED_CONTEXT_MODES),
            "mode_results": mode_results,
        }

    try:
        normalized_stride = int(snapshot_stride)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid snapshot_stride value: {snapshot_stride}") from exc
    if normalized_stride < 1:
        raise ValueError(f"snapshot_stride must be >= 1, got {normalized_stride}")
    try:
        normalized_num_workers = int(num_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid num_workers value: {num_workers}") from exc
    if normalized_num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {normalized_num_workers}")
    normalized_context_mode = requested_context_mode
    if normalized_context_mode not in SUPPORTED_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported context_mode={context_mode}. " f"Supported modes: {', '.join(SUPPORTED_CONTEXT_MODES)}"
        )
    normalized_oracle_results_dir = Path(str(oracle_results_dir))
    if not normalized_oracle_results_dir.exists() or not normalized_oracle_results_dir.is_dir():
        raise FileNotFoundError(f"Oracle results directory not found: {normalized_oracle_results_dir}")
    oracle_patients_dir = normalized_oracle_results_dir / "patients"
    if not oracle_patients_dir.exists() or not oracle_patients_dir.is_dir():
        raise FileNotFoundError(f"Oracle patients directory not found: {oracle_patients_dir}")

    config = get_config()
    effective_llm_provider = str(llm_provider if llm_provider is not None else config.llm_provider).strip()
    if not effective_llm_provider:
        raise ValueError("LLM provider must be set in config.llm.provider or via --llm-provider.")
    effective_llm_model = str(llm_model if llm_model is not None else config.llm_model).strip()
    if not effective_llm_model:
        raise ValueError("LLM model must be set in config.llm.model or via --llm-model.")
    memory_run_dir = resolve_memory_run_dir(memory_run)
    source_run_config = load_memory_run_config(memory_run_dir)

    print("=" * 80)
    print("PATIENT STATUS EXPERIMENT")
    print("=" * 80)
    print(f"Memory Run: {memory_run_dir}")
    print(f"Snapshot Stride (k): {normalized_stride}")
    print(f"Context Mode: {normalized_context_mode}")
    print(f"Oracle Results Dir: {normalized_oracle_results_dir}")
    print(f"Num Workers: {normalized_num_workers}")
    print(f"LLM Provider: {effective_llm_provider}")
    print(f"LLM Model: {effective_llm_model}")

    results_dir = memory_run_dir / _results_dir_name_for_context_mode(normalized_context_mode)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {results_dir}")

    all_records = load_memory_patient_records(memory_run_dir)
    selected_records = filter_memory_patient_records_by_stay_ids(
        records=all_records,
        patient_stay_ids_path=patient_stay_ids_path,
    )

    print(f"Patients in memory run: {len(all_records)}")
    if patient_stay_ids_path:
        print(f"Patients after --patient-stay-ids filter: {len(selected_records)}")
    selected_records, skipped_completed = _filter_completed_patient_records(selected_records, results_dir)
    if skipped_completed:
        print(f"Skipping {skipped_completed} already-completed patients")

    run_config_payload = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "patient_status_experiment",
        "memory_input": {
            "memory_run": str(memory_run_dir),
            "source_generated_at": source_run_config.get("generated_at"),
            "source_experiment": source_run_config.get("experiment"),
            "selected_patients": len(selected_records),
        },
        "logging": {
            "enable_logging": bool(enable_logging),
        },
        "execution": {
            "num_workers": normalized_num_workers,
        },
        "status_prediction": {
            "snapshot_stride": normalized_stride,
            "selection_mode": "oracle_labeled_window_time_match_then_stride",
            "window_alignment_mode": "oracle_interval_time_match",
            "context_mode": str(normalized_context_mode),
            "oracle_results_dir": str(normalized_oracle_results_dir),
            "inference_scope": "every_selected_snapshot",
            "local_events_only_scope": (
                "working_memory_last_window_only"
                if normalized_context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY
                else None
            ),
            "rag_embedding_model": (
                RAG_EMBEDDING_MODEL if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
            ),
            "rag_top_k_windows": (
                int(RAG_TOP_K_WINDOWS) if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
            ),
            "rag_retrieval_scope": (
                RAG_RETRIEVAL_SCOPE if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
            ),
        },
        "llm": {
            "provider": effective_llm_provider,
            "model": effective_llm_model,
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_tokens,
        },
    }
    _save_run_config(results_dir, run_config_payload)

    if not selected_records:
        print("No patients available for prediction.")
        return {}

    all_results: List[Dict[str, Any]] = []
    patient_data = [(idx, record) for idx, record in enumerate(selected_records, 1)]

    def process_patient_wrapper(args):
        idx, patient_record = args
        patient_agent = ICUEvoAgent(
            provider=effective_llm_provider,
            observation_config_path=config.icu_evo_observation_config_path,
            episode_block_windows=config.icu_evo_episode_block_windows,
            insight_block_windows=config.icu_evo_insight_block_windows,
            model=effective_llm_model,
            enable_logging=False,
            window_duration_hours=config.agent_current_window_hours,
            max_working_windows=config.icu_evo_max_working_windows,
            max_insights=config.icu_evo_max_insights,
            max_trajectory_entries=config.icu_evo_max_trajectory_entries,
        )
        return process_single_patient(
            patient_record=patient_record,
            agent=patient_agent,
            memory_run_config=source_run_config,
            snapshot_stride=normalized_stride,
            context_mode=normalized_context_mode,
            oracle_results_dir=normalized_oracle_results_dir,
            memory_run_dir=memory_run_dir,
            results_dir=results_dir,
            patient_idx=idx,
            total_patients=len(selected_records),
            enable_logging=enable_logging,
            verbose=verbose,
        )

    max_workers = min(normalized_num_workers, len(patient_data))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_patient_wrapper, item) for item in patient_data]
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)

    if not all_results:
        print("No patient results generated.")
        return {}

    total_patients = len(all_results)
    total_window_predictions = sum(int(item.get("num_status_predictions", 0)) for item in all_results)
    total_failed_status_predictions = sum(int(item.get("num_failed_status_predictions", 0)) for item in all_results)
    total_oracle_labeled_snapshots = sum(int(item.get("num_oracle_labeled_snapshots", 0)) for item in all_results)
    total_selected_snapshots_for_prediction = sum(
        int(item.get("num_selected_snapshots_for_prediction", 0)) for item in all_results
    )
    total_rag_retrieved_windows = sum(int(item.get("num_rag_retrieved_windows", 0)) for item in all_results)

    status_distribution: Dict[str, int] = {}
    for item in all_results:
        per_patient = item.get("status_distribution", {})
        if not isinstance(per_patient, dict):
            continue
        for label, count in per_patient.items():
            status_distribution[str(label)] = status_distribution.get(str(label), 0) + int(count)

    status_token_totals = _sum_numeric_stats([item.get("status_prediction_tokens", {}) for item in all_results])

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Total Patients: {total_patients}")
    print(f"Total Oracle-Labeled Snapshots: {total_oracle_labeled_snapshots}")
    print(f"Total Selected Snapshots for Prediction: {total_selected_snapshots_for_prediction}")
    print(f"Total Status Predictions: {total_window_predictions}")
    print(f"Failed Status Predictions: {total_failed_status_predictions}")
    print(f"Total RAG Retrieved Windows: {total_rag_retrieved_windows}")
    print("\nStatus Distribution:")
    for label, count in sorted(status_distribution.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {label}: {count}")
    print("\nStatus Predictor Tokens:")
    print(f"  Input: {int(status_token_totals.get('input_tokens', 0))}")
    print(f"  Output: {int(status_token_totals.get('output_tokens', 0))}")
    print(f"  Total: {int(status_token_totals.get('total_tokens', 0))}")

    aggregate = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "patient_status_experiment",
        "memory_run": str(memory_run_dir),
        "context_mode": str(normalized_context_mode),
        "oracle_results_dir": str(normalized_oracle_results_dir),
        "snapshot_stride": normalized_stride,
        "window_alignment_mode": "oracle_interval_time_match",
        "selection_mode": "oracle_labeled_window_time_match_then_stride",
        "num_workers": normalized_num_workers,
        "local_events_only_scope": (
            "working_memory_last_window_only" if normalized_context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY else None
        ),
        "rag_embedding_model": (
            RAG_EMBEDDING_MODEL if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
        ),
        "rag_top_k_windows": (
            int(RAG_TOP_K_WINDOWS) if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
        ),
        "rag_retrieval_scope": (
            RAG_RETRIEVAL_SCOPE if normalized_context_mode == CONTEXT_MODE_RAG_HISTORY_EVENTS else None
        ),
        "total_patients": total_patients,
        "total_oracle_labeled_snapshots": int(total_oracle_labeled_snapshots),
        "total_selected_snapshots_for_prediction": int(total_selected_snapshots_for_prediction),
        "total_window_predictions": total_window_predictions,
        "total_failed_status_predictions": total_failed_status_predictions,
        "total_rag_retrieved_windows": int(total_rag_retrieved_windows),
        "status_distribution": status_distribution,
        "status_prediction_tokens": {
            "input_tokens": int(status_token_totals.get("input_tokens", 0)),
            "output_tokens": int(status_token_totals.get("output_tokens", 0)),
            "total_tokens": int(status_token_totals.get("total_tokens", 0)),
        },
        "individual_results": sorted(all_results, key=lambda item: (item["subject_id"], item["icu_stay_id"])),
    }

    with open(results_dir / AGGREGATE_FILENAME, "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Patient-status prediction from precomputed ICU-Evo memory and event baselines"
    )
    parser.add_argument(
        "--memory-run",
        type=str,
        required=True,
        help="Path to memory run directory produced by experiments/create_memory.py.",
    )
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=8,
        help="Use every k-th oracle-labeled, time-aligned memory snapshot for status prediction.",
    )
    parser.add_argument(
        "--oracle-results-dir",
        type=str,
        required=True,
        help="Path to oracle results directory containing patients/*/oracle_predictions.json.",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        required=True,
        choices=list(SUPPORTED_CONTEXT_MODE_CHOICES),
        help=(
            "Context construction mode: "
            "memory | full_history_events | local_events_only | rag_history_events | all."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers for patient processing (default: 2).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-logging", action="store_true", help="Disable detailed LLM call logging")
    parser.add_argument(
        "--patient-stay-ids",
        type=str,
        default=None,
        help="Optional CSV with columns subject_id,icu_stay_id to filter patients from memory run.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="Optional LLM provider override for this run.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Optional LLM model override for this run.",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("RUNNING: PATIENT STATUS EXPERIMENT")
    print(f"{'='*80}\n")

    run_experiment(
        memory_run=args.memory_run,
        snapshot_stride=args.snapshot_stride,
        context_mode=args.context_mode,
        oracle_results_dir=args.oracle_results_dir,
        verbose=not args.quiet,
        enable_logging=not args.no_logging,
        patient_stay_ids_path=args.patient_stay_ids,
        num_workers=args.num_workers,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
