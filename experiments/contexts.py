"""Construct model inputs from observations, with one masking rule for every mode."""

from copy import deepcopy

from utils.event_format import format_event_line


def current_events(window: dict, task: str, actionable_codes: list[str]) -> list[dict]:
    events = window["current_events"]
    if task == "recommendation":
        return [event for event in events if event["code"] not in actionable_codes]
    return events


def event_text(events: list[dict]) -> str:
    return "\n".join(format_event_line(event) for event in events) or "(No events)"


def memory_context(snapshot: dict, previous_snapshot: dict | None, window: dict, task: str, codes: list[str]) -> str:
    from agents.memory import WorkingWindow
    from agents.rendering import render_snapshot

    if task == "patient_status":
        return render_snapshot(snapshot)
    if previous_snapshot is None:
        raise ValueError("Memory recommendation requires the previous observation snapshot")
    context = deepcopy(previous_snapshot)
    current = WorkingWindow(
        window_id=snapshot["last_processed_window_index"],
        start_hour=window["hours_since_admission"],
        end_hour=window["hours_since_admission"] + window["current_window_hours"],
        events=current_events(window, task, codes),
    )
    context["working_memory"].append(current.to_dict())
    return render_snapshot(context)


def build_context(
    *, mode: str, task: str, windows: list[dict], index: int, actionable_codes: list[str],
    snapshots: dict, retriever,
) -> str:
    window = windows[index]
    events = current_events(window, task, actionable_codes)
    if mode == "memory":
        key = round(window["hours_since_admission"], 6), round(
            window["hours_since_admission"] + window["current_window_hours"], 6
        )
        if key not in snapshots:
            raise ValueError(f"Memory has no snapshot for observation interval {key}")
        previous = None
        if index:
            prior = windows[index - 1]
            prior_key = round(prior["hours_since_admission"], 6), round(
                prior["hours_since_admission"] + prior["current_window_hours"], 6
            )
            previous = snapshots[prior_key]
        return memory_context(snapshots[key], previous, window, task, actionable_codes)
    pre_icu = window["pre_icu_history"]["content"]
    if mode == "local_events_only":
        return event_text(events)
    if mode == "full_history_events":
        history = [event for previous in windows[:index] for event in previous["current_events"]]
        return "\n\n".join(part for part in (pre_icu, event_text(history + events)) if part)
    if mode == "rag_history_events":
        from experiments.window_rag import build_rag_window_record, render_rag_event_context

        current = build_rag_window_record(
            window_index=index, start_hour=window["hours_since_admission"],
            end_hour=window["hours_since_admission"] + window["current_window_hours"],
            event_lines=[format_event_line(event) for event in events],
        )
        retrieved = retriever.retrieve(current_window_index=index, query_text=current.event_text)
        context = render_rag_event_context(current_window=current, retrieved_windows=retrieved, retrieval_top_k=retriever.top_k)
        return "\n\n".join(part for part in (pre_icu, context) if part)
    raise ValueError(f"Unsupported context mode: {mode}")
