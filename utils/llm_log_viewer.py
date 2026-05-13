"""General-purpose HTML viewer for LLM call logs."""

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from utils.status_scoring import STATUS_SCORE_MAP
except ImportError:  # pragma: no cover - script execution fallback
    from status_scoring import STATUS_SCORE_MAP

ACTION_SCORE_MAP: Dict[str, float] = {
    "potentially_harmful": -1.0,
    "acceptable": 0.5,
    "best_practice": 1.0,
}

NON_SCORABLE_ACTION_LABELS = {
    "insufficient_data",
}


def get_call_step_type(call: Dict[str, Any]) -> str:
    """Extract step type from a call record."""
    metadata = call.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("step_type"):
        return str(metadata["step_type"])
    if call.get("step_type"):
        return str(call["step_type"])
    return "unknown"


def build_pipeline_agents(agent: Any, agent_type: str) -> List[Dict[str, Any]]:
    """Build pipeline metadata (agent name + usage) from an agent instance."""
    if agent_type == "multi":
        use_observer_agent = bool(getattr(agent, "use_observer_agent", True))
        use_memory_agent = bool(getattr(agent, "use_memory_agent", False))
        use_reflection_agent = bool(getattr(agent, "use_reflection_agent", False)) and use_memory_agent
        return [
            {"name": "observer", "used": use_observer_agent},
            {"name": "memory_agent", "used": use_memory_agent},
            {"name": "reflection_agent", "used": use_reflection_agent},
            {"name": "predictor", "used": True},
        ]

    if agent_type == "fold":
        return [{"name": "fold_agent", "used": True}]

    if agent_type == "remem":
        return [{"name": "remem_agent", "used": True}]

    if agent_type in {"med_evo", "icu_evo"}:
        episode_every_n = int(getattr(agent, "episode_every_n_windows", 0) or 0)
        return [
            {"name": "observation_agent", "used": True},
            {"name": "observation_rules", "used": True},
            {"name": "insight_agent", "used": True},
            {"name": "assessment_agent", "used": episode_every_n > 0},
            {"name": "predictor", "used": True},
        ]

    return []


def _format_json_block(value: Any) -> str:
    """Format any object as pretty JSON text."""
    if value is None:
        return "null"
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except TypeError:
        return str(value)


def _resolve_pipeline_agents(patient_logs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """Resolve pipeline metadata from logs; infer from calls if missing."""
    pipeline_agents = patient_logs.get("pipeline_agents")
    if isinstance(pipeline_agents, list) and pipeline_agents:
        normalized: List[Dict[str, Any]] = []
        for agent_info in pipeline_agents:
            if not isinstance(agent_info, dict):
                continue
            normalized.append(
                {
                    "name": str(agent_info.get("name", "unknown")),
                    "used": bool(agent_info.get("used", False)),
                }
            )
        if normalized:
            return normalized, False

    calls = patient_logs.get("calls", [])
    step_types = {get_call_step_type(call) for call in calls}
    inferred = [
        {"name": "observer", "used": "observer" in step_types},
        {
            "name": "memory_agent",
            "used": any(step in step_types for step in ["memory_agent", "memory_agent_revision"]),
        },
        {"name": "reflection_agent", "used": "reflection_agent" in step_types},
        {"name": "predictor", "used": "predictor" in step_types},
    ]
    inferred = [agent for agent in inferred if agent["used"]]
    return inferred, True


def _normalize_identity_value(value: Any) -> str:
    """Normalize optional identity values from logs."""
    if value is None:
        return ""
    text = str(value).strip()
    return text if text else ""


def _resolve_llm_identity(patient_logs: Dict[str, Any], calls: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Resolve LLM provider/model, falling back to per-call metadata for older logs."""
    provider = _normalize_identity_value(patient_logs.get("llm_provider"))
    model = _normalize_identity_value(patient_logs.get("llm_model"))
    if provider or model:
        return provider or "unknown", model or "unknown"

    for call in calls:
        metadata = call.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        provider = _normalize_identity_value(metadata.get("llm_provider"))
        model = _normalize_identity_value(metadata.get("llm_model"))
        if provider or model:
            return provider or "unknown", model or "unknown"

    return "unknown", "unknown"


def _format_llm_identity(provider: str, model: str) -> str:
    """Build a compact display string for provider/model."""
    if provider != "unknown" and model != "unknown":
        return f"{provider} / {model}"
    if model != "unknown":
        return model
    if provider != "unknown":
        return provider
    return "unknown"


def _resolve_prompt_outcome_mode(patient_logs: Dict[str, Any], calls: List[Dict[str, Any]]) -> str:
    """Resolve whether ICU outcome was included in prompt context."""
    value = patient_logs.get("include_icu_outcome_in_prompt")
    if isinstance(value, bool):
        return "with_icu_outcome" if value else "without_icu_outcome"

    mode = patient_logs.get("prompt_outcome_mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()

    for call in calls:
        metadata = call.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        meta_value = metadata.get("include_icu_outcome_in_prompt")
        if isinstance(meta_value, bool):
            return "with_icu_outcome" if meta_value else "without_icu_outcome"

    return "unknown"


def _format_prompt_outcome_mode(mode: str) -> str:
    if mode == "with_icu_outcome":
        return "Included"
    if mode == "without_icu_outcome":
        return "Excluded"
    return "Unknown"


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip().lower()


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _call_sort_key(call: Dict[str, Any]) -> Tuple[int, float, str]:
    window_index = _coerce_int(call.get("window_index"))
    if window_index < 0:
        window_index = 10**9

    hours = _coerce_float(call.get("hours_since_admission"))
    if hours != hours:  # NaN
        hours = float("inf")

    timestamp = str(call.get("timestamp") or "")
    return window_index, hours, timestamp


def _format_score(value: float) -> str:
    if value != value:  # NaN check
        return "n/a"
    return f"{value:.2f}"


def _nanmean(values: List[float]) -> float:
    valid_values = [value for value in values if value == value]
    if not valid_values:
        return float("nan")
    return sum(valid_values) / len(valid_values)


def _extract_label(candidate: Any) -> str:
    if isinstance(candidate, dict):
        for key in ("label", "status", "value"):
            label = _normalize_label(candidate.get(key))
            if label:
                return label
        return ""
    return _normalize_label(candidate)


def _is_oracle_parsed_payload(parsed_response: Any) -> bool:
    if not isinstance(parsed_response, dict):
        return False
    keys = {"patient_assessment", "action_review", "overall", "evaluations"}
    return any(key in parsed_response for key in keys)


def _extract_patient_status_payload(parsed_response: Dict[str, Any]) -> Dict[str, Any]:
    patient_assessment = parsed_response.get("patient_assessment")
    if isinstance(patient_assessment, dict):
        return patient_assessment
    return {}


def _extract_patient_overall_label(patient_status: Dict[str, Any]) -> str:
    overall = patient_status.get("overall")
    label = _extract_label(overall)
    if label:
        return label
    return ""


def _extract_action_label(action_eval: Dict[str, Any]) -> str:
    if not isinstance(action_eval, dict):
        return ""
    for key in ("label", "overall", "contextual_appropriateness", "guideline_adherence", "quality_rating"):
        label = _extract_label(action_eval.get(key))
        if label:
            return label
    return ""


def _build_oracle_trend_rows(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trend_rows_by_window: Dict[int, Dict[str, Any]] = {}
    for call in calls:
        parsed = call.get("parsed_response")
        if not _is_oracle_parsed_payload(parsed):
            continue
        step_type = _normalize_label(get_call_step_type(call))
        if step_type not in {"", "unknown", "oracle", "oracle_evaluator"}:
            continue

        window_index = _coerce_int(call.get("window_index"))
        if window_index < 0:
            continue
        hours_since_admission = _coerce_float(call.get("hours_since_admission"))

        patient_status = _extract_patient_status_payload(parsed)
        status_label = _extract_patient_overall_label(patient_status)
        status_score = STATUS_SCORE_MAP.get(status_label, float("nan"))

        action_labels: List[str] = []
        action_scores: List[float] = []
        action_review = parsed.get("action_review")
        action_evaluations = action_review.get("evaluations") if isinstance(action_review, dict) else []
        if isinstance(action_evaluations, list):
            for action_eval in action_evaluations:
                label = _extract_action_label(action_eval)
                if not label:
                    continue
                action_labels.append(label)
                score = ACTION_SCORE_MAP.get(label)
                if score is not None:
                    action_scores.append(score)

        action_score = float("nan")
        if action_scores:
            action_score = sum(action_scores) / len(action_scores)

        if not status_label and not action_labels:
            continue

        trend_rows_by_window[window_index] = {
            "window_index": window_index,
            "hours_since_admission": hours_since_admission,
            "status_label": status_label,
            "status_score": status_score,
            "action_labels": action_labels,
            "action_score": action_score,
            "action_total": len(action_labels),
            "action_scorable": len(action_scores),
        }

    trend_rows = list(trend_rows_by_window.values())
    trend_rows.sort(key=lambda row: row["window_index"])
    return trend_rows


def _build_trend_badges(counter: Dict[str, int], badge_class: str) -> str:
    if not counter:
        return "<span class='trend-badge muted'>none</span>"
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return "".join(
        f"<span class='{badge_class}'><strong>{escape(label)}</strong>: {count}</span>" for label, count in items
    )


def _build_oracle_trend_chart_svg(
    points: List[Dict[str, Any]],
    y_ticks: List[Tuple[float, str]],
    line_color: str,
    point_fill: str,
    empty_message: str,
) -> str:
    width = 960
    height = 260
    margin_left = 72
    margin_right = 20
    margin_top = 16
    margin_bottom = 44
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not points:
        return (
            f"<svg class='trend-svg' viewBox='0 0 {width} {height}' role='img' aria-label='Empty chart'>"
            f"<text x='{width / 2:.1f}' y='{height / 2:.1f}' text-anchor='middle' fill='#6b7280' font-size='14'>"
            f"{escape(empty_message)}</text></svg>"
        )

    y_min = min(value for value, _ in y_ticks)
    y_max = max(value for value, _ in y_ticks)
    if y_max <= y_min:
        y_max = y_min + 1.0

    point_count = len(points)

    def x_pos(index: int) -> float:
        if point_count == 1:
            return margin_left + (plot_width / 2.0)
        return margin_left + (index * plot_width / float(point_count - 1))

    def y_pos(value: float) -> float:
        bounded = min(max(value, y_min), y_max)
        ratio = (bounded - y_min) / (y_max - y_min)
        return margin_top + (1.0 - ratio) * plot_height

    layers: List[str] = []
    layers.append(
        f"<rect x='{margin_left}' y='{margin_top}' width='{plot_width}' height='{plot_height}' "
        "fill='#f8fafc' stroke='#e5e7eb'/>"
    )

    for tick_value, tick_label in y_ticks:
        y = y_pos(tick_value)
        layers.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{margin_left + plot_width}' y2='{y:.2f}' "
            "stroke='#e5e7eb' stroke-width='1'/>"
        )
        layers.append(
            f"<text x='{margin_left - 8}' y='{y + 4:.2f}' text-anchor='end' fill='#4b5563' font-size='11'>"
            f"{escape(tick_label)}</text>"
        )

    tick_step = max(1, point_count // 12)
    x_tick_indices = list(range(0, point_count, tick_step))
    if x_tick_indices[-1] != point_count - 1:
        x_tick_indices.append(point_count - 1)
    for point_index in sorted(set(x_tick_indices)):
        x = x_pos(point_index)
        label = str(points[point_index]["window_index"])
        layers.append(
            f"<line x1='{x:.2f}' y1='{margin_top + plot_height}' x2='{x:.2f}' y2='{margin_top + plot_height + 4}' "
            "stroke='#9ca3af' stroke-width='1'/>"
        )
        layers.append(
            f"<text x='{x:.2f}' y='{margin_top + plot_height + 18}' text-anchor='middle' fill='#6b7280' "
            f"font-size='11'>{escape(label)}</text>"
        )

    segment: List[str] = []
    for point_index, point in enumerate(points):
        score = point["score"]
        if score != score:  # NaN
            if len(segment) >= 2:
                layers.append(
                    f"<polyline points='{' '.join(segment)}' fill='none' stroke='{line_color}' "
                    "stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/>"
                )
            segment = []
            continue
        x = x_pos(point_index)
        y = y_pos(score)
        segment.append(f"{x:.2f},{y:.2f}")
    if len(segment) >= 2:
        layers.append(
            f"<polyline points='{' '.join(segment)}' fill='none' stroke='{line_color}' "
            "stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/>"
        )

    for point_index, point in enumerate(points):
        score = point["score"]
        if score != score:  # NaN
            continue
        x = x_pos(point_index)
        y = y_pos(score)
        tooltip = (
            f"Window {point['window_index']}"
            f" | ICU hour {point['hours_label']}"
            f" | Label: {point['label'] or 'n/a'}"
            f" | Score: {_format_score(score)}"
        )
        layers.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.5' fill='{point_fill}' stroke='white' stroke-width='1.5'>"
            f"<title>{escape(tooltip)}</title></circle>"
        )

    return f"<svg class='trend-svg' viewBox='0 0 {width} {height}' role='img'>{''.join(layers)}</svg>"


def _build_oracle_trend_table(rows: List[Dict[str, Any]]) -> str:
    table_rows: List[str] = []
    for row in rows:
        hours_value = row["hours_since_admission"]
        hours_text = "n/a" if hours_value != hours_value else f"{hours_value:.2f}"
        action_labels = ", ".join(row["action_labels"]) if row["action_labels"] else "-"
        table_rows.append(
            "<tr>"
            f"<td>{row['window_index']}</td>"
            f"<td>{hours_text}</td>"
            f"<td>{escape(row['status_label'] or '-')}</td>"
            f"<td>{escape(_format_score(row['status_score']))}</td>"
            f"<td>{escape(action_labels)}</td>"
            f"<td>{escape(_format_score(row['action_score']))}</td>"
            f"<td>{row['action_scorable']}/{row['action_total']}</td>"
            "</tr>"
        )

    return (
        "<details class='trend-table'><summary>Window-Level Trend Data</summary>"
        "<div class='trend-table-wrap'><table><thead><tr>"
        "<th>Window</th><th>ICU Hour</th><th>Patient Status</th><th>Status Score</th>"
        "<th>Action Labels</th><th>Action Score</th><th>Scorable Actions</th>"
        "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table></div></details>"
    )


def _save_trend_png(
    points: List[Dict[str, Any]],
    y_ticks: List[Tuple[float, str]],
    line_color: str,
    point_fill: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x_values = [point["window_index"] for point in points]
    y_values = [point["score"] if point["score"] == point["score"] else float("nan") for point in points]

    fig, ax = plt.subplots(figsize=(9.4, 3.2), dpi=160)
    ax.set_facecolor("#f8fafc")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=1.0)

    ax.plot(
        x_values,
        y_values,
        color=line_color,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        markerfacecolor=point_fill,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )

    tick_values = [value for value, _ in y_ticks]
    tick_labels = [label for _, label in y_ticks]
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("ICU Window Index")
    ax.set_title(title, fontsize=11)

    if x_values:
        tick_step = max(1, len(x_values) // 12)
        x_ticks = list(x_values[::tick_step])
        if x_values[-1] not in x_ticks:
            x_ticks.append(x_values[-1])
        ax.set_xticks(sorted(set(x_ticks)))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_oracle_trend_figures(
    status_points: List[Dict[str, Any]],
    action_points: List[Dict[str, Any]],
    output_dir: Optional[Path],
) -> Dict[str, str]:
    if output_dir is None:
        return {}

    if not status_points and not action_points:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    status_name = "oracle_patient_status_trend.png"
    action_name = "oracle_doctor_action_score_trend.png"
    status_path = output_dir / status_name
    action_path = output_dir / action_name

    _save_trend_png(
        points=status_points,
        y_ticks=[(-1.0, "Deteriorating"), (-0.5, "Fluctuating"), (0.0, "Stable"), (1.0, "Improving")],
        line_color="#1d4ed8",
        point_fill="#1d4ed8",
        title="Patient Status Trend",
        output_path=status_path,
    )
    _save_trend_png(
        points=action_points,
        y_ticks=[(-1.0, "Potentially harmful"), (0.0, "Insufficient"), (0.5, "Acceptable"), (1.0, "Best practice")],
        line_color="#b45309",
        point_fill="#b45309",
        title="Doctor Action Score Trend",
        output_path=action_path,
    )

    # Remove old SVG exports from earlier versions so only PNG artifacts remain.
    for stale_svg in ("oracle_patient_status_trend.svg", "oracle_doctor_action_score_trend.svg"):
        stale_path = output_dir / stale_svg
        if stale_path.exists():
            stale_path.unlink()

    return {"status": status_name, "action": action_name}


def _build_oracle_trend_section(calls: List[Dict[str, Any]], output_dir: Optional[Path] = None) -> str:
    rows = _build_oracle_trend_rows(calls)
    if not rows:
        return ""

    status_counter: Dict[str, int] = {}
    action_counter: Dict[str, int] = {}
    for row in rows:
        status_label = row["status_label"]
        if status_label:
            status_counter[status_label] = status_counter.get(status_label, 0) + 1
        for action_label in row["action_labels"]:
            action_counter[action_label] = action_counter.get(action_label, 0) + 1

    status_points = [
        {
            "window_index": row["window_index"],
            "hours_label": _format_score(row["hours_since_admission"]),
            "label": row["status_label"],
            "score": row["status_score"],
        }
        for row in rows
    ]
    action_points = [
        {
            "window_index": row["window_index"],
            "hours_label": _format_score(row["hours_since_admission"]),
            "label": ", ".join(row["action_labels"]) if row["action_labels"] else "",
            "score": row["action_score"],
        }
        for row in rows
    ]

    status_svg = _build_oracle_trend_chart_svg(
        points=status_points,
        y_ticks=[(-1.0, "Deteriorating"), (-0.5, "Fluctuating"), (0.0, "Stable"), (1.0, "Improving")],
        line_color="#1d4ed8",
        point_fill="#1d4ed8",
        empty_message="No patient-status trend points",
    )
    action_svg = _build_oracle_trend_chart_svg(
        points=action_points,
        y_ticks=[(-1.0, "Potentially harmful"), (0.0, "Insufficient"), (0.5, "Acceptable"), (1.0, "Best practice")],
        line_color="#b45309",
        point_fill="#b45309",
        empty_message="No doctor-action trend points",
    )
    figure_files = _save_oracle_trend_figures(status_points, action_points, output_dir=output_dir)

    windows_with_status = sum(1 for row in rows if row["status_score"] == row["status_score"])
    windows_with_action_score = sum(1 for row in rows if row["action_score"] == row["action_score"])
    total_actions = sum(row["action_total"] for row in rows)
    total_scorable_actions = sum(row["action_scorable"] for row in rows)

    average_status_score = _nanmean([row["status_score"] for row in rows])
    average_action_score = float("nan")
    if total_scorable_actions > 0:
        weighted_sum = sum(
            row["action_score"] * row["action_scorable"] for row in rows if row["action_score"] == row["action_score"]
        )
        average_action_score = weighted_sum / total_scorable_actions

    non_scorable_count = sum(count for label, count in action_counter.items() if label in NON_SCORABLE_ACTION_LABELS)

    status_mapping_text = "deteriorating=-1, fluctuating=-0.5, stable=0.5, improving=1"
    action_mapping_text = "potentially_harmful=-1, insufficient_data=0, acceptable=0.5, best_practice=1"
    saved_figures_html = ""
    if figure_files:
        status_name = figure_files["status"]
        action_name = figure_files["action"]
        saved_figures_html = (
            "<p class='muted'>Saved figures: "
            f"<a href='{escape(status_name)}'>{escape(status_name)}</a> | "
            f"<a href='{escape(action_name)}'>{escape(action_name)}</a>"
            "</p>"
        )

    return f"""
<section class="trend-section">
  <h2>Oracle Trend View</h2>
  <p class="muted">
    Windows parsed: <strong>{len(rows)}</strong> |
    status points: <strong>{windows_with_status}</strong> |
    action-score points: <strong>{windows_with_action_score}</strong> |
    actions evaluated: <strong>{total_actions}</strong>
  </p>
  <p class="muted">
    Avg patient-status score: <strong>{_format_score(average_status_score)}</strong> |
    Avg doctor-action score: <strong>{_format_score(average_action_score)}</strong>
    <span style="color:#6b7280">(across all scorable action evaluations)</span>
  </p>
  <p class="muted">Status scoring: <code>{escape(status_mapping_text)}</code></p>
  <p class="muted">Action scoring: <code>{escape(action_mapping_text)}</code></p>
  <p class="muted">
    Doctor-action label source used for scoring:
    <code>overall</code> first, then fallback to
    <code>contextual_appropriateness</code>,
    <code>guideline_adherence</code>,
    <code>quality_rating</code>.
  </p>
  <p class="muted">
    Non-scorable action labels (excluded from action score):
    <strong>{non_scorable_count}</strong>
    (<code>insufficient_data</code>, <code>not_enough_context</code>, <code>not_applicable</code>, <code>guideline_unclear</code>)
  </p>
  {saved_figures_html}
  <div class="trend-badges">
    <span class="trend-label">Patient status labels:</span>
    {_build_trend_badges(status_counter, "trend-badge")}
  </div>
  <div class="trend-badges">
    <span class="trend-label">Doctor action labels:</span>
    {_build_trend_badges(action_counter, "trend-badge")}
  </div>
  <div class="trend-grid">
    <div class="trend-card">
      <h3>Patient Status Trend</h3>
      <p class="muted">X axis: ICU windows | Y axis: status score</p>
      {status_svg}
    </div>
    <div class="trend-card">
      <h3>Doctor Action Score Trend</h3>
      <p class="muted">X axis: ICU windows | Y axis: average action score per window</p>
      {action_svg}
    </div>
  </div>
  {_build_oracle_trend_table(rows)}
</section>
"""


def save_llm_calls_html(patient_logs: Dict[str, Any], output_path: Path) -> None:
    """Save an interactive HTML viewer for a patient llm_calls payload."""
    calls = sorted(patient_logs.get("calls", []), key=_call_sort_key)
    pipeline_agents, inferred_pipeline = _resolve_pipeline_agents(patient_logs)
    llm_provider, llm_model = _resolve_llm_identity(patient_logs, calls)
    llm_display = _format_llm_identity(llm_provider, llm_model)
    prompt_outcome_mode = _resolve_prompt_outcome_mode(patient_logs, calls)
    prompt_outcome_display = _format_prompt_outcome_mode(prompt_outcome_mode)
    oracle_trend_section = _build_oracle_trend_section(calls, output_dir=output_path.parent)

    step_counts: Dict[str, int] = {}
    for call in calls:
        step = get_call_step_type(call)
        step_counts[step] = step_counts.get(step, 0) + 1

    step_badges = "".join(
        f"<span class='badge'><strong>{escape(step)}</strong>: {count}</span>"
        for step, count in sorted(step_counts.items())
    )

    pipeline_rows = "".join(
        "<tr>"
        f"<td>{escape(str(agent_info.get('name', 'unknown')))}</td>"
        f"<td>{'Yes' if agent_info.get('used') else 'No'}</td>"
        "</tr>"
        for agent_info in pipeline_agents
    )

    call_sections = []
    for idx, call in enumerate(calls, 1):
        step_type = get_call_step_type(call)
        window_index = call.get("window_index", "n/a")
        hours = call.get("hours_since_admission", "n/a")
        input_tokens = call.get("input_tokens", 0)
        output_tokens = call.get("output_tokens", 0)
        timestamp = escape(str(call.get("timestamp", "")))

        # not show the context_history_events, context_current_window_events, context_future_events from metadata to avoid overwhelming the display with long event lists; these can be inspected in the raw JSON if needed
        meta_data = call.get("metadata", {})
        if isinstance(meta_data, dict):
            meta_data = {
                k: v
                for k, v in meta_data.items()
                if k not in {"context_history_events", "context_current_window_events", "context_future_events"}
            }
        metadata_text = _format_json_block(meta_data)
        parsed_text = _format_json_block(call.get("parsed_response"))
        prompt_text = call.get("prompt")
        if prompt_text is None:
            prompt_text = ""
        response_text = call.get("response")
        if response_text is None:
            response_text = ""

        section = f"""
<details class="call">
  <summary>
    <span class="call-index">#{idx}</span>
    <span class="call-step">{escape(step_type)}</span>
    <span>window {window_index}</span>
    <span>hour {hours}</span>
    <span>tokens {input_tokens} in / {output_tokens} out</span>
    <span class="timestamp">{timestamp}</span>
  </summary>
  <div class="call-content">
    <div class="panel">
      <h3>Metadata</h3>
      <pre>{escape(metadata_text)}</pre>
    </div>
    <div class="panel">
      <h3>Parsed Response</h3>
      <pre>{escape(parsed_text)}</pre>
    </div>
    <div class="panel">
      <h3>Prompt</h3>
      <pre>{escape(prompt_text)}</pre>
    </div>
    <div class="panel">
      <h3>Response</h3>
      <pre>{escape(response_text)}</pre>
    </div>
  </div>
</details>
"""
        call_sections.append(section)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Call Viewer - {escape(patient_logs.get("patient_id", "unknown"))}</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f4f6;
      color: #111827;
    }}
    .container {{
      max-width: 1100px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }}
    .header {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .header h1 {{
      margin: 0 0 8px 0;
      font-size: 20px;
    }}
    .muted {{
      color: #4b5563;
      margin: 0 0 10px 0;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pipeline {{
      margin-top: 10px;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      overflow: hidden;
    }}
    .pipeline table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #fff;
    }}
    .pipeline th {{
      text-align: left;
      background: #f9fafb;
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
    }}
    .pipeline td {{
      border-top: 1px solid #f3f4f6;
      padding: 8px 10px;
    }}
    .badge {{
      background: #eef2ff;
      color: #1e40af;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      border: 1px solid #c7d2fe;
    }}
    .trend-section {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    .trend-section h2 {{
      margin: 0 0 8px 0;
      font-size: 18px;
    }}
    .trend-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-top: 10px;
    }}
    .trend-card {{
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #ffffff;
      padding: 10px;
    }}
    .trend-card h3 {{
      margin: 0 0 6px 0;
      font-size: 14px;
    }}
    .trend-svg {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
      background: #ffffff;
    }}
    .trend-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
    }}
    .trend-label {{
      font-size: 12px;
      color: #4b5563;
    }}
    .trend-badge {{
      background: #ecfeff;
      color: #155e75;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 11px;
      border: 1px solid #a5f3fc;
    }}
    .trend-table {{
      margin-top: 12px;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #ffffff;
      overflow: hidden;
    }}
    .trend-table > summary {{
      cursor: pointer;
      padding: 9px 12px;
      background: #f8fafc;
      font-size: 13px;
      font-weight: 600;
      border-bottom: 1px solid #e5e7eb;
    }}
    .trend-table-wrap {{
      overflow-x: auto;
    }}
    .trend-table table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      min-width: 740px;
    }}
    .trend-table th {{
      text-align: left;
      background: #f8fafc;
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
    }}
    .trend-table td {{
      border-top: 1px solid #f3f4f6;
      padding: 7px 10px;
      vertical-align: top;
    }}
    details.call {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      margin-bottom: 10px;
      overflow: hidden;
    }}
    details.call > summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 10px 14px;
      cursor: pointer;
      list-style: none;
      border-bottom: 1px solid transparent;
      background: #fafafa;
      font-size: 13px;
    }}
    details.call[open] > summary {{
      border-bottom-color: #e5e7eb;
      background: #f9fafb;
    }}
    .call-index {{
      font-weight: 700;
      color: #374151;
    }}
    .call-step {{
      font-weight: 600;
      color: #1d4ed8;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 6px;
      padding: 2px 8px;
    }}
    .timestamp {{
      margin-left: auto;
      color: #6b7280;
      font-size: 12px;
    }}
    .call-content {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      padding: 12px;
    }}
    .panel {{
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #ffffff;
    }}
    .panel h3 {{
      margin: 0;
      padding: 8px 10px;
      font-size: 13px;
      border-bottom: 1px solid #e5e7eb;
      background: #f9fafb;
    }}
    pre {{
      margin: 0;
      padding: 10px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.4;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      background: #fff;
    }}
    @media (min-width: 960px) {{
      .trend-grid {{
        grid-template-columns: 1fr 1fr;
      }}
      .call-content {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>LLM Call Viewer</h1>
      <p class="muted">Patient: <strong>{escape(patient_logs.get("patient_id", "unknown"))}</strong> | Total calls: <strong>{len(calls)}</strong></p>
      <p class="muted">LLM: <strong>{escape(llm_display)}</strong></p>
      <p class="muted">ICU Outcome In Prompt: <strong>{escape(prompt_outcome_display)}</strong> <span style="color:#6b7280">({escape(prompt_outcome_mode)})</span></p>
      <div class="badges">{step_badges}</div>
      <div class="pipeline">
        <table>
          <thead>
            <tr>
              <th>Agent</th>
              <th>Used</th>
            </tr>
          </thead>
          <tbody>{pipeline_rows}</tbody>
        </table>
      </div>
      {"<p class='muted'>Pipeline info inferred from call steps.</p>" if inferred_pipeline else ""}
    </div>
    {oracle_trend_section}
    {"".join(call_sections)}
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_html_from_json(llm_calls_json_path: Path, output_path: Path = None) -> Path:
    """Generate an HTML viewer from a llm_calls.json file."""
    if output_path is None:
        output_path = llm_calls_json_path.with_suffix(".html")

    with open(llm_calls_json_path, "r", encoding="utf-8") as f:
        patient_logs = json.load(f)
    save_llm_calls_html(patient_logs, output_path)
    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML viewer from llm_calls.json")
    parser.add_argument("llm_calls_json", type=Path, help="Path to llm_calls.json")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: same name with .html)")
    args = parser.parse_args()

    output_path = generate_html_from_json(args.llm_calls_json, args.output)
    print(f"Generated HTML viewer: {output_path}")


if __name__ == "__main__":
    _main()
