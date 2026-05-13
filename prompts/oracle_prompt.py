"""Prompt templates for Oracle offline evaluator."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from utils.event_format import format_event_line as format_shared_event_line
from utils.event_format import format_event_lines as format_shared_event_lines
from utils.time_format import format_timestamp_minute

ORACLE_PROMPT_TEMPLATE = """
You are Oracle, a clinical AI evaluator with hindsight access to a patient's full ICU trajectory.
Your task is to evaluate a specific local observation window {window_time} across two parts.

## PART 1 — PATIENT ASSESSMENT

### 1A. Current Status
Assess the patient's clinical direction at this window by reasoning across four domains:
- Hemodynamics: heart rate, MAP, lactate, perfusion, vasopressor requirements
- Respiratory: SpO2, PaO2/FiO2, respiratory rate, ventilator settings
- Renal/Metabolic: creatinine, urine output, electrolytes, acid-base status
- Neurology: GCS, mental status, RASS sedation score

Synthesize your domain reasoning into a single overall status label, weighted by the relative clinical importance of each domain for this specific patient:
- improving: indicators trending toward stability or recovery relative to the provided context
- stable: no meaningful change in either direction
- deteriorating: indicators trending toward worsening or decompensation relative to the provided context
- insufficient_data: available information is not sufficient to make a reliable judgment (e.g., sparse events, conflicting signals)

Important nuances:
- Do NOT conflate outcome with care quality. A patient may be deteriorating despite excellent care.
- A patient may be labelled stable or improving even if they eventually die, if the trajectory at this window genuinely reflects that direction.

### 1B. Active Problems and Risk Factors
Using the full trajectory and current window, identify active clinical problems or emerging risks this patient faces going forward from this window.
- Only include risks that are real and imminent or already developing — not distant or hypothetical
- Each risk must be tied to specific trajectory evidence
- An empty list is expected and acceptable when no urgent risks are present

## PART 2 — ACTION REVIEW

### 2A. Action Evaluation
Evaluate each clinical action taken during this window. For each action, integrate two perspectives into a single judgment:
- Guideline alignment — does this action follow established ICU guidelines (e.g., Surviving Sepsis Campaign, ARDSNet, PADIS, AHA/ACC where relevant)?
- Contextual appropriateness — given this patient's specific condition, trajectory, comorbidities, and the eventual outcome known to you, was this action appropriate?

Assign one overall label:
- best_practice: action is both guideline-aligned and well-suited to this patient's specific situation
- acceptable: action is reasonable given the context, even if not optimal or if guidelines are ambiguous
- potentially_harmful: action poses real risk of harm to this patient, whether due to guideline violation, patient-specific contraindication, or both
- insufficient_data: not enough context to evaluate this action reliably

Use your hindsight knowledge to inform the judgment, but be fair: judge the action against what the context reveals, not against impossible foresight. If an action was reasonable at the time but later context revealed a missed diagnosis, note this nuance explicitly in the rationale.

The action can be identified by its code, which includes and are not limited to: DRUG_START, DRUG_STOP, DRUG_PRESCRIPTION, BODY_INPUT, TRANSFER, LAB_TEST, DIAGNOSIS. 

### 2B. Red Flag Actions
Using the full trajectory and current window, identify any actions that should be strictly avoided for this specific patient going forward.
- Only flag actions that a reasonable clinician might consider but would be harmful for this specific patient — do not list generic contraindications unless directly applicable here
- Each flag must be justified by patient-level evidence (comorbidities, trajectory events, organ function, known sensitivities)
- An empty list is expected and acceptable when no red flags are present

## OUTPUT FORMAT
Output in valid JSON:
{
  "patient_assessment": {
    "overall": {
      "label": "<improving | stable | deteriorating | insufficient_data>",
      "rationale": "<1-2 sentences grounded in trajectory evidence>"
    },
    "active_risks": [
      {
        "risk_name": "<concise name, e.g. 'AKI progression', 'ventilator-associated pneumonia'>",
        "key_evidence": ["<event ID1>, ..."] // 1-3 event IDs
      }
    ] // Or []
  },
  "action_review": {
    "evaluations": [
      {
        "action_id": "<event ID within this ICU stay, e.g., 12>",
        "action_name": "<action name corresponding to the action_id>",
        "label": "<best_practice | acceptable | potentially_harmful | insufficient_data>",
        "rationale": "<1-2 sentences integrating guideline alignment, patient context, and hindsight>"
      }
    ],
    "red_flags": [
      {
        "contraindicated_action": "<name of the action to avoid>",
        "reason": "<specific reason this action is dangerous for this patient>",
        "key_evidence": ["<event ID1>, ..."] // 1-3 event IDs
      }
    ] // Or []
  }
}

## PATIENT ICU CONTEXT WINDOW
{patient_icu_trajectory}

Now, evaluate the CURRENT OBSERVATION WINDOW according to the instructions above.
""".strip()


def format_pre_icu_compression_prompt(pre_icu_history: Dict[str, Any], max_summary_chars: int = 1800) -> str:
    payload = {
        "source": str(pre_icu_history.get("source") or "").strip(),
        "items": int(_safe_float(pre_icu_history.get("items"))),
        "history_hours": _safe_float(pre_icu_history.get("history_hours")),
        "historical_discharge_summary_items": int(
            _safe_float(pre_icu_history.get("historical_discharge_summary_items"))
        ),
        "content": str(pre_icu_history.get("content") or "").strip(),
    }
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"""You compress pre-ICU history for repeated ICU window evaluation prompts.

Return JSON only:
{{
  "compressed_pre_icu_history": "Concise summary (<= {max_summary_chars} chars)."
}}

Requirements:
- Preserve clinically important signal; remove boilerplate.
- Focus on:
  1) Baseline vulnerability (major comorbidities, frailty, immunosuppression)
  2) Pre-ICU trajectory (course, worsening/improving, response to treatments)
  3) Acute severity signals (organ dysfunction, critical abnormal findings)
  4) Working diagnoses if evident

- Prefer interpreted clinical states over raw numbers (e.g. 'worsening hypoxia despite O2').
- Mention uncertainty if source text is sparse or conflicting.
- Do not invent events or outcomes not present in input.
- Keep the summary reusable across all subsequent ICU windows.
- Include critical patient-specific constraints (e.g.treatment-limiting allergies) if present.

Pre-ICU history payload:
{payload_json}
"""


def format_oracle_prompt(
    window_data: Dict[str, Any],
    context_block: str,
    context_mode: str,
    history_hours: Any = None,
    future_hours: Any = None,
    top_k: Any = None,
    include_icu_outcome: bool = True,
) -> str:
    """Format Oracle prompt with local ICU context window + current raw window."""
    metadata = window_data.get("patient_metadata", {})
    window_time = _format_window_time(window_data)
    current_hour_since_admission = _safe_float(window_data.get("hours_since_admission"))
    icu_duration_hours = _safe_float(metadata.get("total_icu_duration_hours"))
    outcome_text = _format_outcome(metadata.get("survived"))
    history_hours_text = _format_hours_for_template(history_hours)
    future_hours_text = _format_hours_for_template(future_hours)
    top_k_text = _format_top_k(top_k)

    patient_context = [
        "## Patient Context",
        f"- Age: {_format_age(metadata.get('age'))}",
        f"- Gender: {_format_gender(metadata.get('gender'))}",
        f"- Total ICU Stay: {icu_duration_hours:.1f} hours",
        f"- Current Hour Since ICU Admission: {current_hour_since_admission:.1f}",
        f"- Context Mode: {context_mode}",
        f"- Current Window Start: {window_data.get('current_window_start')}",
        f"- Current Window End: {window_data.get('current_window_end')}",
        "",
    ]
    if include_icu_outcome:
        patient_context.insert(4, f"- ICU Outcome: {outcome_text}")
    pre_icu_history_lines = _format_pre_icu_history_lines(window_data.get("pre_icu_history"))

    patient_trajectory = "\n".join([*patient_context, *pre_icu_history_lines, "", context_block]).strip()

    prompt = ORACLE_PROMPT_TEMPLATE
    prompt = prompt.replace("{history_hours}", history_hours_text)
    prompt = prompt.replace("{future_hours}", future_hours_text)
    prompt = prompt.replace("{top_k}", top_k_text)
    prompt = prompt.replace("{patient_icu_trajectory}", patient_trajectory)
    prompt = prompt.replace("{window_time}", window_time)
    # Backward-compatible token replacement for legacy templates.
    prompt = prompt.replace("{window time}", window_time)
    return prompt


def _format_pre_icu_history_lines(pre_icu_history: Any) -> List[str]:
    lines = ["## HISTORICAL PRE-ICU SUMMARY"]

    if not isinstance(pre_icu_history, dict):
        lines.append("No historical pre-ICU history provided.")
        return lines

    source = str(pre_icu_history.get("source") or "").strip().lower()
    items = int(_safe_float(pre_icu_history.get("items")))
    content = str(pre_icu_history.get("content") or "").strip()
    history_hours = _safe_float(pre_icu_history.get("history_hours"))

    if source in {"", "none", "disabled"}:
        lines.append("No historical pre-ICU history provided.")
    else:
        if items > 0 and history_hours > 0:
            lines.append(f"({items} item(s) from previous {history_hours:.1f} hours)")
        elif items > 0:
            lines.append(f"({items} item(s))")
        lines.append(f"Source: {source}")
        lines.append(content if content else "No historical pre-ICU history provided.")

    return lines


def format_event_line(event: Dict[str, Any]) -> str:
    """Format one event line for prompt context blocks."""
    return format_shared_event_line(event)


def format_event_lines(events: List[Dict[str, Any]], *, empty_text: str = "(No events)") -> List[str]:
    """Format multiple event lines for prompt context blocks."""
    return format_shared_event_lines(events, empty_text=empty_text)


def _format_age(age: Any) -> str:
    if age is None:
        return "Unknown"
    try:
        return f"{float(age):.1f} years"
    except (TypeError, ValueError):
        return str(age)


def _format_gender(gender: Any) -> str:
    text = str(gender).strip() if gender is not None else ""
    if not text:
        return "Unknown"
    normalized = text.lower()
    if normalized in {"none", "nan", "null", "nat", "unknown"}:
        return "Unknown"
    if normalized in {"m", "male"}:
        return "Male"
    if normalized in {"f", "female"}:
        return "Female"
    return text


def _format_outcome(survived: Any) -> str:
    if isinstance(survived, bool):
        return "Survived after ICU" if survived else "Died after ICU"

    text = str(survived).strip().lower()
    if text in {"true", "1", "yes", "y", "survived", "alive"}:
        return "Survived after ICU"
    if text in {"false", "0", "no", "n", "died", "dead", "deceased"}:
        return "Died after ICU"
    return "Unknown"


def _format_hours_for_template(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _format_top_k(value: Any) -> str:
    if value is None:
        return "3"
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return str(value)
    if as_int < 1:
        as_int = 1
    return str(as_int)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_window_time(window_data: Dict[str, Any]) -> str:
    start = window_data.get("current_window_start")
    end = window_data.get("current_window_end")
    start_text = _format_time(start) if start else "unknown_start"
    end_text = _format_time(end) if end else "unknown_end"
    return f"{start_text} to {end_text}"


def _format_time(value: Any) -> str:
    return format_timestamp_minute(value)
