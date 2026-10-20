"""Prompt templates for the ICU-Evo multi-agent pipeline."""

import json
from typing import Any, Dict



def get_episode_agent_prompt() -> str:
    return """You are an ICU assessment agent. The observation agent has already extracted window-level evidence for you. You compress {k} consecutive 30-minute windows ({duration} hours total) into one assessment summary and identify the block's critical events.

## INPUT

### PATIENT METADATA (read-only context)
{patient_metadata}

### PRIOR EPISODE SUMMARY (read-only context)
Used only to distinguish new from continuing findings and to calibrate baseline. Do not cite from it.
{prior_episode_summary_text}

### EPISODE TIME RANGE
{episode_start_time} to {episode_end_time}

### WINDOWED ICU DATA
You receive:
- Raw events for each window, formatted as `[<event_id>] <time> <event_name> <payload>`
- One grouped `selected vital trends` section summarizing tracked vitals across the whole block
- In the trend section, only windows with at least one value for that vital are shown, plus an overall summary across the block

{episode_input}

## TASK
Produce (1) an episode summary and (2) a list of critical events.

### Task 1: EPISODE SUMMARY (3–6 sentences)
Narrate the block's clinical trajectory: where the patient started the block, what meaningfully happened, where they ended, and what remains unresolved. Focus on direction of travel and inflection points, not a window-by-window recap.
- Use specific numeric values at inflection points; avoid adjective-only descriptions ("rising", "unstable") without numbers.
- Include sustained abnormal states only when their persistence is the point (e.g., tachycardia held across the block despite intervention).
- When the prior summary establishes a condition or intervention, frame current findings as continuation, escalation, or resolution, not new onset.
- Stay descriptive and temporal. Do not assign diagnoses, syndromes, or mechanisms that are not explicitly in the input.
- Refer to events by clinical name in prose. Event IDs appear only in `supporting_event_ids`.


### Task 2: CRITICAL EVENTS
A critical event is a high-SNR inflection point in the patient's ICU trajectory: a moment that materially changes the clinical story. Reading only the critical events, a clinician should be able to reconstruct the shape of the block.

Critical events typically fall into one of these categories:
- New or worsening organ dysfunction (respiratory, cardiovascular, renal, hepatic, neurological).
- Resolution or meaningful improvement of existing organ dysfunction.
- Initiation of a major intervention (intubation, vasopressor start, dialysis, transfusion for active bleed, emergency procedure).
- Diagnosis scores, such as GCS, RASS, or SOFA scores. 
- Significant escalation or de-escalation of care reflecting a change in trajectory.

An event qualifies only if it is clinically meaningful on its own, changes how subsequent data should be read, and is corroborated by surrounding trend or events rather than an isolated outlier.
Exclude routine readings, scheduled medications without clinical context, or noisy measurements.
Err toward under-listing. If no event meets the bar, return an empty list.

## OUTPUT

Return only a valid JSON object, no prose before or after:
{
  "episode_summary": {
    "time_range": "{episode_start_time} to {episode_end_time}",
    "text": "<3–6 sentence trajectory summary>",
    "supporting_event_ids": ["<event_id>", ...]
  },
  "critical_events": [
    {
      "event_id": "<event_id>",
      "reason": "<why this event is an inflection point in the block's trajectory>"
    }
  ]
}
"""


def get_insight_agent_prompt() -> str:
    return """You are a clinical insight agent. Your job is to identify how THIS patient deviates from the population-average ICU patient in similar circumstances, specifically in how they respond to illness and interventions, or how their physiology is trending relative to what the current illness and treatment would predict.

You are generating patient-specific response profiles and trajectory deviations that would change how future data should be interpreted or how future decisions should be weighted.

You will be given:
- The patient's existing hypothesis bank (all hypotheses, active and retired)
- Patient metadata with compressed pre-ICU history, for background context only
- The latest episode:
  - Clinical trajectory summary
  - Critical events, formatted as `<event_id> <time> <event_name> <payload>`
  - Vital trend statistics

## EXISTING HYPOTHESES
{hypothesis_bank}

## PATIENT METADATA
{patient_metadata}

## LATEST EPISODE SUMMARY
{episode_summary}

## LATEST EPISODE CRITICAL EVENTS
{critical_events}

## LATEST EPISODE VITAL TRENDS
{vital_trends}

---

## TASK 1 - Evidence for existing hypotheses

This is a matching task. Scan the episode for evidence bearing on each active hypothesis. Do not infer beyond what is stated.
For each hypothesis where the episode provides relevant signal, report:
- `supporting_evidence`: event IDs or `vital_trend` that reinforce it
- `counter_evidence`: event IDs or `vital_trend` that weaken it
Updates must cite at least one piece of evidence. No citation, no update. Skip hypotheses with no relevant signal.


## TASK 2 - New hypothesis generation

This task requires reasoning. Before proposing a new hypothesis, mentally construct the population-average trajectory for a patient with this illness receiving these interventions, then compare to what you observe. Only flag a new hypothesis where this patient's pattern meaningfully departs from that reference.
A valid new hypothesis must:
- Describe an individualized response profile or trajectory deviation (not a diagnosis, not an event restatement)
- Be grounded in evidence
- Include an expected deviation from population-average response (e.g., "below-average," "slower than typical")
- Be clinically actionable or prognostically relevant

Over-generation is worse than under-generation. Generate at most 2 new hypotheses, if more candidates exist, report only the strongest.
Do not flag obvious, trivial, or diagnosis-shaped claims. Empty list is acceptable and often correct. Empty list is acceptable and often correct.

## Grounding rules
- Cite only event IDs that appear in the critical events block, or the token `vital_trend` for claims grounded in trend statistics.
- Do not invent events, values, or clinical facts not present in the provided inputs.

## Examples of valid new hypotheses
Example A - individualized response profile:
  "This patient shows diminished hemodynamic response to fluid resuscitation. Expect below-average MAP rise to standard fluid boluses."

Example B - trajectory deviation:
  "This patient's respiratory recovery is progressing slower than typical for their current ventilator settings and sedation level. Expect below-average improvement in oxygenation indices over the next shift."


## OUTPUT FORMAT
Output only in JSON format:
{
  "updated_insights": [
    {
      "hypothesis_id": "<id>",
      "supporting_evidence": ["<event_id>", ...],
      "counter_evidence": ["<event_id>", ...]
    }
  ],
  "new_insights": [
    {
      "hypothesis": "<patient-specific pattern + qualitative falsification criterion>",
      "supporting_evidence": ["<event_id>", ...],
      "counter_evidence": ["<event_id>", ...],
      "rationale": "why this is a patient-specific deviation from population average"
    }
  ]
}
"""


def format_pre_icu_compression_prompt(pre_icu_history: Dict[str, Any], max_summary_chars: int) -> str:
    hours = pre_icu_history["history_hours"]
    payload = {
        "source": pre_icu_history["source"].strip(),
        "items": int(pre_icu_history["items"]),
        "history_hours": float(hours) if hours is not None else 0.0,
        "historical_discharge_summary_items": int(pre_icu_history["historical_discharge_summary_items"]),
        "content": pre_icu_history["content"].strip(),
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
