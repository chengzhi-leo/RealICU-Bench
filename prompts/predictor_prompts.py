def get_survival_prediction_prompt() -> str:
    return """You are a clinical decision support AI specializing in ICU outcome prediction. Your task is to predict whether the patient will die during their current ICU stay, based on the observed ICU timeline provided below.

## INPUT
You will receive one of:
(A) Raw ICU events in chronological order
(B) An ICU-Evo memory object with the following layers:
    - patient_metadata
    - working_memory: raw events from recent windows — current local status
    - trend_memory: observation-agent trend statistics
    - critical_events_memory: assessment-level critical events
    - trajectory_memory: assessment summaries
    - insights: patient-specific deviations from typical ICU trajectories

## INSTRUCTIONS
1. Assess the patient's overall clinical state, trajectory, and risk of in-ICU death.
2. Prioritize high-signal information:
   - Hemodynamic instability (e.g., refractory hypotension, escalating vasopressors)
   - Respiratory failure (e.g., worsening oxygenation, ventilator dependence)
   - Organ dysfunction and progression
   - Deterioration vs. stabilization trajectory
3. If Memory is provided, use all layers — each captures a different scope of the patient's trajectory.
4. If raw events are provided, reason through key clinical states and trends internally before outputting your prediction.
5. Ground every claim strictly in the provided data.

## OUTPUT FORMAT
Respond in JSON format only:
{{
  "prediction": "Survive" | "Die",
  "confidence": "Low" | "Moderate" | "High",
  "supporting_evidence": [
    {{
      "source": "raw_event" | "trend" | "trajectory" | "insight" | "metadata",
      "content": "Brief summary of the evidence",
      "significance": "Why this supports the prediction"
    }}
  ],
  "rationale": "1–3 sentences of concise clinical reasoning"
}}

## CONSTRAINTS
- Every claim in the rationale must map to an entry in supporting_evidence.
- Include only the strongest 2–4 evidence items. Avoid redundancy.
- Do not hallucinate or infer data not present in the input.
- If data is insufficient for a confident prediction, set confidence to "Low" and explain the uncertainty in rationale.

## CLINICAL CONTEXT
{context}

Based on the above, predict whether this patient will die during their ICU stay.
"""


def get_patient_status_prediction_prompt() -> str:
    return """You are a clinical decision support AI. Assess the patient's overall clinical status for the current ICU window.

## INPUT
You will receive one of:
(A) Raw ICU events in chronological order
(B) An ICU-Evo memory object with the following layers:
    - patient_metadata
    - working_memory: raw events from recent windows — current local status
    - trend_memory: observation-agent trend statistics
    - critical_events_memory: assessment-level critical events
    - trajectory_memory: assessment summaries
    - insights: patient-specific deviations from typical ICU trajectories

## INSTRUCTIONS
### Task1. Overall Status Assessment
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

### Task2. Active Problem Identification
Identify active clinical problems or emerging risks this patient faces going forward from this window.
- Only include risks that are real and imminent or already developing — not distant or hypothetical
- Each risk must be tied to specific trajectory evidence
- An empty list is expected and acceptable when no urgent risks are present

## OUTPUT FORMAT
Respond in JSON only:
{{
  "patient_assessment": {{
    "overall": {{
      "label": "improving" | "stable" | "deteriorating" | "insufficient_data",
      "rationale": "<1-2 sentences>"
    }},
    "active_risks": [
      {{
        "risk_name": "<concise name, e.g. 'AKI progression', 'ventilator-associated pneumonia'>",
        "key_evidence": ["<Brief summary of the evidence>", ...],
      }}
    ] // Or []
  }}
}}

## CLINICAL CONTEXT
{context}
"""


def get_recommendation_action_prompt(top_k_actions: int, prediction_horizon_hours: float) -> str:
    if int(top_k_actions) < 1:
        raise ValueError(f"top_k_actions must be >= 1, got {top_k_actions}")
    if float(prediction_horizon_hours) <= 0:
        raise ValueError(f"prediction_horizon_hours must be > 0, got {prediction_horizon_hours}")

    return f"""You are a clinical decision support AI. Based on the patient's current ICU status, recommend the most appropriate clinical actions for the next {float(prediction_horizon_hours):g}-hour horizon.

## INPUT
You will receive either one of:
(A) Raw ICU events in chronological order
(B) An ICU-Evo memory object with the following layers:
    - patient_metadata
    - working_memory: raw events from recent windows — current local status
    - trend_memory: observation-agent trend statistics
    - critical_events_memory: assessment-level critical events
    - trajectory_memory: assessment summaries
    - insights: patient-specific deviations from typical ICU trajectories

## INSTRUCTIONS
### Task1. Action Recommendation
1. Recommend up to {int(top_k_actions)} distinct actions that are clinically actionable in the next {float(prediction_horizon_hours):g}-hour horizon.
2. Only recommend actions that are clearly justified by the available data. 
3. It is totally acceptable to return fewer than {int(top_k_actions)}. If data is insufficient to justify a recommendation with at least low confidence, omit it.
4. Order actions from highest to lowest clinical priority (rank 1 = most urgent).
5. Prioritize interventions with the highest expected impact on short-term stability and outcome.
6. Ground every recommendation strictly in the provided context. Do not infer or invent missing data.

### Task2. Red Flag Actions
Using the full trajectory and current window, identify any actions that should be strictly avoided for this specific patient going forward.
- Only flag actions that a reasonable clinician might consider but would be harmful for this specific patient — do not list generic contraindications unless directly applicable here
- Each flag must be justified by patient-level evidence (comorbidities, trajectory events, organ function, known sensitivities)
- An empty list is expected and acceptable when no red flags are present

## OUTPUT FORMAT
Keep language concise and clinically precise. Respond in JSON only,
{{
  "recommended_actions": [
    {{
      "rank": <integer, 1 = highest priority>,
      "action_name": "<Short action label>",
      "action_description": "<Concrete, specific action recommendation>",
      "rationale": "<1 sentence grounded in patient data>",
      "key_evidence": ["<Brief summary of the evidence>", ...],
      "confidence": "Low" | "Moderate" | "High"
    }}
  ],
  "red_flags": [
      {{
        "contraindicated_action": "<name of the action to avoid>",
        "reason": "<specific reason this action is dangerous for this patient>",
        "key_evidence": ["<Brief summary of the evidence>", ...],
      }}
    ] // Or []
}}

## CLINICAL CONTEXT
{{context}}
"""
