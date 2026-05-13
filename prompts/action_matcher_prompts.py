def get_action_matcher_prompt() -> str:
    return """
You are a clinical action matcher for ICU evaluation. For each predicted action, determine whether it semantically matches:
1) Ground Truth recommended actions
2) Ground Truth red-flag actions

## Match if:
- Same clinical intent and direction (increase/decrease/initiate/discontinue)
- Synonymous drugs within the same class used interchangeably (e.g., "norepinephrine" vs "noradrenaline")
- Different phrasings of the same action (e.g., "start vasopressors" vs "initiate norepinephrine")
- Abbreviations and full forms (e.g., "NE" vs "norepinephrine", "MAP" vs "mean arterial pressure")
- Active vs passive voice for the same action (e.g., "increase FiO2" vs "FiO2 was increased")

## Do not match if:
- Opposite directions (e.g., increase vs decrease)
- Different intervention categories (e.g., fluid bolus vs vasopressor)
- Hold vs wean vs discontinue (clinically distinct)

## Input
Predicted actions:
{predicted_actions}

Ground Truth Recommended actions:
{oracle_recommended_actions}

Ground Truth red-flag actions:
{oracle_red_flag_actions}

## Output (JSON only)
{
  "recommended_action_matches": [
    {"pred_idx": 0, "gt_indices": [...]},
    {"pred_idx": 1, "gt_indices": [...]},
    ...
  ],
  "red_flag_matches": [
    {"pred_idx": 0, "gt_indices": [...]},
    {"pred_idx": 1, "gt_indices": [...]},
    ...
  ]
}

## Important Notes
A predicted action may match zero, one, or multiple actions in each target list.
"""


def get_active_problem_matcher_prompt() -> str:
    return """
You are a clinical matcher for ICU active problems.

Match each predicted active problem to Ground Truth (GT) active problems if they refer to the same clinical condition.

## Match if:
- Same or synonymous medical condition (e.g., AKI = acute kidney injury)
- Different wording of the same problem (e.g., respiratory failure = respiratory insufficiency)
- Reasonable differences in granularity (e.g., sepsis vs septic shock)

## Do NOT match if:
- Different clinical conditions (e.g., pneumonia vs heart failure)
- One is not a medical problem
- Negation vs presence of disease

## Input
Predicted active problems:
{predicted_problems}

GT Active Problems:
{oracle_active_problems}

## Output (JSON only)
{
  "problem_matches": [
    {"pred_idx": 0, "gt_indices": [...]}
  ]
}

## Important Notes
Each predicted problem may match zero, one, or multiple GT problems.
"""
