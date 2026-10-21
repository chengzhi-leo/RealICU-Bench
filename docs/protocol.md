# RealICU protocol `realicu-v1`

This version records the masking, sampling, reference, and metric definitions
used for the published RealICU results. The machine-readable declaration is
`config/protocols/realicu-v1.json`. Experimental choices such as K, lookback,
stride, model, and horizon remain explicit run settings.

## Observation windows

Join trajectories with `stay_id`; an event's encounter-specific `icu_stay_id`
can refer to a historical encounter. Preserve `source_event_index` and
`icu_event_index`. Select events in the actual half-open interval
`[window_start_time, window_end_time)` from published timestamps. Nominal hours
and nonconsecutive window indices do not establish actual endpoints.

Memory and prediction runners create the same chronological observation sequence.
It contains every window with at least one observed event, including unlabeled
gaps; windows without events are skipped and do not receive an index. The
configured cadence must agree with published windows. Short final intervals
retain their actual endpoints. Observation caps include only complete published
evaluation windows. `input.excluded_event_codes` are removed from ICU
observation windows.

Local-window predictions see only current observations. Full-history predictions
see past ICU windows plus pre-ICU history if explicitly enabled. RAG selects past
ICU windows using the current observed events as query; future and current
windows are excluded from retrieval candidates. Memory mode uses ICU-Evo's
compressed history and memory structures.

The current-window mode requires pre-ICU input to be disabled. Full-history and
RAG contexts include the chosen pre-ICU history when enabled; memory mode uses
its compressed version. Pre-ICU history lists every event in the lookback before
ICU admission, labeled with its `source_event_index`, followed by the
`input.pre_icu_discharge_summaries` most recent earlier discharge summaries.
Window selection by timestamp does not establish when every source clinical
note became available.

## Recommendation masking and windows

Before recommendation prediction, remove current observations whose codes are
`DRUG_START`, `DRUG_PRESCRIPTION`, `PROCEDURE`, `BODY_INPUT`, or `TRANSFER`.
Historical recorded actions remain visible. In memory mode, combine the previous
observation snapshot with masked current raw observations; do not expose
summaries generated from unmasked current actions.

Positive references are recorded actions labeled `acceptable` or
`best_practice`. Recommendation prediction considers windows whose available
action review is nonempty and applies the stride to them. It then skips the
first observation window and windows without positive references among
annotation windows whose actual start lies in **[S, S + H]**, where S is the
current window start and H is `prediction.horizon_hours`.

Scoring compares recommendations with the positive references of the current
window only. Windows without current positive references are not scored for
recommendations or red flags.

## Sampling and failures

Sort eligible windows chronologically, select every configured k-th window from
the first, and retain the final eligible window. Status prediction also
produces acute problems; it runs where status or active-risk references are
available, and each scorer uses its own availability.

Inference and parse errors produce explicit failed prediction records. Status
failures receive an `inference_error` predicted label, count as incorrect, and
contribute no acute problems. Recommendation failures provide zero recommended
actions and therefore zero hit and recall. A
full-history context-limit failure prevents further inference on that stay;
later selected windows remain explicit failures. Missing predictions are
reported in coverage, not silently replaced by inferred labels.

Malformed data or configuration fail the run. ICU-Evo retains the earlier
schema-recovery behavior after its explicitly configured attempt limit. Failed
pre-ICU compression leaves the history uncompressed and omits its summary from
memory metadata. Failed episode generation produces the original deterministic
window/trend summary with event references. Failed insight generation retains
the original evidence-based fallback hypothesis. Model-supplied episodes with
no valid supporting IDs use the first twelve available block event IDs. These
are model-output recovery rules, not defaults for missing configuration.

## Matching

The embedding matcher selects the highest-similarity reference for each
proposed item and retains it when cosine similarity reaches the configured
threshold. The LLM matcher uses the saved action or active-problem matching
prompt. Both count each matched prediction once within each reference group;
several predictions may match the same reference.

## Patient status

`acc` averages correctness within each stay and then across stays. `f1` is the
unweighted mean of per-class F1 over the union of reference and predicted labels,
including the inference-error sentinel when present. Available
`insufficient_data` remains a valid class.

## Acute problems

Acute problems are the `active_risks` of status predictions, scored on windows
with a nonempty available reference list. Each item is represented by its
`risk_name` (or the first available problem-name field); duplicate names are
removed without regard to case. All predicted problems are considered.

- `hit` is 1 when at least one predicted problem matches a reference.
- `recall` is distinct matched references divided by references.

## Recommended actions and harmful recommendations

Only the first `evaluation.top_k_actions` (5 in the published results) proposed
actions are considered:

- `hit` is 1 when at least one considered action matches a positive reference.
- `recall` is matched considered actions divided by positive references. It can
  exceed 1 when several actions match the same reference; no one-to-one
  assignment or clipping is applied.

The harmful recommendation rate `hrr` uses scored recommendation windows with
nonempty available reference red flags. Within each stay, it divides considered
actions matching a red flag by considered actions across those windows; the
ratios are averaged across stays with at least one considered action. A failure
does not invent a harmful action.

`hit`, `recall`, and `acc` are averaged within each stay and then across stays.

## Saved results

`metrics.json` contains summary metrics and coverage. Undefined JSON metrics are
null. `details.csv` stores per-sample scores. Matching results are cached in
`matches.jsonl` with the run's prediction and label fingerprints.
