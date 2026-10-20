# RealICU protocol `realicu-v1`

This version records the existing release's masking, sampling, horizon, and
metric definitions. The machine-readable declaration is
`config/protocols/realicu-v1.json`. Experimental choices such as K, lookback,
stride, model, and horizon remain explicit run settings.

## Observation windows

Join trajectories with `stay_id`; an event's encounter-specific `icu_stay_id`
can refer to a historical encounter. Preserve `source_event_index` and
`icu_event_index`. Select events in the actual half-open interval
`[window_start_time, window_end_time)` from published timestamps. Nominal hours
and nonconsecutive window indices do not establish actual endpoints.

Memory and baseline runners create the same chronological observation sequence,
including unlabeled gaps and empty windows. The configured cadence must agree
with published windows. Short final intervals retain their actual endpoints.
Observation caps include only complete published evaluation windows.

Current-window baselines see only current observations. Full-history baselines
see past ICU windows plus pre-ICU history if explicitly enabled. RAG selects past
ICU windows using the current observed events as query; future and current
windows are excluded from retrieval candidates. Memory mode uses ICU-Evo's
compressed history and memory structures.

The current-window mode requires pre-ICU input to be disabled. Full-history and
RAG contexts include the chosen pre-ICU history when enabled; memory mode uses
its compressed version. The pre-ICU lookback is measured from ICU admission. Null retains all available
history; zero retains none. Exclusions apply before context construction.
Window selection by timestamp does not establish when every source clinical
note became available.

## Recommendation masking and horizon

Before recommendation prediction, remove current observations whose codes are
`DRUG_START`, `DRUG_PRESCRIPTION`, `PROCEDURE`, `BODY_INPUT`, or `TRANSFER`.
Historical recorded actions remain visible. In memory mode, combine the previous
snapshot with masked current raw observations; do not expose summaries generated
from unmasked current actions. The first window starts with empty prior memory.

Positive references are recorded actions labeled `acceptable` or
`best_practice`. For horizon H and current window start S, include positive
references from annotation windows whose actual start is in **[S, S + H]**, with
both endpoints included. This preserves the earlier implementation's
window-start rule; it is not filtering individual action timestamps to a
half-open future interval.

Unavailable recommended-action references at the current window are not scored,
even if later windows have available references. Available empty lists contain
no retained items and are not treated as unavailable. Red-flag references use
only the current window and their own availability flag.

## Sampling and failures

Sort eligible windows chronologically, select every configured k-th window from
the first, and retain the final eligible window. Status prediction may also
produce acute problems for a window with unavailable status; status scoring
excludes that window. Recommendation prediction runs where either action or
red-flag references are available.

Inference and parse errors produce explicit failed prediction records. Status
failures receive an `inference_error` predicted label and count as incorrect.
Recommendation failures provide zero recommended actions and therefore zero
hit/precision/recall when positive references exist. A full-history context-limit
failure prevents further inference on that stay; later selected windows remain
explicit failures. Missing predictions are reported in coverage, not silently
replaced by inferred labels.

Malformed data or configuration fail the run. ICU-Evo retains the earlier
schema-recovery behavior after its explicitly configured attempt limit. Failed
pre-ICU compression leaves the history uncompressed and omits its summary from
memory metadata. Failed episode generation produces the original deterministic
window/trend summary with event references. Failed insight generation retains
the original evidence-based fallback hypothesis. Model-supplied episodes with
no valid supporting IDs use the first twelve available block event IDs. These
are model-output recovery rules, not defaults for missing configuration.

## Patient status

Window accuracy averages correctness over scored windows. Mean per-stay accuracy
first averages within each stay and then across stays. Macro F1 is the unweighted
mean of per-class F1 over the union of reference and predicted labels, including
the inference-error sentinel when present. Available `insufficient_data` remains
a valid class. There is no separate acute-problem score in this release.

## Recommended actions and harmful recommendations

The embedding matcher selects the highest-similarity reference for each proposed
action and retains it when cosine similarity reaches the configured threshold.
The LLM matcher uses the saved action-matching prompt. Both count each matched
prediction once within each reference group; several predictions may match the
same reference.

For the first min(K, number of predictions) proposed actions:

- Hit is 1 when at least one proposed action matches a positive reference.
- Precision is matched prediction count divided by considered prediction count;
  it is zero when no actions were proposed and positive references exist.
- Recall is matched prediction count divided by positive reference count.

Recall preserves the earlier implementation's numerator. It can exceed 1 when
multiple proposed actions match the same reference; no implicit one-to-one
assignment or clipping is applied. Windows without positive references have
undefined recommendation scores and are excluded from recommendation means.

Names beginning with `micro_` preserve the earlier release's window-average
metrics. Names beginning with `macro_` first average within each stay and then
across stays. They are not ratios of globally pooled match counts.

Red-flag scoring considers only windows with nonempty available reference red
flags. Harmful recommendation rate first divides matched harmful predictions by
considered predictions across those windows within each stay, then averages the
ratios across stays with at least one considered prediction. A failure does not
invent a harmful action. Available empty and unavailable red-flag lists do not
establish that all proposed actions are safe and do not contribute to this rate.

## Saved results

`metrics.json` contains summary metrics and coverage. Undefined JSON metrics are
null. `details.csv` stores per-sample scores, with one row per K for
recommendations. Matching results are cached in `matches.jsonl` with the run's
prediction and label fingerprints.

