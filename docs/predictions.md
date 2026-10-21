# Prediction contract

The public interface is UTF-8 JSON Lines: one object per `(sample_id, task)`.
Copy `sample_id` exactly from the selected published annotation set. Gold and
Scale identifiers are distinct even when observation times coincide. Duplicate
identifiers, unknown identifiers, and malformed records fail evaluation.

Two task groups correspond to the two prediction prompts:

| `task` | Required contents of `prediction` | Scored outputs |
| --- | --- | --- |
| `patient_status` | `patient_assessment.overall.label` and list-valued `patient_assessment.active_risks` | Overall patient status (`task="patient_status"`) and acute problems (`task="active_risks"`) |
| `recommendation` | List-valued `recommended_actions` and `red_flags` | Recommended actions, including matches against reference red flags |

A patient-status record:

```json
{"sample_id":"gold/10001884_37510196/0.000000|0.500000","task":"patient_status","prediction":{"patient_assessment":{"overall":{"label":"stable","rationale":"Observed clinical course."},"active_risks":[]}},"error":null}
```

Valid status labels are `stable`, `improving`, `deteriorating`, and
`insufficient_data`. Each active risk is an object with a `risk_name`, as in the
prediction prompt, or a plain string. The evaluator independently checks
reference availability.

A recommendation record:

```json
{"sample_id":"gold/10001884_37510196/0.000000|0.500000","task":"recommendation","prediction":{"recommended_actions":[{"action_name":"Increase oxygen supplementation","action_description":"Treat the observed hypoxemia."}],"red_flags":[]},"error":null}
```

Each recommended action requires a nonempty `action_name`. List order determines
which actions fall within K. Descriptions are optional matching context. Use
`[]` for an available empty output, not null. The model's predicted `red_flags`
list is retained as auxiliary output; current harmful-recommendation scoring
matches proposed recommendations against the dataset's reference red flags.

Represent an inference or output-parsing failure explicitly:

```json
{"sample_id":"gold/10001884_37510196/0.000000|0.500000","task":"patient_status","prediction":null,"error":{"type":"TimeoutError","message":"The model request timed out."}}
```

Failed predictions are scored as failures; they are not silently removed. Omitted
samples are reported separately as missing predictions in coverage statistics.

Optional `usage` records token consumption and is not needed for scoring.
Inputs, reference answers, trajectory paths, and memory paths are not part of
the prediction contract.

A runner-generated file has a neighboring `run_config.json`. The evaluator checks
its task and protocol against evaluation settings. External models can supply
just the JSONL file; recommendation evaluation considers the first
`evaluation.top_k_actions` actions of each record.

# Run artifacts

```text
runs/<run_name>/
    run_config.json
    predictions.jsonl
    summary.json
    logs/                       optional prompts and model responses
    evaluation/
        run_config.json
        metrics.json
        details.csv
        matches.jsonl           recommendation matching cache
```

ICU-Evo builders instead write `memory/<stay_id>.json` inside their own run.
Prediction runs reference that directory through `--memory-dir` and can use a
subset of its stays. The data root, annotation set, and observation input settings
must match; the prediction model can differ from the memory-building model.

Rerun an unchanged command to resume. Prediction records, including failures,
remain fixed. Existing outputs are never reused under different settings. To
retry failed samples, create a new prediction run or explicitly prepare a new
external JSONL file containing the desired replacement records.
