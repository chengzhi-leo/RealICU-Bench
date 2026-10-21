# RealICU: Do LLM Agents Understand Long-Context ICU Data? A Benchmark Beyond Behavior Imitation [NeurIPS 2026]

🎉🎉🎉 Updates: RealICU has been accepted at NeurIPS 2026 

<div align="center">

[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red?style=for-the-badge)](http://arxiv.org/abs/2605.13542)
[![Website](https://img.shields.io/badge/🌐%20Project%20Website-4285F4?style=for-the-badge)](https://chengzhi-leo.github.io/RealICU-Bench/)
[![Dataset](https://img.shields.io/badge/🗄️%20Dataset-Coming%20Soon-orange?style=for-the-badge)](https://github.com/chengzhi-leo/RealICU-Bench)

</div>



RealICU evaluates ICU reasoning against hindsight annotations rather than treating
recorded clinical actions as optimal decisions. RealICU-Gold contains 930 windows
from 94 ICU stays; RealICU-Scale contains 11,862 windows from the same cohort.

This repository provides prediction runners, evaluation, and the ICU-Evo
structured-memory agent. It supports four context modes: current local window, full
history, retrieved history (RAG), and ICU-Evo memory.

| Task | Output | Scoring |
| --- | --- | --- |
| Patient Status | Overall clinical status | Accuracy, F1 |
| Acute Problems | Active clinical risks | Hit, Recall |
| Recommended Actions | Top-5 ordered treatment recommendations | Hit, Recall |
| Red Flag Actions | Actions to avoid | Harmful recommendation rate (HRR) |

Accuracy, hit, recall, and HRR are averaged within each stay and then across
stays; F1 is the macro F1 over status classes. `metrics.json` reports them as
`acc`, `f1`, `hit`, `recall`, and `hrr`.

RealICU-Gold and RealICU-Scale share MIMIC-IV based patient trajectories and are evaluation sets only.


## 1. Install

Run these commands from the repository root:

```bash
python -m pip install -e ".[openai,retrieval]"
export OPENAI_API_KEY="your-api-key"
```

`retrieval` provides the PubMedBERT encoder used by RAG and by acute-problem and
action matching. Use `anthropic` or `google` instead of `openai` for another
provider. Credentials come from environment variables; the package does not load
`.env` automatically. Credential names, provider settings, and Vertex settings
are described in [configuration](docs/configuration.md).

## 2. Prepare the dataset

RealICU uses MIMIC-IV derived data. To access it, visit PhysioNet, complete the
[CITI training](https://www.citiprogram.org/), and request access to the MIMIC-IV
database. After obtaining access, download the RealICU data and preserve this
layout:

```text
data/realicu/
├── trajectories/
│   ├── events.parquet
│   └── icu_stay.parquet
├── gold/labels/<stay_id>.json
└── scale/labels/<stay_id>.json
```

Only the published data format is supported.

## 3. Run an experiment

An experiment is one prediction run followed by evaluation. `experiments.predict`
is the single entry point; its configuration selects the context mode and the
task. Choose a mode:

| Mode | Model input | Configuration in `config/examples/` | Before prediction |
| --- | --- | --- | --- |
| Local window | Current window only | `local_<task>.json` | — |
| Full history | Current window and all preceding ICU events | `full_history_<task>.json` | — |
| Retrieval (RAG) | Current window and the most similar past windows | `rag_<task>.json` | — |
| ICU-Evo | ICU-Evo evolving memory | `icu_evo_<task>.json` | Build memory once |

`<task>` is `status` (patient status and acute problems) or `recommendation`
(recommended actions and red flags). Before running, edit `data` and `llm` in the
chosen configuration: the examples select one gold stay (`data.stay_ids`; use
`null` for all stays) and an example model.

**Step 1, ICU-Evo only: build memory.** Build it once and reuse it for both tasks
and for any prediction model:

```bash
python -m experiments.create_memory \
  --config config/examples/icu_evo_memory.json \
  --output runs/memory
```

**Step 2: predict.**

```bash
python -m experiments.predict \
  --config config/examples/rag_status.json \
  --output runs/rag_status
```

For ICU-Evo, also pass the memory:

```bash
python -m experiments.predict \
  --config config/examples/icu_evo_status.json \
  --memory-dir runs/memory \
  --output runs/icu_evo_status
```

**Step 3: evaluate.** Choose the evaluation configuration by task and set its
`labels_dir` to the label set used for prediction:

| Prediction | Evaluation configuration | Scores |
| --- | --- | --- |
| `*_status` | `evaluate_status.json` | Patient status |
| `*_status` | `evaluate_active_risks.json` | Acute problems |
| `*_recommendation` | `evaluate_recommendation.json` | Recommended actions and HRR |

```bash
python -m evaluation.evaluate \
  --config config/examples/evaluate_status.json \
  --predictions runs/rag_status/predictions.jsonl \
  --output runs/rag_status/evaluation

cat runs/rag_status/evaluation/metrics.json
```

Patient-status scoring needs no model. Acute problems and actions are matched to
the references with PubMedBERT, which may be downloaded on first use; an LLM
matcher is also supported.

### What the modes see

- Labels, future events, and ICU outcomes are never model inputs. Before
  recommendation, the current window's recorded actions are masked; ICU-Evo
  combines the previous memory snapshot with the masked current window.
- Pre-ICU history consists of the lookback events and the most recent earlier
  discharge summaries. ICU-Evo compresses it into its memory; the local-window
  mode excludes it, and the full-history and RAG examples disable it as in the
  paper.
- Status is predicted for every labeled window. Recommendations are predicted
  for reviewed windows after the first observation window with a positive
  reference within `prediction.horizon_hours`, and scored against the current
  window's references.
- Full-history prompts of long stays can exceed `llm.max_input_tokens`; the
  remaining windows of that stay are then recorded as failures.

See [the evaluation protocol](docs/protocol.md) for exact masking, sampling,
recovery, and scoring rules.

## Outputs and resume

```text
runs/rag_status/
├── run_config.json
├── predictions.jsonl
├── summary.json
├── logs/                  enabled by logging.enabled
└── evaluation/
    ├── metrics.json
    ├── details.csv
    └── matches.jsonl      acute-problem or action matching cache
```

Rerun the same command to resume. Completed prediction records, including
failures, and completed stay memories are retained. Use a new output directory
when changing configuration or retrying failed predictions.

To evaluate another model, write the standard
[prediction JSONL format](docs/predictions.md) and run Step 3. To rescore with
another `evaluation.top_k_actions` without repeating action matching, use a new
output directory and `--reuse-matches <previous-evaluation-directory>`.

## Configuration and repository layout

All experiment settings are explicit. Missing required settings fail with an
error; example values are illustrative choices. Paths in configuration are
relative to the working directory. The three installed commands
`realicu-memory`, `realicu-predict`, and `realicu-evaluate` expose
the same interfaces as the module commands above.

```text
agents/       ICU-Evo observations, reasoning, and memory
experiments/  Memory/prediction runners and context construction
dataset/      Published data loading and window construction
evaluation/   Matching and scoring
model/        Provider clients and text embeddings
prompts/      Predictor, memory, and action-matcher prompts
config/       Run examples, observation settings, and evaluation protocol
docs/         Format/protocol documentation and project website
```

`pyproject.toml` defines installation dependencies and command-line entrypoints.


## License

Copyright 2026 RealICU-Bench contributors.

The project's code, prompts, configurations, and runner documentation are
licensed under the [Apache License 2.0](LICENSE). Clinical data are distributed
separately under their applicable terms. The project website (`docs/index.html`
and `docs/static/`) retains its existing CC BY-SA 4.0 attribution and third-party
asset licenses.

## Citation

```bibtex
@article{shen2026realicu,
  title={RealICU: Do LLM Agents Understand Long-Context ICU Data? A Benchmark Beyond Behavior Imitation},
  author={Shen, Chengzhi and Shen, Weixiang and Susetzky, Tobias and Li, Jun and Liu, Yuyuan and Zhang, Xuepeng and Gong, Zhenyu and Rueckert, Daniel and Pan, Jiazhen and others},
  journal={arXiv preprint arXiv:2605.13542},
  year={2026}
}
```
