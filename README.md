# RealICU-Bench [Neurips 2026]

<div align="center">

[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red?style=for-the-badge)](http://arxiv.org/abs/2605.13542)
[![Website](https://img.shields.io/badge/🌐%20Project%20Website-4285F4?style=for-the-badge)](https://chengzhi-leo.github.io/RealICU-Bench/)
[![Dataset](https://img.shields.io/badge/🗄️%20Dataset-Coming%20Soon-orange?style=for-the-badge)](https://github.com/chengzhi-leo/RealICU-Bench)

</div>

**RealICU: Do LLM Agents Understand Long-Context ICU Data? A Benchmark Beyond Behavior Imitation**

RealICU evaluates ICU reasoning against hindsight annotations rather than treating
recorded clinical actions as optimal decisions. RealICU-Gold contains 930 windows
from 94 ICU stays; RealICU-Scale contains 11,862 windows from the same cohort.

This repository provides prediction runners, evaluation, and the ICU-Evo
structured-memory agent. It supports four context modes: current events, full
history, retrieved history (RAG), and ICU-Evo memory.

| Task | Output | Scoring |
| --- | --- | --- |
| Patient Status | Overall clinical status | Accuracy, mean per-stay accuracy, macro F1 |
| Acute Problems | Active clinical risks | Auxiliary output; no independent scorer |
| Recommended Actions | Ordered treatment recommendations | Hit, precision, recall at K |
| Red Flag Actions | Actions to avoid | Harmful recommendation rate: proposed actions matched against reference red flags |

Gold and Scale share trajectories and are evaluation sets, not train/test splits.
Clinical data and experiment outputs are not included in this repository.

## 1. Install

Tested with Python 3.11. Run these commands from the repository root:

```bash
python -m pip install -e ".[openai]"
export OPENAI_API_KEY="your-api-key"
```

Use `.[anthropic]` or `.[google]` for another provider, `.[retrieval]` for RAG and
PubMedBERT matching. Extras can be combined, for example
`python -m pip install -e ".[openai,retrieval]"`.

Credentials come from environment variables. The package does not load `.env`
automatically. Credential names, provider settings, and Vertex settings are
described in [configuration](docs/configuration.md).

## 2. Prepare the dataset

Download the published RealICU data separately and preserve this layout:

```text
data/realicu/
├── trajectories/
│   ├── events.parquet
│   └── icu_stay.parquet
├── gold/labels/<stay_id>.json
└── scale/labels/<stay_id>.json
```

Set `data.root` and `data.label_set` in your prediction configuration. Examples
select one stay; set `data.stay_ids` to `null` for all stays. Evaluation uses
`labels_dir`, which must point to the corresponding `gold/labels` or
`scale/labels` directory. Only the published data format is supported.

## 3. Run a baseline

Edit `config/examples/patient_status.json` for your data, provider, model, and
input settings, and `config/examples/evaluate_status.json` for your labels:

```bash
python -m experiments.predict \
  --config config/examples/patient_status.json \
  --output runs/local

python -m evaluation.evaluate \
  --config config/examples/evaluate_status.json \
  --predictions runs/local/predictions.jsonl \
  --output runs/local/evaluation

cat runs/local/evaluation/metrics.json
```

Patient-status evaluation is local and needs no model. For recommendations,
install `.[retrieval]`, configure both recommendation examples, and run:

```bash
python -m experiments.predict \
  --config config/examples/recommendation.json \
  --output runs/recommendation

python -m evaluation.evaluate \
  --config config/examples/evaluate_recommendation.json \
  --predictions runs/recommendation/predictions.jsonl \
  --output runs/recommendation/evaluation
```

The recommendation evaluator uses PubMedBERT to match actions. Its first uncached
run may download the configured embedding model. An LLM matcher is also supported;
see [configuration](docs/configuration.md).

## 4. Run ICU-Evo

Set matching `data` and `input` settings in `memory.json` and `icu_evo.json`.
Build memory once, then predict from it:

```bash
python -m experiments.create_memory \
  --config config/examples/memory.json \
  --output runs/memory

python -m experiments.predict \
  --config config/examples/icu_evo.json \
  --memory-dir runs/memory \
  --output runs/icu_evo
```

Evaluate `runs/icu_evo/predictions.jsonl` using the same evaluator as the baseline.
Memory construction stores a snapshot at every observation window, including
unlabeled gaps and empty windows. Baselines and evaluation run independently of
ICU-Evo memory.

## Choose a context mode

| Prediction example | Context |
| --- | --- |
| `patient_status.json`, `recommendation.json` | Current window only |
| `full_history.json` | Current window and all preceding ICU events |
| `rag.json` | Current window and retrieved past windows |
| `icu_evo.json` | Precomputed ICU-Evo memory |

Every mode supports both task groups. For recommendations, set
`task="recommendation"`, `prediction.top_k_actions`, and
`prediction.horizon_hours` as in `recommendation.json`. The current-window mode
requires `input.include_pre_icu_data=false`. Other modes use the configured
pre-ICU history; ICU-Evo compresses it.

Current recorded actions are masked before recommendation prediction. Memory
recommendations use the previous snapshot plus masked current observations.
Labels, future events, and ICU outcomes are excluded from predictor inputs.
See [the evaluation protocol](docs/protocol.md) for exact masking, horizon,
availability, recovery, and scoring rules.

## Outputs and resume

```text
runs/local/
├── run_config.json
├── predictions.jsonl
├── summary.json
├── logs/                  enabled by logging.enabled
└── evaluation/
    ├── metrics.json
    ├── details.csv
    └── matches.jsonl      recommendation matching cache
```

Rerun the same command to resume. Completed prediction records, including
failures, and completed stay memories are retained. Use a new output directory
when changing configuration or retrying failed predictions.

To evaluate another model, write the standard
[prediction JSONL format](docs/predictions.md), then call the evaluator. It needs
only predictions, labels, and evaluation settings. To evaluate new K values
without repeating action matching, use a new output directory and
`--reuse-matches <previous-evaluation-directory>`.

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
evaluation/   Action matching and scoring
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
