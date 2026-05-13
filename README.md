# RealICU-Bench: Do LLM Agents Understand Long-Context ICU Data?
### A Benchmark Beyond Behavior Imitation

<div align="center">

[![Paper](https://img.shields.io/badge/📄%20Paper-Coming%20Soon-red?style=for-the-badge)](https://github.com/chengzhi-leo/RealICU-Bench)
[![Website](https://img.shields.io/badge/🌐%20Project%20Website-4285F4?style=for-the-badge)](https://chengzhi-leo.github.io/RealICU-Bench/)
[![Dataset](https://img.shields.io/badge/🗄️%20Dataset-Coming%20Soon-orange?style=for-the-badge)](https://github.com/chengzhi-leo/RealICU-Bench)

</div>

---

## Overview

Intensive care units (ICU) generate long, dense and evolving streams of clinical information, where physicians must repeatedly reassess patient states under time pressure, underscoring a clear need for reliable AI decision support. Existing ICU benchmarks typically treat historical clinician actions as ground truth. However, these actions are made **under incomplete information and limited temporal context of the underlying patient state**, and may therefore be suboptimal, making it difficult to assess the true reasoning capabilities of AI systems.

**RealICU** is a hindsight-annotated benchmark for evaluating large language models (LLMs) under realistic ICU conditions, where labels are created after senior physicians review **the full patient trajectory using hindsight**. We formulate four physician-motivated tasks — in consultations with over 30 board-certified clinicians — and release two datasets:

| Dataset | Windows | Patients | Labels |
|---|---|---|---|
| **RealICU-Gold** | 930 | 94 | Physician-consensus annotations from five senior ICU physicians |
| **RealICU-Scale** | 11,862 | — | Extended by *Oracle*, a physician-validated LLM hindsight labeler |

### Four Physician-Motivated Tasks

| Task | Description |
|---|---|
| 🩺 **Patient Status** | Assess the current overall status of the patient based on all available clinical data |
| 🔍 **Acute Problems** | Identify acute clinical problems requiring continuous attention and monitoring in future bedside care |
| 💊 **Recommended Actions** | Propose short-term treatment recommendations appropriate for the patient's condition |
| ⚠️ **Red Flag Actions** | Identify actions that should be avoided for this specific patient that might cause unsafe outcomes |

### ICU-Evo Agent

We further introduce **ICU-Evo** to study structured-memory agents that improves long-horizon reasoning but does not fully eliminate safety failures. It is an observation-grounded multi-agent pipeline with dynamic memory that organizes clinical context into heterogeneous memory types aligned with how clinicians reason.

---

## Repo Layout

```text
RealICU-Bench/
├── agents/                  # ICU-Evo multi-agent pipeline
├── config/                  # Runtime config & observation schema
├── data/                    # Data placeholders (see Data Layout below)
├── docs/                    # GitHub Pages website source
├── evaluation/              # Evaluation scripts & PubMedBERT matcher
├── experiments/             # Memory creation & prediction experiments
├── model/                   # Unified LLM client (OpenAI / Anthropic / Google)
├── prompts/                 # Prompt templates for all pipeline stages
├── utils/                   # Shared helpers (formatting, scoring, RAG)
├── data_parser.py           # ICU trajectory builder from parquet
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
conda create -n realicu python=3.10 -y
conda activate realicu
pip install -r requirements.txt
```

### 2. Set an LLM provider API key

```bash
export OPENAI_API_KEY=...
# or
export ANTHROPIC_API_KEY=...
# or
export GOOGLE_API_KEY=...
```

> You can also copy `.env.example` to `.env` and fill in your keys — `python-dotenv` will pick them up automatically.

---

## Data Layout

Place parquet-readable ICU data under `data/`. The default config expects:

```text
data/mimic-demo/events       # ICU event records (vitals, labs, medications, …)
data/mimic-demo/icu_stay     # ICU stay metadata
```

Each path can be a single `.parquet` file or a directory of parquet shards — `pandas.read_parquet(...)` handles both. If your data lives elsewhere, edit `config/config.json`.

---

## Pipeline

### Block 1 — Build ICU-Evo Memory

Prepare a CSV listing the ICU stays to process:

```csv
subject_id,icu_stay_id
10006,20001
10007,20002
```

Run memory construction:

```bash
python experiments/create_memory.py \
  --patient-stay-ids path/to/patient_stay_ids.csv \
  --num-workers 4 \
  --llm-provider google \
  --llm-model qwen/qwen3-235b-a22b-instruct-2507-maas
```

Outputs are written to:

```text
<memory_dir>/
```

To resume an interrupted run:

```bash
python experiments/create_memory.py \
  --patient-stay-ids path/to/patient_stay_ids.csv \
  --resume-run <memory_dir>
```

---

### Block 2 — Run Predictions

Both scripts require an ICU-Evo memory run and an oracle results directory. **Supported context modes:** `memory` · `full_history_events` · `local_events_only` · `rag_history_events` · `all`

#### Patient Status & Acute Problems

`patient_status_experiment.py` predicts both **Patient Status** and **Acute Problems** in a single run.

```bash
# Single context mode
python experiments/patient_status_experiment.py \
  --memory-run <memory_dir> \
  --oracle-results-dir <oracle_dir> \
  --context-mode memory \
  --snapshot-stride 8

# All context modes in one pass
python experiments/patient_status_experiment.py \
  --memory-run <memory_dir> \
  --oracle-results-dir <oracle_dir> \
  --context-mode all
```

#### Recommended Actions

```bash
# Single context mode
python experiments/recommendation_experiment.py \
  --memory-run <memory_dir> \
  --oracle-results-dir <oracle_dir> \
  --context-mode memory \
  --snapshot-stride 8 \
  --top-k-actions 5 \
  --prediction-horizon-hours 1

# All context modes
python experiments/recommendation_experiment.py \
  --memory-run <memory_dir> \
  --oracle-results-dir <oracle_dir> \
  --context-mode all
```

---

### Block 3 — Evaluate

#### Patient Status & Acute Problems

```bash
# Single mode
python evaluation/eval_patient_status.py \
  --icu-evo-dir <memory_dir>/patient_status/memory \
  --gt-dir <oracle_dir>

# All modes (after running --context-mode all)
python evaluation/eval_patient_status.py \
  --icu-evo-dir <memory_dir>/patient_status
```

#### Recommended Actions

```bash
python evaluation/eval_recommendation.py \
  --prediction-dir <memory_dir>/recommendation_experiment \
  --gt-dir <oracle_dir>
```

The default matcher backend is **PubMedBERT**. To specify it explicitly:

```bash
python evaluation/eval_recommendation.py \
  --prediction-dir <memory_dir>/recommendation_experiment \
  --gt-dir <oracle_dir> \
  --matcher-backend pubmedbert
```

---

## Notes

- **Oracle labels** — Oracle label generation is not included in this release. Evaluation assumes those labels already exist under `oracle_results/`.
- **RAG mode** — `rag_history_events` requires `sentence-transformers` and an embedding model available locally or via Hugging Face.
- **Artifacts** — Run artifacts are written under `experiment_results/` and evaluation results under `evaluation_results/`.

---

<div align="center">

If you find this work useful, please consider citing our paper (coming soon).

[![Website](https://img.shields.io/badge/🌐%20Visit%20Project%20Website-4285F4?style=flat-square)](https://chengzhi-leo.github.io/RealICU-Bench/)

</div>
