# Configuration

Run one task and one context mode per prediction configuration. All experiment
parameters must be supplied explicitly; examples are starting points, not
implicit defaults. Paths are relative to the command's working directory.

## Examples

| File in `config/examples/` | Workflow |
| --- | --- |
| `patient_status.json` | Current-window patient-status prediction |
| `recommendation.json` | Current-window action recommendations |
| `full_history.json` | Patient status from full ICU history |
| `rag.json` | Patient status with retrieved past windows |
| `memory.json` | ICU-Evo memory construction |
| `icu_evo.json` | Prediction from saved ICU-Evo memory |
| `evaluate_status.json` | Patient-status scoring |
| `evaluate_recommendation.json` | Action matching and recommendation scoring |

## Data and observations

- `data.root`: published dataset root; `data.label_set`: `gold` or `scale`.
- `data.stay_ids`: explicit stay IDs, or `null` for all published stays.
- `input.current_window_hours`: cadence matching the published observation windows.
- `input.observation_hours`: observation cap, or `null` for all available windows.
- `input.include_pre_icu_data`: whether to include the chosen pre-ICU history.
- `input.pre_icu_history_hours`: pre-admission lookback; `null` retains all history,
  zero retains none.
- `input.excluded_event_codes`: event codes removed before context construction.
- `prediction.snapshot_stride`: select every Nth eligible sample and retain the
  final eligible sample.
- `prediction.top_k_actions` and `prediction.horizon_hours`: required for
  recommendation prediction. Match the evaluation horizon to the prediction.

Memory construction and memory prediction must use matching data roots,
annotation sets, and input settings. Prediction can select a subset of the
memory's stays. Episode cadence, insight cadence, memory limits, schema retry
count, and pre-ICU summary length are explicit in `icu_evo`.

## Model providers

`llm` specifies the provider, model, temperature, token limits, token encoding,
retry policy, timeout, and connection settings. Credentials are read from the
process environment:

| Provider | API credential | Installation extra |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `openai` |
| `anthropic` | `ANTHROPIC_API_KEY` | `anthropic` |
| `google` | `GOOGLE_API_KEY` | `google` |

For API access, use `connection.backend="api"`. OpenAI-compatible services can
set `connection.base_url`; use `null` for the provider's normal endpoint.
For Google or Anthropic on Vertex, use `connection.backend="vertex"` with explicit
`project` and `location`, and authenticate with application-default credentials.
Vertex does not require the provider API key.

For `gpt-5*` and `o1*`, the OpenAI client preserves the earlier request format:
`max_completion_tokens` and no temperature parameter. Other models use the
explicit configured temperature and `max_tokens`.

## Retrieval and action matching

RAG uses `rag.top_k_windows` and `rag.encoder`. Embedding action matching uses
`matcher.backend="pubmedbert"`, `matcher.encoder`, and
`matcher.similarity_threshold`. Both encoders require a model name, device,
batch size, and `local_files_only`. Set the latter to `true` to require local
weights, or `false` to allow the embedding library to download them.

For LLM action matching, set `matcher.backend="llm"` and replace the encoder and
threshold settings with a full `matcher.llm` configuration. The earlier matcher
used temperature zero; supply `temperature: 0` to reproduce it. Completed matches
are cached. A fully cached evaluation needs no API credentials.

