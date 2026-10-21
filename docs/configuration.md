# Configuration

Run one task and one context mode per prediction configuration. All experiment
parameters must be supplied explicitly; examples are starting points, not
implicit defaults. Paths are relative to the command's working directory.

## Examples

| File in `config/examples/` | Workflow |
| --- | --- |
| `local_status.json`, `local_recommendation.json` | Prediction from the current window only |
| `full_history_status.json`, `full_history_recommendation.json` | Prediction from all preceding ICU events |
| `rag_status.json`, `rag_recommendation.json` | Prediction with retrieved past windows |
| `icu_evo_memory.json` | ICU-Evo memory construction |
| `icu_evo_status.json`, `icu_evo_recommendation.json` | Prediction from saved ICU-Evo memory |
| `evaluate_status.json` | Patient-status scoring |
| `evaluate_active_risks.json` | Acute-problem matching and scoring of status predictions |
| `evaluate_recommendation.json` | Action matching and recommendation scoring |

## Data and observations

- `data.root`: published dataset root; `data.label_set`: `gold` or `scale`.
- `data.stay_ids`: explicit stay IDs, or `null` for all published stays.
- `input.current_window_hours`: cadence matching the published observation windows.
- `input.observation_hours`: observation cap, or `null` for all available windows.
- `input.include_pre_icu_data`: whether to include the chosen pre-ICU history.
- `input.pre_icu_history_hours`: pre-admission event lookback; zero retains no events.
- `input.pre_icu_discharge_summaries`: number of most recent earlier discharge
  summaries appended to the pre-ICU history; the published runs used 2.
- `input.excluded_event_codes`: event codes removed from ICU observation windows.
- `prediction.snapshot_stride`: select every Nth eligible sample and retain the
  final eligible sample.
- `prediction.top_k_actions` and `prediction.horizon_hours`: required for
  recommendation prediction. The horizon appears in the prompt and selects
  windows with a positive reference within it; scoring uses current-window
  references.

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

## Retrieval and matching

RAG uses `rag.top_k_windows` and `rag.encoder`. Embedding matching of actions and
acute problems uses `matcher.backend="pubmedbert"`, `matcher.encoder`, and
`matcher.similarity_threshold`; the published results used a threshold of 0.5.
Both encoders require a model name, device, batch size, and `local_files_only`.
Set the latter to `true` to require local weights, or `false` to allow the
embedding library to download them.

For LLM matching, set `matcher.backend="llm"` and replace the encoder and
threshold settings with a full `matcher.llm` configuration. The earlier action
matcher used temperature zero and the earlier acute-problem matcher temperature
one. Completed matches are cached. A fully cached evaluation needs no API
credentials.

