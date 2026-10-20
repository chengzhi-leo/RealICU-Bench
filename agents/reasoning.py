"""Generate ICU-Evo episodes and insights."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from model.llms import LLMClient
from prompts.icu_evo_prompts import get_episode_agent_prompt, get_insight_agent_prompt


from agents.validation import (
    _chat_with_schema_validation,
    _safe_text,
    _validate_episode_output_schema,
    _validate_insight_output_schema,
)
from agents.memory import (
    CriticalEventsMemoryEntry,
    Episode,
    Insight,
    TrendMemoryEntry,
    WorkingWindow,
)
from agents.rendering import (
    _format_episode_input_lines,
    _format_patient_metadata_lines,
    _format_trend_block_lines,
    _render_prompt,
    _serialize_critical_event_strings,
)

class AssessmentAgent:
    def __init__(self, llm_client: LLMClient, schema_max_attempts: int):
        self.llm_client = llm_client
        self.schema_max_attempts = schema_max_attempts

    def analyze(
        self,
        raw_window_block: List[WorkingWindow],
        trend_block: List[TrendMemoryEntry],
        patient_metadata: Optional[Dict[str, Any]] = None,
        prior_episode_summary: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_episode_agent_prompt()
        patient_metadata_text = "\n".join(_format_patient_metadata_lines(patient_metadata))
        prior_episode_summary_text = _safe_text(prior_episode_summary) or "None"

        episode_input_lines = _format_episode_input_lines(raw_window_block, trend_block)
        episode_input_text = "\n".join(line for line in episode_input_lines if line is not None).strip() or "- None"

        if raw_window_block:
            first_window = raw_window_block[0]
            last_window = raw_window_block[-1]
            episode_start_time = f"hour {first_window.start_hour:.1f}"
            episode_end_time = f"hour {last_window.end_hour:.1f}"
            episode_duration = max(last_window.end_hour - first_window.start_hour, 0.0)
        else:
            episode_start_time = "N/A"
            episode_end_time = "N/A"
            episode_duration = 0.0

        prompt = _render_prompt(
            prompt_template,
            {
                "k": str(len(raw_window_block)),
                "duration": f"{episode_duration:.1f}",
                "episode_start_time": episode_start_time,
                "episode_end_time": episode_end_time,
                "patient_metadata": patient_metadata_text,
                "episode_input": episode_input_text,
                "prior_episode_summary": prior_episode_summary_text,
                "previous_episode_summary": prior_episode_summary_text,
                "prior_episode_summary_text": prior_episode_summary_text,
            },
        )

        parsed, raw, usage, parse_error, attempt_index = _chat_with_schema_validation(
            llm_client=self.llm_client,
            prompt=prompt,
            schema_validator=_validate_episode_output_schema,
            max_attempts=self.schema_max_attempts,
        )
        if parse_error is not None:
            parse_error = f"{parse_error} | final_attempt={attempt_index}/{self.schema_max_attempts}"

        return parsed, raw, usage, prompt, parse_error


class InsightAgent:
    def __init__(self, llm_client: LLMClient, schema_max_attempts: int):
        self.llm_client = llm_client
        self.schema_max_attempts = schema_max_attempts

    def analyze(
        self,
        current_insights: List[Insight],
        episode: Episode,
        critical_events_entry: Optional[CriticalEventsMemoryEntry],
        trend_block: List[TrendMemoryEntry],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_insight_agent_prompt()

        if current_insights:
            insights_text = "\n".join(
                [f"[{insight.insight_id}] {_safe_text(insight.hypothesis)}" for insight in current_insights]
            )
        else:
            insights_text = "- None"

        episode_summary_text = (
            f"Window {episode.start_window}-{episode.end_window} "
            f"(hour {episode.start_hour:.1f}-{episode.end_hour:.1f}): {episode.episode_summary}"
        )
        critical_events_text = "- None"
        if critical_events_entry is not None and critical_events_entry.critical_events:
            critical_events_lines = _serialize_critical_event_strings(critical_events_entry.critical_events)
            if critical_events_lines:
                critical_events_text = "\n".join(critical_events_lines)

        trend_lines = _format_trend_block_lines(trend_block)
        vital_trends_text = "\n".join(line for line in trend_lines if line is not None).strip() or "- None"
        patient_metadata_text = "\n".join(_format_patient_metadata_lines(patient_metadata))

        prompt = _render_prompt(
            prompt_template,
            {
                "hypothesis_bank": insights_text,
                "patient_metadata": patient_metadata_text,
                "episode_summary": episode_summary_text,
                "critical_events": critical_events_text,
                "vital_trends": vital_trends_text,
                "observations": vital_trends_text,
            },
        )

        parsed, raw, usage, parse_error, attempt_index = _chat_with_schema_validation(
            llm_client=self.llm_client,
            prompt=prompt,
            schema_validator=_validate_insight_output_schema,
            max_attempts=self.schema_max_attempts,
        )
        if parse_error is not None:
            parse_error = f"{parse_error} | final_attempt={attempt_index}/{self.schema_max_attempts}"

        return parsed, raw, usage, prompt, parse_error
